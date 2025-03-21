import os
import sys
import math
import pprint
from itertools import islice
import time
import copy
import random
import itertools
import pickle
from sklearn.model_selection import KFold
import numpy as np
import copy
import torch
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support
)

import torch
import torch_geometric as pyg
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

# If your model structure supports it, you can import checkpointing:
# from torch.utils.checkpoint import checkpoint

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import tasks, util
from ultra.models import Ultra
from tqdm import tqdm

import datetime

separator = ">" * 30
line = "-" * 30

import torch.nn as nn

def get_hard_negatives(model, data_gpu, patient_id_tensor, all_drug_indices, device, n_hard, relation_dict_f):
    """Get the hardest negative samples based on model predictions."""
    model.eval()
    with torch.no_grad():
        all_drug_indices = all_drug_indices.to(device)
        batch_size = 128
        all_scores = []
        
        for start_idx in range(0, len(all_drug_indices), batch_size):
            end_idx = min(start_idx + batch_size, len(all_drug_indices))
            batch_drugs = all_drug_indices[start_idx:end_idx].to(device)
            
            pairs = torch.cat([
                patient_id_tensor.repeat(len(batch_drugs), 1).to(device),
                batch_drugs.view(-1, 1).to(device),
                torch.full((len(batch_drugs), 1), relation_dict_f["TREAT"], device=device)
            ], dim=1).unsqueeze(1)
            
            scores = model(data_gpu, pairs).squeeze()
            all_scores.append(scores)
            
        all_scores = torch.cat(all_scores, dim=0)
        _, hard_negative_indices = torch.topk(all_scores, k=min(n_hard, len(all_scores)))
        hard_negatives = all_drug_indices[hard_negative_indices]
        
    model.train()
    return hard_negatives.to(device)

def get_hard_negatives_batched(model, data_gpu, patient_id_tensor, all_drug_indices, device, n_hard, relation_dict_f, batch_size=64):
    """Memory efficient hard negative mining with smaller batches."""
    model.eval()
    with torch.no_grad():
        # Process in smaller chunks to save memory
        all_scores = []
        treat_id = relation_dict_f["TREAT"]
        
        # Process drugs in chunks
        for start_idx in range(0, len(all_drug_indices), batch_size):
            end_idx = min(start_idx + batch_size, len(all_drug_indices))
            current_batch_size = end_idx - start_idx
            
            # Create batch components
            drug_batch = all_drug_indices[start_idx:end_idx].view(-1, 1)
            patient_batch = patient_id_tensor.repeat(current_batch_size)
            treat_batch = torch.full((current_batch_size, 1), treat_id, device=device)
            
            # Create pairs
            pairs = torch.cat([
                patient_batch.view(-1, 1),
                drug_batch,
                treat_batch
            ], dim=1).unsqueeze(1)
            
            # Get scores
            scores = model(data_gpu, pairs).squeeze()
            all_scores.append(scores)
            
            # Clear memory
            del pairs
            torch.cuda.empty_cache()
        
        # Combine scores and get top k
        all_scores = torch.cat(all_scores)
        _, hard_negative_indices = torch.topk(all_scores, k=min(n_hard, len(all_scores)))
        hard_negatives = all_drug_indices[hard_negative_indices]
        
        # Clear memory
        del all_scores
        torch.cuda.empty_cache()
        
    model.train()
    return hard_negatives


def train_with_hard_negatives_efficient(
    cfg, model, train_data_cpu, valid_data_cpu, filtered_data_cpu,
    device, batch_per_epoch, logger, val_drugs, relation_dict_f,
    entity_dict_f, use_personalized=False, patient_data_fn=None
):
    # Initialize criterion and optimizer
    if use_personalized:
        try:
            prs_values, expr_values, disease_names, protein_names = load_patient_data(patient_data_fn, device=device)
            prs_values = torch.tensor(prs_values, device=device)
            expr_values = torch.tensor(expr_values, device=device)
            
            # Normalize values
            prs_values = (prs_values - prs_values.mean()) / (prs_values.std() + 1e-8)
            expr_values = (expr_values - expr_values.mean()) / (expr_values.std() + 1e-8)

            criterion = PersonalizedDrugLoss(
                disease_names=disease_names,
                protein_names=protein_names,
                lambda1=0.01,
                lambda2=0.01
            ).to(device)
        except Exception as e:
            raise Exception(f"Error setting up personalized learning: {str(e)}")
    else:
        criterion = BPRLoss(margin=1.5).to(device)

    # Training setup
    best_val_mrr = float('-inf')
    best_model_state = copy.deepcopy(model.state_dict())
    patience = 10
    epochs_no_improve = 0
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True, threshold=1e-3
    )
    
    # Pre-compute static data
    # Pre-compute static data
    logger.warning("Preparing data...")
    data_gpu = move_data_to_gpu(train_data_cpu, device)
    all_drug_indices = get_drug_node_indices(train_data_cpu, relation_dict_f).to(device)
    patient_id = get_patient_node_id(train_data_cpu, relation_dict_f).item()
    patient_id_tensor = torch.tensor([patient_id], dtype=torch.long, device=device)
    val_drug_indices = torch.tensor([
        entity_dict_f[drug] for drug in val_drugs if drug in entity_dict_f
    ], device=device)
    
    # Training hyperparameters
    neg_per_pos = 3  # Reduced from 5
    curriculum_epochs = 3  # Reduced from 5
    num_iterations = 50  # Reduced from 100
    train_batch_size = 32  # Added batch size control
    
    # Pre-compute positive pairs
    pos_pairs = create_patient_drug_pairs(patient_id_tensor, val_drug_indices, device)
    
    
    for epoch in range(cfg.train.num_epoch):
        epoch_start = time.time()
        model.train()
        losses = []
        
        hard_ratio = min(epoch / curriculum_epochs, 1.0)
        n_hard = int(neg_per_pos * hard_ratio)
        n_random = neg_per_pos - n_hard
        
        logger.warning(f"\nEpoch {epoch+1}")
        logger.warning(f"Hard negatives: {n_hard}, Random: {n_random}")
        
        # Get hard negatives if needed
        if n_hard > 0:
            hard_negative_start = time.time()
            try:
                hard_negatives = get_hard_negatives_batched(
                    model, data_gpu, patient_id_tensor, 
                    all_drug_indices, device, 
                    n_hard * len(val_drug_indices), 
                    relation_dict_f,
                    batch_size=32
                )
                logger.warning(f"Hard negative mining took: {time.time() - hard_negative_start:.2f} seconds")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    logger.warning("OOM during hard negative mining, falling back to random negatives")
                    n_hard = 0
                    n_random = neg_per_pos
                else:
                    raise e
        
        # Training loop
        pbar = tqdm(range(num_iterations), desc=f"Training")
        for iter_idx in pbar:
            iter_start = time.time()  # Start timing for this iteration
            
            # Process positive examples
            pos_scores = model(data_gpu, pos_pairs).squeeze(-1)
            
            neg_indices_list = []
            
            # Add hard negatives if available
            if n_hard > 0:
                hard_indices = hard_negatives[
                    torch.randperm(len(hard_negatives))[:n_hard * len(val_drug_indices)]
                ]
                neg_indices_list.append(hard_indices)
            
            # Add random negatives
            if n_random > 0:
                random_indices = all_drug_indices[
                    torch.randperm(len(all_drug_indices))[:n_random * len(val_drug_indices)]
                ]
                neg_indices_list.append(random_indices)
            
            # Process negatives in batches
            neg_scores_list = []
            neg_indices = torch.cat(neg_indices_list)
            
            for start_idx in range(0, len(neg_indices), train_batch_size):
                end_idx = min(start_idx + train_batch_size, len(neg_indices))
                batch_neg_indices = neg_indices[start_idx:end_idx]
                
                neg_pairs = create_patient_drug_pairs(patient_id_tensor, batch_neg_indices, device)
                batch_neg_scores = model(data_gpu, neg_pairs).squeeze(-1)
                neg_scores_list.append(batch_neg_scores)
                
                # Clear memory
                del neg_pairs
                torch.cuda.empty_cache()
            
            neg_scores = torch.cat(neg_scores_list)
            neg_scores = neg_scores.view(len(val_drug_indices), -1)
            
            # Compute loss
            if use_personalized:
                loss = criterion(pos_scores, neg_scores, prs_values, expr_values)
            else:
                loss = criterion(pos_scores, neg_scores)
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            losses.append(loss.item())
            
            # Calculate iteration time and update progress bar
            iter_time = time.time() - iter_start
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{sum(losses)/len(losses):.4f}",
                'iter_time': f"{iter_time:.2f}s"
            })

        # Validation step
        if epoch % getattr(cfg, 'validate_every', 1) == 0:
            val_metrics = test_patient_cs(
                cfg, model, valid_data_cpu,
                ground_truth_drugs=val_drugs,
                device=device, logger=logger,
                relation_dict_f=relation_dict_f,
                entity_dict_f=entity_dict_f
            )
            val_mrr = val_metrics.get("MRR", 0.0)
            scheduler.step(val_mrr)

            logger.warning(f"\nValidation metrics:")
            logger.warning(f"MRR: {val_mrr:.4f}")
            logger.warning(f"Hits@1: {val_metrics['hits']['hits@1']:.4f}")
            logger.warning(f"Hits@3: {val_metrics['hits']['hits@3']:.4f}")
            logger.warning(f"Hits@10: {val_metrics['hits']['hits@10']:.4f}")

            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                logger.warning("New best model saved!")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.warning("Early stopping triggered.")
                break

        # Log epoch summary
        avg_loss = sum(losses) / len(losses) if losses else 0
        epoch_time = time.time() - epoch_start
        logger.warning(f"Epoch {epoch+1} - Average loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

    model.load_state_dict(best_model_state)
    return model

def train_with_hard_negatives(
    cfg, 
    model, 
    train_data_cpu, 
    valid_data_cpu, 
    filtered_data_cpu,
    device, 
    batch_per_epoch, 
    logger, 
    val_drugs, 
    relation_dict_f,
    entity_dict_f,    # Added this parameter
    use_personalized=False, 
    patient_data_fn=None
):
    if use_personalized:
        try:
            prs_values, expr_values, disease_names, protein_names = load_patient_data(patient_data_fn, device=device)
            prs_values = torch.tensor(prs_values, device=device)
            expr_values = torch.tensor(expr_values, device=device)
            
            prs_values = (prs_values - prs_values.mean()) / prs_values.std()
            expr_values = (expr_values - expr_values.mean()) / expr_values.std()

            criterion = PersonalizedDrugLoss(
                disease_names=disease_names,
                protein_names=protein_names,
                lambda1=0.01,
                lambda2=0.01
            ).to(device)
        except Exception as e:
            raise Exception(f"Error setting up personalized learning: {str(e)}")
    else:
        criterion = BPRLoss(margin=1.5).to(device)

    best_val_mrr = float('-inf')
    best_model_state = copy.deepcopy(model.state_dict())
    patience = 10
    epochs_no_improve = 0
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    #scheduler = optim.ReduceLROnPlateau(
    #    optimizer, mode='max', factor=0.5, patience=3, verbose=True, threshold=1e-3
    #)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True, threshold=1e-3
    )

    val_drug_indices = torch.tensor([entity_dict_f[drug] for drug in val_drugs if drug in entity_dict_f], device=device)
    
    all_drug_indices = get_drug_node_indices(train_data_cpu, relation_dict_f).to(device)
    patient_id = get_patient_node_id(train_data_cpu, relation_dict_f).item()
    patient_id_tensor = torch.tensor([patient_id], dtype=torch.long, device=device)

    data_gpu = move_data_to_gpu(train_data_cpu, device)

    neg_per_pos = 5
    curriculum_epochs = 5
    
    logger.warning(f"\nTraining with hard negative mining:")
    logger.warning(f"Negatives per positive: {neg_per_pos}")
    logger.warning(f"Curriculum epochs: {curriculum_epochs}")
    
    for epoch in range(cfg.train.num_epoch):
        model.train()
        losses = []
        num_iterations = 100
        
        hard_ratio = min(epoch / curriculum_epochs, 1.0)
        n_hard = int(neg_per_pos * hard_ratio)
        n_random = neg_per_pos - n_hard
        
        logger.warning(f"\nEpoch {epoch+1}")
        logger.warning(f"Hard negatives: {n_hard}, Random: {n_random}")
        
        for iter_idx in range(num_iterations):
            pos_pairs = create_patient_drug_pairs(patient_id_tensor, val_drug_indices, device)
            pos_scores = model(data_gpu, pos_pairs).squeeze(-1)
            
            neg_pairs_list = []
            
            if n_hard > 0:
                hard_negative_indices = get_hard_negatives(
                    model, data_gpu, patient_id_tensor, 
                    all_drug_indices, device, n_hard, relation_dict_f
                )
            
            for _ in range(len(val_drug_indices)):
                neg_indices = []
                
                if n_hard > 0:
                    neg_indices.append(hard_negative_indices)
                
                if n_random > 0:
                    random_indices = all_drug_indices[torch.randperm(len(all_drug_indices))[:n_random]].to(device)
                    neg_indices.append(random_indices)
                
                neg_indices = torch.cat(neg_indices)
                neg_pairs = create_patient_drug_pairs(patient_id_tensor, neg_indices, device)
                neg_pairs_list.append(neg_pairs)
            
            neg_pairs = torch.cat(neg_pairs_list, dim=0)
            neg_scores = model(data_gpu, neg_pairs).squeeze(-1)
            neg_scores = neg_scores.view(len(val_drug_indices), neg_per_pos)

            if use_personalized:
                loss = criterion(pos_scores, neg_scores, prs_values, expr_values)
            else:
                loss = criterion(pos_scores, neg_scores)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            losses.append(loss.item())

            if iter_idx % 20 == 0:
                logger.warning(f"Iteration {iter_idx}/{num_iterations} - Loss: {loss.item():.4f}")

        if epoch % getattr(cfg, 'validate_every', 1) == 0:
            val_metrics = test_patient_cs(
                cfg, model, valid_data_cpu,
                ground_truth_drugs=val_drugs,
                device=device, logger=logger,
                relation_dict_f=relation_dict_f,
                entity_dict_f=entity_dict_f
            )
            val_mrr = val_metrics.get("MRR", 0.0)
            scheduler.step(val_mrr)

            logger.warning(f"\nValidation metrics:")
            logger.warning(f"MRR: {val_mrr:.4f}")
            logger.warning(f"Hits@1: {val_metrics['hits']['hits@1']:.4f}")
            logger.warning(f"Hits@3: {val_metrics['hits']['hits@3']:.4f}")
            logger.warning(f"Hits@10: {val_metrics['hits']['hits@10']:.4f}")

            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                logger.warning("New best model saved!")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.warning("Early stopping triggered.")
                break

        avg_loss = sum(losses) / len(losses) if losses else 0
        logger.warning(f"Epoch {epoch+1} - Average loss: {avg_loss:.4f}")

    model.load_state_dict(best_model_state)
    return model

def run_cross_validation_with_test(
    cfg, base_model, train_data_cpu, valid_data_cpu, filtered_data_cpu,
    gt_drugs, device, batch_per_epoch, logger, relation_dict_f, entity_dict_f,
    test_size=0.2, use_personalized=False, patient_data_fn=None, n_folds=3
):
    """
    Perform k-fold cross-validation with a held-out test set for small datasets.
    """
    # First split into train+val and test sets
    train_val_drugs, test_drugs = train_test_split(
        gt_drugs, 
        test_size=test_size, 
        random_state=42
    )
    
    logger.warning(f"\nSplit sizes:")
    logger.warning(f"Train+Val set: {len(train_val_drugs)} drugs")
    logger.warning(f"Test set: {len(test_drugs)} drugs")
    
    # Initialize metrics dictionaries
    cv_metrics = {
        'MRR': [], 'MR': [], 'roc_auc': [], 'auprc': [],
        'hits@1': [], 'hits@3': [], 'hits@10': []
    }
    
    # Setup k-fold cross validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    train_val_array = np.array(train_val_drugs)
    
    best_val_mrr = float('-inf')
    best_model_state = None
    
    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_array)):
        logger.warning(f"\nFold {fold + 1}/{n_folds}")
        logger.warning("-" * 50)
        
        # Get train and validation drugs for this fold
        train_drugs_fold = train_val_array[train_idx].tolist()
        val_drugs_fold = train_val_array[val_idx].tolist()
        
        logger.warning(f"Train set size: {len(train_drugs_fold)}")
        logger.warning(f"Validation set size: {len(val_drugs_fold)}")
        
        # Create a fresh copy of the model for this fold
        model = copy.deepcopy(base_model)
        model = model.to(device)
        
        # Train the model on this fold
        train_with_hard_negatives_efficient(
            cfg=cfg,
            model=model,
            train_data_cpu=train_data_cpu,
            valid_data_cpu=valid_data_cpu,
            filtered_data_cpu=filtered_data_cpu,
            device=device,
            batch_per_epoch=batch_per_epoch,
            logger=logger,
            val_drugs=val_drugs_fold,
            relation_dict_f=relation_dict_f,
            entity_dict_f=entity_dict_f,    # This parameter is now properly defined
            use_personalized=use_personalized,
            patient_data_fn=patient_data_fn
        )
        
        # Evaluate on validation set
        val_metrics = test_patient_cs(
            cfg=cfg,
            model=model,
            data_cpu=valid_data_cpu,
            ground_truth_drugs=val_drugs_fold,
            device=device,
            logger=logger,
            relation_dict_f=relation_dict_f,
            entity_dict_f=entity_dict_f
        )
        
        # Store metrics for this fold
        for metric in cv_metrics:
            if metric.startswith('hits@'):
                k = metric.split('@')[1]
                cv_metrics[metric].append(val_metrics['hits'][f'hits@{k}'])
            else:
                cv_metrics[metric].append(val_metrics[metric])
        
        # Track best model based on validation MRR
        if val_metrics['MRR'] > best_val_mrr:
            best_val_mrr = val_metrics['MRR']
            best_model_state = copy.deepcopy(model.state_dict())
        
        logger.warning(f"\nFold {fold + 1} Validation Results:")
        logger.warning(f"MRR: {val_metrics['MRR']:.4f}")
        logger.warning(f"MR: {val_metrics['MR']:.4f}")
        logger.warning(f"ROC AUC: {val_metrics['roc_auc']:.4f}")
        logger.warning(f"AUPRC: {val_metrics['auprc']:.4f}")
        for k in [1, 3, 10]:
            logger.warning(f"Hits@{k}: {val_metrics['hits'][f'hits@{k}']:.4f}")
    
    # Calculate and log average cross-validation metrics
    final_cv_metrics = {}
    logger.warning("\nCross-validation Results (mean ± std):")
    logger.warning("-" * 50)
    for metric in cv_metrics:
        values = cv_metrics[metric]
        mean_val = np.mean(values)
        std_val = np.std(values)
        final_cv_metrics[metric] = {'mean': mean_val, 'std': std_val}
        logger.warning(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Evaluate on test set using best model
    logger.warning("\nEvaluating on held-out test set...")
    logger.warning("-" * 50)
    
    final_model = copy.deepcopy(base_model)
    final_model.load_state_dict(best_model_state)
    final_model = final_model.to(device)
    
    test_metrics = test_patient_cs(
        cfg=cfg,
        model=final_model,
        data_cpu=test_data_cpu,
        ground_truth_drugs=test_drugs,
        device=device,
        logger=logger,
        relation_dict_f=relation_dict_f,
        entity_dict_f=entity_dict_f
    )
    
    return final_cv_metrics, test_metrics

'''
class BPRLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(BPRLoss, self).__init__()
        self.margin = margin

    def forward(self, pos_scores, neg_scores):
        # Ensure proper dimensions
        if len(pos_scores.shape) == len(neg_scores.shape):
            pos_scores = pos_scores.unsqueeze(1)
        pos_scores = pos_scores.expand_as(neg_scores)
        
        difference = pos_scores - neg_scores
        loss = torch.clamp(self.margin - difference, min=0)
        return loss.mean()
'''
class BPRLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(BPRLoss, self).__init__()
        self.margin = margin

    def forward(self, pos_scores, neg_scores):
        # Ensure pos_scores and neg_scores have consistent dimensions
        pos_scores = pos_scores.view(-1, 1)  # Reshape to [batch_size, 1]
        neg_scores = neg_scores.view(pos_scores.size(0), -1)  # Reshape to [batch_size, num_neg]
        
        # Expand positive scores to match negative scores shape
        pos_scores = pos_scores.expand_as(neg_scores)
        
        # Calculate loss
        difference = pos_scores - neg_scores
        loss = torch.clamp(self.margin - difference, min=0)
        return loss.mean()

### LOSSES
class DrugRecommendationLoss(nn.Module):
    """Base class for drug recommendation loss functions."""
    def __init__(self):
        super(DrugRecommendationLoss, self).__init__()
    
    def forward(self, pos_scores, neg_scores):
        """Each loss implementation must define its forward pass."""
        raise NotImplementedError

class BPRDrugLoss(DrugRecommendationLoss):
    """Basic BPR loss for drug recommendation."""
    def __init__(self, margin=1.5):
        super(BPRDrugLoss, self).__init__()
        self.margin = margin
    
    def forward(self, pos_scores, neg_scores):
        # Our existing BPR loss implementation
        if len(pos_scores.shape) == len(neg_scores.shape):
            pos_scores = pos_scores.unsqueeze(1)
        pos_scores = pos_scores.expand_as(neg_scores)
        
        difference = pos_scores - neg_scores
        loss = torch.clamp(self.margin - difference, min=0)
        return loss.mean()


class PersonalizedDrugLoss(DrugRecommendationLoss):
    def __init__(self, disease_names, protein_names, lambda1=0.01, lambda2=0.01, dropout=0.2):
        super(PersonalizedDrugLoss, self).__init__()
        self.disease_names = disease_names
        self.protein_names = protein_names

        # Initialize with smaller values and add eps to prevent division by zero
        self.eps = 1e-10
        self.dropout = nn.Dropout(dropout)
        
        # Only create weights for valid features
        if len(disease_names) > 0:
            self.disease_weights = nn.Parameter(torch.randn(len(disease_names)) * 0.001)
        else:
            self.disease_weights = None
            
        if len(protein_names) > 0:
            self.protein_weights = nn.Parameter(torch.randn(len(protein_names)) * 0.001)
        else:
            self.protein_weights = None
            
        self.lambda1 = lambda1 if len(disease_names) > 0 else 0
        self.lambda2 = lambda2 if len(protein_names) > 0 else 0
        self.weight_decay = 0.001
        self.last_components = {}

    def forward(self, pos_scores, neg_scores, prs_values, expr_values):
        # Ensure proper shapes
        batch_size, num_neg = neg_scores.shape
        pos_scores = pos_scores.view(-1, 1).expand(batch_size, num_neg)

        # Clip values for numerical stability
        pos_scores = torch.clamp(pos_scores, min=-100, max=100)
        neg_scores = torch.clamp(neg_scores, min=-100, max=100)
        
        # Base BPR loss with numerical stability
        diff = pos_scores - neg_scores
        bpr = -torch.mean(torch.log(torch.sigmoid(diff) + self.eps))
        
        # Initialize personalization terms
        prs_term = torch.tensor(0.0, device=pos_scores.device)
        expr_term = torch.tensor(0.0, device=pos_scores.device)
        l2_reg = torch.tensor(0.0, device=pos_scores.device)
        
        # Only compute PRS impact if we have disease features
        if self.disease_weights is not None and len(prs_values) > 0:
            disease_weights_norm = F.normalize(self.disease_weights, p=2, dim=0)
            prs_impact = torch.sum(disease_weights_norm * prs_values)
            prs_impact = torch.tanh(prs_impact)
            score_diffs = torch.clamp((pos_scores - neg_scores).abs(), min=0, max=10)
            attention = torch.sigmoid(score_diffs)
            prs_term = torch.mean(attention * prs_impact)
            l2_reg = l2_reg + self.weight_decay * torch.norm(disease_weights_norm)
            
        # Only compute expression impact if we have protein features
        if self.protein_weights is not None and len(expr_values) > 0:
            protein_weights_norm = F.normalize(self.protein_weights, p=2, dim=0)
            expr_impact = torch.sum(protein_weights_norm * expr_values)
            expr_impact = torch.tanh(expr_impact)
            score_diffs = torch.clamp((pos_scores - neg_scores).abs(), min=0, max=10)
            attention = torch.sigmoid(score_diffs)
            expr_term = torch.mean(attention * expr_impact)
            l2_reg = l2_reg + self.weight_decay * torch.norm(protein_weights_norm)

        # Store components for logging
        self.last_components = {
            'bpr': bpr.item(),
            'prs': prs_term.item(),
            'expr': expr_term.item(),
            'reg': l2_reg.item()
        }
        
        # Combine terms (personalization terms will be 0 if features not present)
        total_loss = (
            bpr + 
            self.lambda1 * torch.clamp(prs_term, min=-1, max=1) + 
            self.lambda2 * torch.clamp(expr_term, min=-1, max=1) + 
            l2_reg
        )
        
        return total_loss

    def get_importance_scores(self):
        """Returns dictionaries of importance scores for diseases and proteins."""
        disease_importance = {}
        protein_importance = {}
        
        # Only compute disease importance if we have disease features
        if self.disease_weights is not None:
            disease_weights = F.normalize(self.disease_weights, p=2, dim=0)
            disease_importance = {
                name: weight.item() 
                for name, weight in zip(self.disease_names, disease_weights)
            }
            disease_importance = dict(sorted(
                disease_importance.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            ))
            
        # Only compute protein importance if we have protein features
        if self.protein_weights is not None:
            protein_weights = F.normalize(self.protein_weights, p=2, dim=0)
            protein_importance = {
                name: weight.item() 
                for name, weight in zip(self.protein_names, protein_weights)
            }
            protein_importance = dict(sorted(
                protein_importance.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            ))
            
        return disease_importance, protein_importance
      
# ----------------------------------------------------------------------
# Utility functions for patient/drug relations
# ----------------------------------------------------------------------

class AdaptiveLambdaScheduler:
    def __init__(self, initial_lambda=0.1, max_lambda=0.5, warmup_epochs=15):  # Increased values and slower warmup
        self.initial_lambda = initial_lambda
        self.max_lambda = max_lambda
        self.warmup_epochs = warmup_epochs
    
    def get_lambda(self, epoch):
        # More gradual increase with sigmoid function
        progress = min(epoch / self.warmup_epochs, 1.0)
        sigmoid_progress = 1 / (1 + np.exp(-10 * (progress - 0.5)))
        return self.initial_lambda + (self.max_lambda - self.initial_lambda) * sigmoid_progress


# Function to load the data back:
def load_patient_data(input_path, device=None):
    """
    Load patient data from pickle, handling missing/nan values by filtering them out.
    """
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    # Convert to numpy arrays first
    prs_values = np.array(data['prs_values'], dtype=np.float32)
    expr_values = np.array(data['expr_values'], dtype=np.float32)
    disease_names = data['disease_names']
    protein_names = data['protein_names']

    # Filter out nan/invalid values for PRS
    valid_prs_mask = ~np.isnan(prs_values) & ~np.isinf(prs_values)
    prs_values = prs_values[valid_prs_mask]
    disease_names = [name for i, name in enumerate(disease_names) if valid_prs_mask[i]]

    # Filter out nan/invalid values for expression
    valid_expr_mask = ~np.isnan(expr_values) & ~np.isinf(expr_values)
    expr_values = expr_values[valid_expr_mask]
    protein_names = [name for i, name in enumerate(protein_names) if valid_expr_mask[i]]

    print(f"\nAfter filtering:")
    print(f"Valid PRS values: {len(prs_values)} (was {len(valid_prs_mask)})")
    print(f"Valid expression values: {len(expr_values)} (was {len(valid_expr_mask)})")
    
    # If device is provided, convert to torch tensors
    if device is not None:
        prs_values = torch.tensor(prs_values, device=device)
        expr_values = torch.tensor(expr_values, device=device)
    
    return (prs_values, expr_values, disease_names, protein_names)

def load_relation_dicts(processed_dir):
    dicts_path = os.path.join(processed_dir, "entity_relation_dicts.pt")
    print(f"Loading entity and relation dictionaries from: {dicts_path}")
    
    loaded_dicts = torch.load(dicts_path)
    # loaded_dicts is a dict with 4 keys:
    #   "entity_dict_forward"
    #   "relation_dict_forward"
    #   "entity_dict_inverted"
    #   "relation_dict_inverted"

    entity_dict_forward = loaded_dicts["entity_dict_forward"]
    relation_dict_forward = loaded_dicts["relation_dict_forward"]
    entity_dict_inverted = loaded_dicts["entity_dict_inverted"]
    relation_dict_inverted = loaded_dicts["relation_dict_inverted"]

    # Return them in whatever order you prefer
    return (entity_dict_forward, relation_dict_forward,
            entity_dict_inverted, relation_dict_inverted)

import torch
import re

def get_drug_node_indices(data, relation_dict_f, entity_dict_i=None):
    """
    Return all indices of nodes that are 'drugs' in the graph,
    AND whose names match the pattern DB followed by 5 digits (e.g. DB00437).

    :param data: PyG Data object with edge_index, edge_type, etc.
    :param relation_dict_f: dict mapping relation-name -> relation-ID (e.g. {"DRUG_GENE": 0, ...})
    :param entity_dict_f: dict mapping entity-id -> entity-name (e.g. {13: "DB00437", 42: "DB00945", ...})
                          Optional. If None, no name-based filtering is done.
    :return: A tensor of node indices that (1) appear as "drug" nodes, and
             (2) optionally have a name matching the DB{5 digits} pattern.
    """
    # 1) Identify edges that correspond to "drug" relations
    drug_relations = [
        relation_dict_f["DRUG_GENE"],
        relation_dict_f["GENE_DRUG"],
        relation_dict_f["DRUG_SIDER"],
        relation_dict_f["SIDER_DRUG"],
        relation_dict_f["TREAT"],
        relation_dict_f["THERAPY"]
    ]
    # drug_edges is a boolean mask for edges of the above types
    drug_edges = torch.any(torch.stack([data.edge_type == rel for rel in drug_relations]), dim=0)
    # drug_nodes: All nodes on the 'tail' side of these edges
    drug_nodes = torch.unique(data.edge_index[1][drug_edges])

    # 2) If we have an entity_dict_i for node ID -> name, filter by pattern
    #    '^DB\d{5}$' means "DB" followed by exactly 5 digits.
    if entity_dict_i is not None:
        pattern = re.compile(r'^DB\d{5}$')
        filtered = []
        for node_id in drug_nodes:
            # node_id might be a scalar tensor, so convert to int
            node_int = node_id.item()  
            # Lookup the string name (or fallback to "")
            node_name = entity_dict_i.get(node_int, "")
            if pattern.match(node_name):
                filtered.append(node_id)
        # Convert Python list of Tensors back to a single tensor
        if len(filtered) > 0:
            drug_nodes = torch.stack(filtered, dim=0)  # shape [N_filtered]
        else:
            # If none matched, return empty tensor
            drug_nodes = torch.tensor([], dtype=drug_nodes.dtype, device=drug_nodes.device)

    return drug_nodes


def get_patient_node_id(data, relation_dict_f):
    """
    Return the single index of the 'patient' node (assuming only 1?).
    """
    patient_relations = [
        relation_dict_f["UP_REGULATED"],
        relation_dict_f["DOWN_REGULATED"],
        relation_dict_f["WAS_DIAGNOSED"]
    ]
    patient_edges = torch.any(torch.stack([data.edge_type == rel for rel in patient_relations]), dim=0)
    return data.edge_index[0][patient_edges][0]

def create_patient_drug_pairs(patient_id, drug_indices, device):
    """
    Create positive (patient -> drug) pairs in a single tensor.
    """
    num_pairs = len(drug_indices)
    # Convert patient_id to scalar if it's a tensor
    if isinstance(patient_id, torch.Tensor):
        patient_id = patient_id.item()
    
    heads = torch.full((num_pairs, 1), patient_id, device=device)
    tails = drug_indices.view(-1, 1).to(device)
    relation = torch.full((num_pairs, 1), 15, device=device)  # 15 is 'TREAT' or 'THERAPY'
    return torch.cat([heads, tails, relation], dim=1).unsqueeze(1)

def drug_negative_sampling(data, batch, num_negative, relation_dict_f):
    """
    Basic negative sampling: picks random drugs for each positive triple.
    """
    drug_indices = get_drug_node_indices(data, relation_dict_f, entity_dict_i)  # All drug nodes
    neg_samples = []
    for _ in range(num_negative):
        # Randomly pick from the set of drug_indices
        neg = torch.randint(0, len(drug_indices), batch.shape[:-1], device=batch.device)
        neg_samples.append(drug_indices[neg])
    return torch.stack(neg_samples, dim=-1)

# ----------------------------------------------------------------------
# Example: mini-batch training to reduce GPU usage
# ----------------------------------------------------------------------

import torch.nn as nn
def train_and_validate_patient_cs(
    cfg, model, train_data_cpu, valid_data_cpu, filtered_data_cpu,
    device, batch_per_epoch, logger, val_drugs, relation_dict_f,
    use_personalized=False, patient_data_fn=None
):
    if use_personalized:
        try:
            prs_values, expr_values, disease_names, protein_names = load_patient_data(patient_data_fn, device=device)
            
            prs_values = torch.tensor(prs_values, device=device)
            expr_values = torch.tensor(expr_values, device=device)
            
            prs_values = (prs_values - prs_values.mean()) / prs_values.std()
            expr_values = (expr_values - expr_values.mean()) / expr_values.std()

            criterion = PersonalizedDrugLoss(
                disease_names=disease_names,
                protein_names=protein_names,
                lambda1=0.01,
                lambda2=0.01
            ).to(device)
        except Exception as e:
            raise Exception(f"Error setting up personalized learning: {str(e)}")
    else:
        criterion = BPRLoss(margin=1.5).to(device)

    best_val_mrr = float('-inf')
    best_model_state = copy.deepcopy(model.state_dict())
    patience = 10
    epochs_no_improve = 0
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True, threshold=1e-3
    )

    # Convert validation drugs to their numeric IDs
    val_drug_indices = torch.tensor([entity_dict_f[drug] for drug in val_drugs if drug in entity_dict_f], device=device)
    
    # Get all drug indices and patient ID
    all_drug_indices = get_drug_node_indices(train_data_cpu, relation_dict_f)
    patient_id = get_patient_node_id(train_data_cpu, relation_dict_f).item()
    patient_id_tensor = torch.tensor([patient_id], dtype=torch.long, device=device)#[0]

    data_gpu = move_data_to_gpu(train_data_cpu, device)
    
    # Pre-compute all drug indices and reuse
    all_drug_indices = get_drug_node_indices(train_data_cpu, relation_dict_f)
    
    # Convert validation drugs to numeric IDs once
    val_drug_indices = torch.tensor([
        entity_dict_f[drug] for drug in val_drugs 
        if drug in entity_dict_f
    ], device=device)

    # Get patient ID once
    patient_id = get_patient_node_id(train_data_cpu, relation_dict_f).item()
    patient_id_tensor = torch.tensor([patient_id], dtype=torch.long, device=device)[0]

    # Pre-compute positive pairs once
    pos_pairs = create_patient_drug_pairs(patient_id_tensor, val_drug_indices, device)

    for epoch in range(cfg.train.num_epoch):
        model.train()
        losses = []
        
        num_iterations = 50
        neg_per_pos = 10
        batch_size = len(val_drug_indices)  # number of positive examples

        # Use tqdm for progress tracking
        pbar = tqdm(range(num_iterations), desc=f"Epoch {epoch+1}")
        
        for _ in pbar:
            # Forward pass for positive pairs - do this once per iteration
            pos_scores = model(data_gpu, pos_pairs).squeeze(-1)
            
            # Efficiently sample negative examples
            neg_indices = all_drug_indices[
                torch.randperm(len(all_drug_indices), device=device)[:batch_size * neg_per_pos]
            ]
            
            # Create negative pairs efficiently
            neg_pairs = create_patient_drug_pairs(
                patient_id_tensor, 
                neg_indices, 
                device
            )
            
            # Single forward pass for all negatives
            neg_scores = model(data_gpu, neg_pairs).squeeze(-1)
            neg_scores = neg_scores.view(batch_size, neg_per_pos)

            # Compute loss
            if use_personalized:
                loss = criterion(pos_scores, neg_scores, prs_values, expr_values)
            else:
                loss = criterion(pos_scores, neg_scores)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            losses.append(loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{sum(losses)/len(losses):.4f}"
            })

        validate_every = getattr(cfg, 'validate_every', 3)
        
        # Validation step
        if epoch % validate_every == 0:
            val_metrics = test_patient_cs(
                cfg, model, valid_data_cpu,
                ground_truth_drugs=val_drugs,
                device=device, logger=logger,
                relation_dict_f=relation_dict_f,
                entity_dict_f=entity_dict_f
            )
            val_mrr = val_metrics.get("MRR", 0.0)
            scheduler.step(val_mrr)

            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.warning("Early stopping triggered.")
                break

        avg_loss = sum(losses) / len(losses) if losses else 0
        logger.warning(f"Epoch {epoch+1} - avg loss: {avg_loss:.4f}")

    model.load_state_dict(best_model_state)
    return model

# ----------------------------------------------------------------------

@torch.no_grad()
def test_patient_cs(
    cfg,
    model,
    data_cpu,
    ground_truth_drugs, 
    device,
    logger,
    relation_dict_f,   # e.g. {"TREAT": 15, "THERAPY": 16, ...}
    entity_dict_f,     # e.g. {"DB00437": 13, "DB00945": 42, ...}
    threshold=0.5,     # For binarizing scores for P/R/F1
    hits_ks=(1, 3, 10) # Which Hits@K metrics to compute
):
    """
    Evaluate multiple metrics for the patient->drug link-prediction task.

    ground_truth_drugs: list of strings (e.g. ["DB00437", "DB00945"]) that 
        are the *correct* drugs for the patient. We'll convert them to numeric IDs
        via entity_dict_f.
    """

    model.eval()
    batch_size = 8

    # 1) Move the graph data to GPU for inference
    data_gpu = move_data_to_gpu(data_cpu, device)

    # 2) Find all candidate "drug" nodes & the single "patient" node
    drug_indices = get_drug_node_indices(data_cpu, relation_dict_f)  # shape [D]
    patient_id = get_patient_node_id(data_cpu, relation_dict_f).item()

    # Convert the patient ID to a small GPU tensor
    patient_id_tensor = torch.tensor([patient_id], dtype=torch.long, device=device)#[0]

    # 3) Convert ground-truth *string* IDs to numeric IDs
    numeric_drugs = []
    for drug_str in ground_truth_drugs:
        if drug_str in entity_dict_f:
            numeric_id = entity_dict_f[drug_str]
            numeric_drugs.append(numeric_id)
        else:
            logger.warning(f"Drug {drug_str} not found in entity_dict_f; skipping.")

    # If no valid drug left, can't compute metrics
    if not numeric_drugs:
        logger.warning("No valid ground-truth drugs after mapping strings -> IDs!")
        return {}

    numeric_drugs = torch.tensor(numeric_drugs, device=device)  # shape [G]

    # 4) Forward pass in small batches to get scores for ALL candidate drugs
    all_scores = []
    treat_rel_id = relation_dict_f["TREAT"]  # e.g. 15

    for start in range(0, len(drug_indices), batch_size):
        batch_indices_cpu = drug_indices[start : start + batch_size]
        batch_indices = batch_indices_cpu.to(device)

        # shape [B, 3] after concatenation
        batch_pairs = torch.cat([
            patient_id_tensor.repeat(len(batch_indices), 1),
            batch_indices.view(-1, 1),
            torch.full((len(batch_indices), 1), treat_rel_id, device=device)
        ], dim=1).unsqueeze(1)  # shape [B, 1, 3]

        # Forward pass
        batch_scores = model(data_gpu, batch_pairs)  # shape [B, 1]
        all_scores.append(batch_scores)

    # Combine
    scores = torch.cat(all_scores, dim=0)  # [D, 1]
    scores = scores.view(-1)               # [D]

    # 5) RANKING METRICS (MR, MRR, Hits@k, MAP)
    # ----------------------------------------------------------------
    # We'll find the rank for each ground-truth drug among [drug_indices].
    # Then we'll compute MR (mean rank) & MRR (mean reciprocal rank).
    # We'll also do Hits@k for each k in hits_ks.
    # We'll do MAP by looking at positions of all positives in the sorted list.
    # ----------------------------------------------------------------

    # A) Create a "drug_id -> index" mapping for quick lookup
    #    Instead of calling (drug_indices == drug_id).nonzero() each time.
    #    We'll store index_in_drug_indices for each drug_id in candidate set.
    #    If a drug is not in drug_indices, skip.
    drug_indices_np = drug_indices.cpu().numpy()
    idx_map = {}  # drug_id -> index in 'drug_indices'
    for i, did in enumerate(drug_indices_np):
        idx_map[did] = i

    # B) For each ground-truth drug, collect its "rank"
    #    We define rank by "1 + number of scores strictly greater"
    #    or "sum of scores >= this one" depending on your style.
    #    We'll do the same approach you had: rank = (scores >= drug_score).sum().
    ranks = []
    found_positives = []  # keep track of which candidate indices are "true"

    for drug_id in numeric_drugs:
        drug_id_cpu = drug_id.item()
        if drug_id_cpu not in idx_map:
            # Not in candidate set
            continue
        idx_in_candidates = idx_map[drug_id_cpu]
        drug_score = scores[idx_in_candidates]
        rank = (scores >= drug_score).sum().item()  # rank=1 is best
        ranks.append(rank)
        found_positives.append(idx_in_candidates)   # store that this index is positive

    # If we found none in the candidate set, skip
    if len(ranks) == 0:
        logger.warning("No ground-truth drugs found in the candidate drug_indices!")
        return {}

    ranks = torch.tensor(ranks, device=device, dtype=torch.float)
    mr = ranks.mean().item()  # mean rank
    mrr = (1.0 / ranks).mean().item()

    # C) Hits@k: fraction of ground-truth drugs with rank <= k
    hits_results = {}
    for k in hits_ks:
        hits_k = (ranks <= k).float().mean().item()  # ratio of positives with rank <= k
        hits_results[f"hits@{k}"] = hits_k

    # D) MAP: Mean Average Precision
    #    We interpret the ranking as sorting all candidates by descending score,
    #    then compute average precision for the set of positives. 
    #    If you have multiple positives, MAP is the average of each positive's precision 
    #    at its rank in the sorted list.
    #    We'll do a direct computation here:
    scores_sorted, indices_sorted = torch.sort(scores, descending=True)
    # Create a 0/1 label vector: label[i] = 1 if i in found_positives else 0
    label_vec = torch.zeros_like(scores, dtype=torch.float, device=device)
    for pos_idx in found_positives:
        label_vec[pos_idx] = 1.0

    # We can compute average precision by enumerating the sorted candidates:
    # For each position p in sorted list, if that candidate is a positive,
    # precision is (# of positives up to p) / p. Then average these over all positives.
    # We'll accumulate partial sums
    total_precision = 0.0
    count_positives = 0
    for rank_idx, cand_idx in enumerate(indices_sorted):
        # rank_idx is 0-based, so actual rank = rank_idx+1
        if label_vec[cand_idx] > 0:
            count_positives += 1
            prec_at_p = count_positives / (rank_idx + 1)
            total_precision += prec_at_p

    # 6) BINARY CLASSIFICATION METRICS (AUC, AUPRC, Precision/Recall/F1)
    # ----------------------------------------------------------------
    # For AUC, we need *all* candidates labeled 0 or 1:
    #   label=1 if this candidate drug is in numeric_drugs, else 0.
    # We'll use the same label_vec we built above.
    labels_np = label_vec.cpu().numpy()  # shape [D], 0/1
    scores_np = scores.cpu().numpy()     # shape [D], float
    try:
        roc_auc = roc_auc_score(labels_np, scores_np)
    except ValueError:
        roc_auc = float("nan")  # might happen if all labels are 1 or 0

    try:
        auprc = average_precision_score(labels_np, scores_np)
    except ValueError:
        auprc = float("nan")

    # 7) Log and Return
    logger.warning(f"test_patient_cs - MR: {mr:.4f}")
    logger.warning(f"test_patient_cs - MRR: {mrr:.4f}")
    for k, val in hits_results.items():
        logger.warning(f"test_patient_cs - {k}: {val:.4f}")
    logger.warning(f"test_patient_cs - ROC AUC: {roc_auc:.4f}")
    logger.warning(f"test_patient_cs - AUPRC: {auprc:.4f}")

    # Return them in a dict for your reference if needed
    metrics = {
        "MR": mr,
        "MRR": mrr,
        "roc_auc": roc_auc,
        "auprc": auprc,
        "hits": hits_results
    }
    return metrics

# ----------------------------------------------------------------------
import copy

def move_data_to_gpu(data_cpu, device):
    data_gpu = Data(
        edge_index=data_cpu.edge_index.to(device),
        edge_type=data_cpu.edge_type.to(device),
        num_nodes=data_cpu.num_nodes
    )
    
    # If your data has data_cpu.num_relations:
    if hasattr(data_cpu, 'num_relations'):
        # If it's just an integer
        data_gpu.num_relations = data_cpu.num_relations
        # If it's a tensor, do: data_gpu.num_relations = data_cpu.num_relations.to(device)

    # If your data has other attributes you need (like relation_graph),
    # copy them as well:
    if hasattr(data_cpu, 'relation_graph'):
        data_gpu.relation_graph = data_cpu.relation_graph.to(device)

    return data_gpu



# ----------------------------------------------------------------------
# Your other functions (train_and_validate, infer, test, etc.) remain mostly the same,
# but you would want to replicate the same mini-batch logic (and the same "move_data_to_gpu")
# approach in each place you're handling large data objects or large negative-sample expansions.
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Keep all the initialization code
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    
    task_name = cfg.task["name"]
    dataset = util.build_dataset(cfg)

    # Keep data on CPU for chunked usage
    train_data_cpu, valid_data_cpu, test_data_cpu = dataset[0], dataset[1], dataset[2]

    # Get the entity and relation dictionaries
    pid = 1272434
    processed_dir = f"/usr/homes/cxo147/git/ULTRA/kg-datasets/pKG_{pid}/processed/"
    (entity_dict_f, relation_dict_f, entity_dict_i, relation_dict_i) = load_relation_dicts(processed_dir)

    # Build base model - this will be used as a template for each fold
    base_model = Ultra(
        rel_model_cfg=cfg.model.relation_model,
        entity_model_cfg=cfg.model.entity_model,
    )

    # Load checkpoint if specified
    if "checkpoint" in cfg and cfg.checkpoint is not None:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        base_model.load_state_dict(state["model"])

    device = util.get_device(cfg)
    base_model = base_model.to(device)

    # For transductive setting
    filtered_data_cpu = Data(
        edge_index=dataset._data.target_edge_index,
        edge_type=dataset._data.target_edge_type,
        num_nodes=dataset[0].num_nodes
    )

    # Your ground truth drugs
    gt_drugs = ['DB01616', 'DB00321', 'DB00436', 'DB06698', 'DB06724', 'DB00169', 'DB00341', 'DB11089', 'DB01184', 'DB00736', 'DB02703', 'DB00698', 'DB01062', 'DB00316', 'DB00394', 'DB01001']
    #['DB00437', 'DB00945', 'DB00436', 'DB00722', 'DB00641']
    
    # Set whether to use personalized loss
    use_personalized = True
    patient_data_path = f'/usr/homes/cxo147/ismb/KGIA/reasoning/patient_data/G20_pickles/patient_{pid}_data.pkl'

    logger.warning("\nStarting cross-validation with test set evaluation...")
    logger.warning("=" * 50)
    logger.warning(f"Total number of ground truth drugs: {len(gt_drugs)}")
    logger.warning(f"Using personalized loss: {use_personalized}")
    
    # Run cross-validation with test set evaluation
    cv_metrics, test_metrics = run_cross_validation_with_test(
        cfg=cfg,
        base_model=base_model,
        train_data_cpu=train_data_cpu,
        valid_data_cpu=valid_data_cpu,
        filtered_data_cpu=filtered_data_cpu,
        gt_drugs=gt_drugs,
        device=device,
        batch_per_epoch=cfg.train.batch_per_epoch,
        logger=logger,
        relation_dict_f=relation_dict_f,
        entity_dict_f=entity_dict_f,
        test_size=0.2,  # 20% for test set
        use_personalized=use_personalized,
        patient_data_fn=patient_data_path if use_personalized else None
    )

    # Final summary
    logger.warning("\nFinal Results Summary")
    logger.warning("=" * 50)
    
    logger.warning("\nCross-validation Performance:")
    for metric, values in cv_metrics.items():
        logger.warning(f"{metric}:")
        logger.warning(f"  Mean: {values['mean']:.4f}")
        logger.warning(f"  Std:  {values['std']:.4f}")
    
    logger.warning("\nHeld-out Test Set Performance:")
    logger.warning(f"MRR: {test_metrics['MRR']:.4f}")
    logger.warning(f"MR: {test_metrics['MR']:.4f}")
    logger.warning(f"ROC AUC: {test_metrics['roc_auc']:.4f}")
    logger.warning(f"AUPRC: {test_metrics['auprc']:.4f}")
    for k in [1, 3, 10]:
        logger.warning(f"Hits@{k}: {test_metrics['hits'][f'hits@{k}']:.4f}")
