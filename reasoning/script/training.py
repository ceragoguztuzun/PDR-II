#training.py
import torch
from torch import optim
from tqdm import tqdm
import copy
from losses import StandardKGLoss, PersonalizedRepurposingLoss
from utils import move_data_to_gpu
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import traceback

from itertools import islice

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import tasks, util
from ultra.models import Ultra
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from model import initialize_model

import torch
import torch_geometric as pyg
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

from losses import PersonalizedRepurposingLoss
from utils import (
    extract_entity_disease_relationships,
    extract_entity_protein_relationships,
    move_data_to_gpu,
    get_entity_name,
    get_entity_id,
    get_relation_id
)

################
''' TRAINING '''
################


def case_study_rank_drugs(cfg, patient_id, model, criterion, test_data, patient_data, device, logger, target_disease_node = 'C0002395', target_relation = 'THERAPY', case_study_fn = 'ad_pre.tsv'):

    logger.warning("Starting case study eval...")
    test_data_gpu = move_data_to_gpu(test_data, device)
    
    # Extract patient data
    prs_values = torch.tensor(patient_data['prs_values'], device=device)
    expr_values = torch.tensor(patient_data['expr_values'], device=device)

    # Parse your TSV file
    current_directory = os.getcwd()
    print(current_directory)
    drug_ids = []
    with open(f'script/data/{case_study_fn}', 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) == 3 and fields[0] == target_disease_node and fields[1] == target_relation:
                drug_id = get_entity_id(fields[2], patient_id) 
                if drug_id not in drug_ids:
                    drug_ids.append(drug_id)

    # Get Alzheimer's entity ID
    alzheimers_id = get_entity_id(target_disease_node, patient_id)  

    # Rank drugs
    ranked_drugs = rank_drugs_for_disease(
        model, 
        patient_id,
        test_data_gpu, 
        criterion, 
        alzheimers_id, 
        drug_ids, 
        prs_values, 
        expr_values, 
        device
    )

    if len(ranked_drugs) > 0:
        scores = [score for _, score in ranked_drugs]
        print(f"Score stats - Min: {min(scores):.4f}, Max: {max(scores):.4f}, Mean: {sum(scores)/len(scores):.4f}")
        print(f"Unique scores: {len(set([round(s, 4) for s in scores]))} out of {len(scores)}")

    # Print top-ranked drugs
    print("Top 50 personalized drug recommendations for Alzheimer's:")
    for i, (drug_id, score) in enumerate(ranked_drugs[:50]):
        drug_name = get_entity_name(drug_id, patient_id)  
        print(f"{i+1}. {drug_name} (ID: {drug_id}) - Score: {score:.4f}")

    return ranked_drugs

def rank_drugs_for_disease(model, patient_id, data, criterion, disease_id, drug_ids, prs_values, expr_values, device):
    """
    Rank drugs for a specific disease using personalized scores.
    
    Args:
        model: The trained model
        data: Knowledge graph data
        criterion: Personalized loss criterion with trained weights
        disease_id: Entity ID for the disease (e.g., for Alzheimer's)
        drug_ids: List of drug entity IDs to rank
        prs_values: Patient's PRS values
        expr_values: Patient's expression values
        patient_id: Patient ID for the dataset
        device: Computation device
    
    Returns:
        Sorted list of (drug_id, score) tuples
    """
    from ultra import tasks
    import torch
    from tqdm import tqdm
    
    # Move data to device if needed
    data_gpu = move_data_to_gpu(data, device)
    
    # Get relation ID for THERAPY
    relation_id = get_relation_id("THERAPY", patient_id)
    
    # Process drugs in small batches to avoid OOM
    batch_size = 8  # Adjust based on available memory
    all_drug_scores = []

    # Initialize precomputed_adjustments by calling apply_personalization once with a dummy batch.
    dummy_batch = torch.tensor([[disease_id, drug_ids[0], relation_id]], device=device)
    with torch.no_grad():
        t_batch, _ = tasks.all_negative(data_gpu, dummy_batch)
        t_pred = model(data_gpu, t_batch)
        apply_personalization(t_pred, dummy_batch, data_gpu, criterion, prs_values, expr_values, patient_id)
        
    # Process in batches
    for i in tqdm(range(0, len(drug_ids), batch_size), desc="Scoring drugs in batches"):
        # Get current batch of drug IDs
        batch_drug_ids = drug_ids[i:i+batch_size]

        # Near the top of your rank_drugs_for_disease function
        print("Personalization adjustments for these drugs:")
        for drug_id in batch_drug_ids:
            if drug_id < len(apply_personalization.precomputed_adjustments):
                adj = apply_personalization.precomputed_adjustments[drug_id].item()
                print(f"Drug ID {drug_id}: Adjustment = {adj:.8f}")
        
        # Create batch of triplets
        batch = []
        for drug_id in batch_drug_ids:
            batch.append([disease_id, drug_id, relation_id])
        
        # Convert to tensor
        batch_tensor = torch.tensor(batch, device=device)
        
        # Get predictions
        try:
            with torch.no_grad():
                # Generate negative samples
                t_batch, _ = tasks.all_negative(data_gpu, batch_tensor)
                
                # Get predictions
                t_pred = model(data_gpu, t_batch)
                print(f"Base scores: {t_pred[:, 0]}")  # Print base scores before personalization
                
                # Apply personalization
                personalized_pred = apply_personalization(t_pred, batch_tensor, data_gpu, criterion, prs_values, expr_values, patient_id)
                print(f"Personalized scores: {personalized_pred[:, 0]}")  # Print scores after personalization
                
                # Extract scores for positive entities
                scores = personalized_pred[:, 0].cpu().tolist()
                
                # Add to results
                batch_scores = list(zip(batch_drug_ids, scores))
                all_drug_scores.extend(batch_scores)
                
                # Clean up memory
                del t_batch, t_pred, personalized_pred
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            # Handle OOM errors by further reducing batch size
            if "out of memory" in str(e):
                print(f"OOM error with batch size {batch_size}. Try reducing batch_size further.")
                # Could implement recursive retry with smaller batch size
                # For now, just skip this batch
                continue
            else:
                raise e
    
    # Sort by score in descending order
    ranked_drugs = sorted(all_drug_scores, key=lambda x: x[1], reverse=True)
    
    return ranked_drugs

def apply_personalization(batch_scores, batch, data, criterion, prs_values, expr_values, patient_id):
    """
    Apply personalization adjustments to predictions using knowledge of entity identifier patterns.
    """
    # Clone scores to avoid modifying the original
    personalized_scores = batch_scores.clone()
    
    # Extract parameters from criterion
    lambda1 = criterion.lambda1
    lambda2 = criterion.lambda2
    disease_weights = criterion.disease_weights
    protein_weights = criterion.protein_weights
    
    # Initialize and cache relationship mappings
    if not hasattr(apply_personalization, 'precomputed_adjustments'):
        print("Initializing personalization adjustments...")
        try:
            # Load dictionaries with the correct path
            dicts_path = f"/usr/homes/cxo147/git/ULTRA/kg-datasets/biomedical_KG_{patient_id}/processed/entity_relation_dicts.pt"
            dictionaries = torch.load(dicts_path)

            # Inspect what's in the dictionary
            print(f"Available dictionary keys: {list(dictionaries.keys())}")
            
            entity_dict_inverted = dictionaries["entity_dict_inverted"]  # ID -> name
            disease_inv_vocab = dictionaries["disease_inv_vocab"]  # disease_name -> entity_id
            drug_inv_vocab = dictionaries["drug_inv_vocab"]  # drug_name -> entity_id
            gene_inv_vocab = dictionaries["gene_inv_vocab"]  # gene_name -> entity_id
            
            # Map criterion disease names to entity IDs
            criterion_disease_to_entity_id = {}
            for disease_name in criterion.disease_names:
                # Check if this disease name exists in the knowledge graph
                if disease_name in disease_inv_vocab:
                    entity_id = disease_inv_vocab[disease_name]
                    disease_idx = list(criterion.disease_names).index(disease_name)
                    criterion_disease_to_entity_id[disease_idx] = entity_id
            
            # Map criterion protein/gene names to entity IDs
            criterion_protein_to_entity_id = {}
            for protein_name in criterion.protein_names:
                # Check if this protein/gene name exists in the knowledge graph
                if protein_name in gene_inv_vocab:
                    entity_id = gene_inv_vocab[protein_name]
                    protein_idx = list(criterion.protein_names).index(protein_name)
                    criterion_protein_to_entity_id[protein_idx] = entity_id
            
            print(f"Mapped {len(criterion_disease_to_entity_id)} criterion diseases to KG entities")
            print(f"Mapped {len(criterion_protein_to_entity_id)} criterion proteins to KG entities")
            
            # Get drug entity IDs
            drug_entity_ids = set(drug_inv_vocab.values())
            
            # Extract relationships between drugs and diseases/proteins
            drug_disease_rels = {}  # drug_id -> [(disease_idx, strength)]
            drug_protein_rels = {}  # drug_id -> [(protein_idx, strength)]
            
            # Get the graph edges
            edge_index = data.edge_index.t()
            
            # Create disease and protein entity ID sets for faster lookup
            disease_entity_ids = set(disease_inv_vocab.values())
            gene_entity_ids = set(gene_inv_vocab.values())
            
            # For each edge, check if it connects a drug to a disease or protein
            for idx in range(min(1000000, edge_index.size(0))):  # Limit to 1M edges for faster processing
                if idx % 200000 == 0 and idx > 0:
                    print(f"Processed {idx} edges...")
                
                h, t = edge_index[idx, 0].item(), edge_index[idx, 1].item()
                
                # Drug -> Disease relationships
                if h in drug_entity_ids and t in disease_entity_ids:
                    # Find which disease_idx this disease entity corresponds to
                    for disease_idx, disease_entity_id in criterion_disease_to_entity_id.items():
                        if disease_entity_id == t:
                            if h not in drug_disease_rels:
                                drug_disease_rels[h] = []
                            drug_disease_rels[h].append((disease_idx, 1.0))
                
                # Disease -> Drug relationships
                elif t in drug_entity_ids and h in disease_entity_ids:
                    # Find which disease_idx this disease entity corresponds to
                    for disease_idx, disease_entity_id in criterion_disease_to_entity_id.items():
                        if disease_entity_id == h:
                            if t not in drug_disease_rels:
                                drug_disease_rels[t] = []
                            drug_disease_rels[t].append((disease_idx, 1.0))
                
                # Drug -> Protein relationships
                if h in drug_entity_ids and t in gene_entity_ids:
                    # Find which protein_idx this gene entity corresponds to
                    for protein_idx, protein_entity_id in criterion_protein_to_entity_id.items():
                        if protein_entity_id == t:
                            if h not in drug_protein_rels:
                                drug_protein_rels[h] = []
                            drug_protein_rels[h].append((protein_idx, 1.0))
                
                # Protein -> Drug relationships
                elif t in drug_entity_ids and h in gene_entity_ids:
                    # Find which protein_idx this gene entity corresponds to
                    for protein_idx, protein_entity_id in criterion_protein_to_entity_id.items():
                        if protein_entity_id == h:
                            if t not in drug_protein_rels:
                                drug_protein_rels[t] = []
                            drug_protein_rels[t].append((protein_idx, 1.0))
            
            print(f"Found {len(drug_disease_rels)} drugs with disease relationships")
            print(f"Found {len(drug_protein_rels)} drugs with protein relationships")
            
            if len(drug_disease_rels) > 0:
                print(f"Sample drug-disease relationships: {list(drug_disease_rels.items())[:2]}")
            
            if len(drug_protein_rels) > 0:
                print(f"Sample drug-protein relationships: {list(drug_protein_rels.items())[:2]}")
            
            # Pre-compute personalization adjustments
            num_entities = data.num_nodes
            apply_personalization.precomputed_adjustments = torch.zeros(num_entities, device=batch_scores.device)
            
            # Apply PRS adjustments
            prs_adjustment_count = 0
            for drug_id, disease_rels in drug_disease_rels.items():
                if drug_id < num_entities:
                    prs_adjustment = 0.0
                    for disease_idx, strength in disease_rels:
                        prs_adjustment += (
                            lambda1 * 
                            disease_weights[disease_idx] * 
                            prs_values[disease_idx] * 
                            strength
                        )
                    apply_personalization.precomputed_adjustments[drug_id] += prs_adjustment
                    prs_adjustment_count += 1
            
            # Apply expression adjustments
            expr_adjustment_count = 0
            for drug_id, protein_rels in drug_protein_rels.items():
                if drug_id < num_entities:
                    expr_adjustment = 0.0
                    for protein_idx, strength in protein_rels:
                        expr_adjustment += (
                            lambda2 * 
                            protein_weights[protein_idx] * 
                            expr_values[protein_idx] * 
                            strength
                        )
                    apply_personalization.precomputed_adjustments[drug_id] += expr_adjustment
                    expr_adjustment_count += 1
            
            print(f"Applied PRS adjustments to {prs_adjustment_count} entities")
            print(f"Applied expression adjustments to {expr_adjustment_count} entities")
            
            # Check for non-zero adjustments
            non_zero_count = (apply_personalization.precomputed_adjustments != 0).sum().item()
            print(f"Total entities with non-zero adjustments: {non_zero_count}")
            
            if non_zero_count > 0:
                non_zero_indices = torch.nonzero(apply_personalization.precomputed_adjustments).squeeze().cpu().tolist()
                if not isinstance(non_zero_indices, list):
                    non_zero_indices = [non_zero_indices]
                print(f"Sample entity IDs with adjustments: {non_zero_indices[:5]}")
                print(f"Sample adjustment values: {[apply_personalization.precomputed_adjustments[i].item() for i in non_zero_indices[:5]]}")
            
            # If no relationships were found, provide a clear warning and use synthetic adjustments
            if non_zero_count == 0:
                print("\n" + "!"*80)
                print("WARNING: No entity relationships found for personalization!")
                print("This means personalization cannot be applied properly.")
                print("Using synthetic adjustments to demonstrate personalization mechanism.")
                print("!"*80 + "\n")
                
                # Create synthetic adjustments for demonstration
                num_entities = batch_scores.shape[1]
                adjustments = torch.zeros(num_entities, device=batch_scores.device)
                
                # Add synthetic adjustments based on lambda values
                for entity_id in range(0, num_entities, 10):
                    # Apply a small adjustment that depends on lambda values
                    adjustment = (lambda1 + lambda2) * 0.1
                    adjustments[entity_id] = adjustment
                
                apply_personalization.precomputed_adjustments = adjustments
                
                # Debug: check adjustment values
                non_zero_count = (apply_personalization.precomputed_adjustments != 0).sum().item()
                print(f"Created synthetic adjustment tensor with {non_zero_count} non-zero values")
            else:
                print(f"Successfully pre-computed personalization adjustments for all entities")
            
        except Exception as e:
            print("\n" + "!"*80)
            print(f"ERROR during personalization setup: {str(e)}")
            print("Falling back to synthetic adjustments for testing purposes.")
            print("!"*80 + "\n")
            import traceback
            print(traceback.format_exc())
            quit()
            
    # Apply pre-computed adjustments to all entities in the batch
    for i in range(batch_scores.shape[0]):
        valid_range = min(batch_scores.shape[1], len(apply_personalization.precomputed_adjustments))
        
        # Extract entity IDs from the batch
        if batch.shape[1] >= 3:
            # For each item in the batch, the drug/entity ID is typically in position 1 (tail entity)
            entity_id = batch[i, 1].item()  # Get the entity ID from the batch
            
            # Apply amplified adjustment for this entity
            amplification_factor = 1.0 #Reduce the amplification factor.
            if entity_id < len(apply_personalization.precomputed_adjustments):
                adj = apply_personalization.precomputed_adjustments[entity_id] * amplification_factor
                personalized_scores[i, 0] += adj  # Add adjustment to base score
                # Print adjustment for debugging
                #print(f"Entity ID {entity_id}: Adjustment = {adj:.8f}")
        
    # Debug: verify that scores have been modified
    if torch.allclose(batch_scores, personalized_scores):
        print("WARNING: No change in scores after personalization!")
    else:
        # Show the magnitude of changes (but not on every batch to reduce log spam)
        if random.random() < 0.5:  # Show stats for ~5% of batches
            diff = (personalized_scores - batch_scores).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            ##print(f"Score adjustments - Max: {max_diff:.6f}, Mean: {mean_diff:.6f}")
    
    return personalized_scores

def evaluate_model(cfg, model, test_data, device, logger):
    """
    Memory-efficient evaluation that calculates metrics incrementally without
    storing all predictions in memory.
    """
    logger.warning("Starting model evaluation on test data...")

    # Move test data to GPU
    test_data_gpu = move_data_to_gpu(test_data, device)
    
    model.eval()
    with torch.no_grad():
        batch_size = 32
        chunk_size = 1000
        
        # Creat43e test triplets
        test_triplets = torch.cat([
            test_data_gpu.target_edge_index,
            test_data_gpu.target_edge_type.unsqueeze(0)
        ]).t()
        
        # Pre-compute relation-specific entity sets
        relation_entities = defaultdict(set)
        edge_index = test_data_gpu.edge_index.t()
        edge_type = test_data_gpu.edge_type
        
        for idx in range(len(edge_type)):
            h, t, r = edge_index[idx, 0].item(), edge_index[idx, 1].item(), edge_type[idx].item()
            relation_entities[r].add(t)
        
        # Initialize metric accumulators
        all_ranks = []
        running_auc_stats = RunningAUCStats()
        
        # Process in chunks
        chunk_pbar = tqdm(range(0, len(test_triplets), chunk_size), desc="Testing chunks")
        for chunk_start in chunk_pbar:
            chunk_end = min(chunk_start + chunk_size, len(test_triplets))
            chunk_triplets = test_triplets[chunk_start:chunk_end]
            
            # Process chunk in batches
            for batch_start in range(0, len(chunk_triplets), batch_size):
                torch.cuda.empty_cache()
                
                batch_end = min(batch_start + batch_size, len(chunk_triplets))
                batch = chunk_triplets[batch_start:batch_end]
                
                try:
                    # Generate predictions
                    t_batch, _ = tasks.all_negative(test_data_gpu, batch)
                    t_pred = model(test_data_gpu, t_batch)
                    
                    # Create relation-aware mask
                    t_mask = torch.zeros_like(t_pred, dtype=torch.bool)
                    for i, (_, _, r) in enumerate(batch):
                        valid_tails = relation_entities[r.item()]
                        t_mask[i, list(valid_tails)] = True
                    
                    # Get positive indices and compute rankings
                    _, pos_t_index, _ = batch.t()
                    rankings = compute_ranking_corrected(t_pred, pos_t_index, t_mask)
                    all_ranks.extend(rankings.tolist())
                    
                    # Update AUC statistics incrementally
                    scores = t_pred.cpu()
                    labels = torch.zeros_like(scores)
                    for i, pos_idx in enumerate(pos_t_index):
                        labels[i, pos_idx] = 1
                    running_auc_stats.update(scores.numpy(), labels.numpy())
                    
                    # Clean up memory
                    del t_batch, t_pred, t_mask, scores, labels
                    torch.cuda.empty_cache()
                    
                    # Update progress bar with current MRR
                    current_mrr = float(torch.mean(1.0 / torch.tensor(all_ranks, dtype=torch.float)))
                    chunk_pbar.set_postfix({"MRR": f"{current_mrr:.4f}"})
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning("OOM in batch. Reducing batch size...")
                        torch.cuda.empty_cache()
                        batch_size = max(1, batch_size // 2)
                        continue
                    raise e
        
        # Calculate final metrics
        ranks = torch.tensor(all_ranks, dtype=torch.float)
        metrics = {
            'mr': float(torch.mean(ranks)),
            'mrr': float(torch.mean(1.0 / ranks)),
            'hits@1': float(torch.mean((ranks <= 1).float())),
            'hits@3': float(torch.mean((ranks <= 3).float())),
            'hits@10': float(torch.mean((ranks <= 10).float())),
        }
        
        # Add AUC metrics
        auc_metrics = running_auc_stats.compute_metrics()
        metrics.update(auc_metrics)
        
        # Log results
        logger.warning("\nTest Results:")
        for metric, value in metrics.items():
            logger.warning(f"{metric}: {value:.4f}")
    
    return metrics

def evaluate_model_personalized(cfg, patient_id, model, criterion, test_data, patient_data, device, logger):
    """
    Evaluation that incorporates personalized weights using entity type information.
    """
    logger.warning("Starting personalized model evaluation...")
    test_data_gpu = move_data_to_gpu(test_data, device)
    
    # Extract patient data
    prs_values = torch.tensor(patient_data['prs_values'], device=device)
    expr_values = torch.tensor(patient_data['expr_values'], device=device)
    
    # Verify entity types are available
    if not hasattr(test_data_gpu, 'entity_type'):
        logger.warning("Entity type information missing! Personalization may not work correctly.")
    
    model.eval()
    with torch.no_grad():
        # Most of your existing evaluate_model code here...
        
        batch_size = 32
        chunk_size = 1000
        
        # Create test triplets
        test_triplets = torch.cat([
            test_data_gpu.target_edge_index,
            test_data_gpu.target_edge_type.unsqueeze(0)
        ]).t()
        
        # Pre-compute relation-specific entity sets
        relation_entities = defaultdict(set)
        edge_index = test_data_gpu.edge_index.t()
        edge_type = test_data_gpu.edge_type
        
        for idx in range(len(edge_type)):
            h, t, r = edge_index[idx, 0].item(), edge_index[idx, 1].item(), edge_type[idx].item()
            relation_entities[r].add(t)
        
        # Initialize metric accumulators
        all_ranks = []
        running_auc_stats = RunningAUCStats()
        
        # Process in chunks
        chunk_pbar = tqdm(range(0, len(test_triplets), chunk_size), desc="Testing chunks")
        for chunk_start in chunk_pbar:
            chunk_end = min(chunk_start + chunk_size, len(test_triplets))
            chunk_triplets = test_triplets[chunk_start:chunk_end]
            
            # Process chunk in batches
            for batch_start in range(0, len(chunk_triplets), batch_size):
                torch.cuda.empty_cache()
                
                batch_end = min(batch_start + batch_size, len(chunk_triplets))
                batch = chunk_triplets[batch_start:batch_end]
                
                try:
                    # Generate predictions
                    t_batch, _ = tasks.all_negative(test_data_gpu, batch)
                    t_pred = model(test_data_gpu, t_batch)
                    
                    # Apply personalization to the predictions
                    t_pred = apply_personalization(t_pred, batch, test_data_gpu, criterion, prs_values, expr_values, patient_id)
                    
                    # Rest of your existing evaluation code...
                    # Create relation-aware mask
                    t_mask = torch.zeros_like(t_pred, dtype=torch.bool)
                    for i, (_, _, r) in enumerate(batch):
                        valid_tails = relation_entities[r.item()]
                        t_mask[i, list(valid_tails)] = True
                    
                    # Get positive indices and compute rankings
                    _, pos_t_index, _ = batch.t()
                    rankings = compute_ranking_corrected(t_pred, pos_t_index, t_mask)
                    all_ranks.extend(rankings.tolist())
                    
                    # Update AUC statistics incrementally
                    scores = t_pred.cpu()
                    labels = torch.zeros_like(scores)
                    for i, pos_idx in enumerate(pos_t_index):
                        labels[i, pos_idx] = 1
                    running_auc_stats.update(scores.numpy(), labels.numpy())
                    
                    # Clean up memory
                    del t_batch, t_pred, t_mask, scores, labels
                    torch.cuda.empty_cache()
                    
                    # Update progress bar with current MRR
                    current_mrr = float(torch.mean(1.0 / torch.tensor(all_ranks, dtype=torch.float)))
                    chunk_pbar.set_postfix({"MRR": f"{current_mrr:.4f}"})
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning("OOM in batch. Reducing batch size...")
                        torch.cuda.empty_cache()
                        batch_size = max(1, batch_size // 2)
                        continue
                    raise e
                    
        # Calculate final metrics
        ranks = torch.tensor(all_ranks, dtype=torch.float)
        metrics = {
            'mr': float(torch.mean(ranks)),
            'mrr': float(torch.mean(1.0 / ranks)),
            'hits@1': float(torch.mean((ranks <= 1).float())),
            'hits@3': float(torch.mean((ranks <= 3).float())),
            'hits@10': float(torch.mean((ranks <= 10).float())),
        }
        
        # Add AUC metrics
        auc_metrics = running_auc_stats.compute_metrics()
        metrics.update(auc_metrics)
        
        # Log results
        logger.warning("\nTest Results:")
        for metric, value in metrics.items():
            logger.warning(f"{metric}: {value:.4f}")
    
    return metrics

class RunningAUCStats:
    """Helper class to calculate AUC metrics incrementally."""
    def __init__(self):
        self.all_scores = []
        self.all_labels = []
        self.max_stored = 1000000  # Maximum number of predictions to store
    
    def update(self, scores, labels):
        """Update statistics with new batch of predictions."""
        self.all_scores.append(scores.flatten())
        self.all_labels.append(labels.flatten())
        
        # If we've accumulated too many predictions, compute intermediate statistics
        if len(self.all_scores) * self.all_scores[0].size > self.max_stored:
            self.consolidate()
    
    def consolidate(self):
        """Consolidate stored predictions to save memory."""
        scores = np.concatenate(self.all_scores)
        labels = np.concatenate(self.all_labels)
        
        # Randomly sample a subset of predictions to keep
        indices = np.random.choice(len(scores), self.max_stored, replace=False)
        self.all_scores = [scores[indices]]
        self.all_labels = [labels[indices]]
    
    def compute_metrics(self):
        """Compute final AUC metrics."""
        scores = np.concatenate(self.all_scores)
        labels = np.concatenate(self.all_labels)
        return {
            'roc_auc': float(roc_auc_score(labels, scores)),
            'auprc': float(average_precision_score(labels, scores))
        }
    
@torch.no_grad()
def validate(cfg, model, valid_data, device, logger):
    """
    Evaluates model performance with relation-aware filtering.
    """
    model.eval()
    logger.warning("\nRunning validation...")
    
    # Create validation triplets
    valid_triplets = torch.cat([
        valid_data.target_edge_index,
        valid_data.target_edge_type.unsqueeze(0)
    ]).t()
    
    # Pre-compute relation-specific entity sets
    # We need to look at edge_index and edge_type together
    relation_entities = defaultdict(set)
    edge_index = valid_data.edge_index.t()  # Shape: [num_edges, 2]
    edge_type = valid_data.edge_type        # Shape: [num_edges]
    
    # Iterate over the edges using both tensors
    for idx in range(len(edge_type)):
        h = edge_index[idx, 0].item()  # head entity
        t = edge_index[idx, 1].item()  # tail entity
        r = edge_type[idx].item()      # relation type
        relation_entities[r].add(t)     # Add tail entity for this relation
    
    batch_size = 128
    num_triplets = len(valid_triplets)
    all_ranks = []
    
    val_pbar = tqdm(range(0, num_triplets, batch_size),
                   desc="Validating",
                   total=(num_triplets + batch_size - 1) // batch_size)
    
    for start_idx in val_pbar:
        end_idx = min(start_idx + batch_size, num_triplets)
        batch = valid_triplets[start_idx:end_idx]
        
        # Generate predictions
        t_batch, _ = tasks.all_negative(valid_data, batch)
        t_pred = model(valid_data, t_batch)
        
        # Create relation-aware mask
        t_mask = torch.zeros_like(t_pred, dtype=torch.bool)
        for i, (_, _, r) in enumerate(batch):
            valid_tails = relation_entities[r.item()]
            t_mask[i, list(valid_tails)] = True
        
        # Get positive indices and compute rankings
        _, pos_t_index, _ = batch.t()
        rankings = compute_ranking_corrected(t_pred, pos_t_index, t_mask)
        all_ranks.extend(rankings.tolist())
        
        # Update progress
        current_mrr = float(torch.mean(1.0 / torch.tensor(all_ranks, dtype=torch.float)))
        val_pbar.set_postfix({"MRR": f"{current_mrr:.4f}"})
        
        del t_batch, t_pred
        torch.cuda.empty_cache()
    
    metrics = calculate_metrics(all_ranks, logger)
    log_metrics(metrics, logger)
    
    return metrics

def compute_ranking_corrected(scores, positive_index, mask, debug=False, logger=None):
    """
    Computes rankings with proper handling of ties and filtering.
    
    Args:
        scores: Tensor of shape [batch_size, num_entities] containing prediction scores
        positive_index: Tensor of shape [batch_size] containing indices of positive entities
        mask: Boolean tensor of shape [batch_size, num_entities] for filtering
        debug: Whether to print debug information
        logger: Logger for debug output
    
    Returns:
        Tensor of shape [batch_size] containing ranks of positive entities
    """
    batch_size = scores.size(0)
    
    # Clone and mask scores
    scores_masked = scores.clone()
    scores_masked[~mask] = float('-inf')
    
    if debug and logger is not None:
        logger.warning(f"Score statistics before ranking:")
        logger.warning(f"Mean score: {scores_masked[mask].mean():.4f}")
        logger.warning(f"Std score: {scores_masked[mask].std():.4f}")
    
    # Get positive scores
    positive_scores = scores_masked[torch.arange(batch_size), positive_index]
    
    # For each row, count entities that score STRICTLY HIGHER than the positive entity
    strictly_higher = (scores_masked > positive_scores.unsqueeze(1)).sum(dim=1)
    
    # For each row, count entities that have EXACTLY THE SAME score
    equal_scores = (scores_masked == positive_scores.unsqueeze(1)).sum(dim=1)
    
    # Compute rank using the optimistic tie-breaking strategy:
    # rank = #(strictly higher scores) + 1
    ranks = strictly_higher + 1
    
    if debug and logger is not None:
        logger.warning(f"\nRanking statistics:")
        logger.warning(f"Number of strictly higher scores: {strictly_higher[:5]}")
        logger.warning(f"Number of equal scores: {equal_scores[:5]}")
        logger.warning(f"Resulting ranks: {ranks[:5]}")
    
    return ranks

def calculate_metrics(ranks, logger):
    """
    Calculates metrics with debugging information.
    """
    ranks = torch.tensor(ranks, dtype=torch.float)
    logger.warning(f"\nRanks statistics:")
    logger.warning(f"Min rank: {ranks.min().item()}")
    logger.warning(f"Max rank: {ranks.max().item()}")
    logger.warning(f"Mean rank: {ranks.mean().item():.4f}")
    
    return {
        'mrr': float(torch.mean(1.0 / ranks)),
        'hits@1': float(torch.mean((ranks <= 1).float())),
        'hits@3': float(torch.mean((ranks <= 3).float())),
        'hits@10': float(torch.mean((ranks <= 10).float()))
    }

def log_metrics(metrics, logger):
    """Helper function to log metrics."""
    logger.warning("\nValidation Results:")
    for metric, value in metrics.items():
        logger.warning(f"{metric}: {value:.4f}")


def compute_ranking(scores, positive_index, mask):
    """
    Computes the ranking of the positive entity among all entities.
    A rank of 1 means the model scored the true entity highest.
    """
    # Apply mask to scores
    scores[~mask] = float('-inf')
    
    # Get score of positive entity
    positive_score = scores[torch.arange(len(scores)), positive_index]
    
    # Count how many entities scored higher than or equal to the positive entity
    rank = torch.sum((scores >= positive_score.unsqueeze(1)), dim=1)
    
    return rank
def get_training_batches(data, batch_size, device=None):
    """
    Creates batches of training triplets (head, relation, tail) from the knowledge graph.
    
    This function takes the full graph data and creates manageable batches of triplets
    for training. It handles the specific format needed for knowledge graph embedding
    training where each batch contains positive triplets from the graph.
    
    Args:
        data: PyG Data object containing:
            - edge_index: Tensor of shape [2, num_edges] containing source and target nodes
            - edge_type: Tensor of shape [num_edges] containing relation types
        batch_size: Number of triplets per batch
        device: torch device (optional). If provided, moves batches to this device
    
    Returns:
        Iterator yielding batches of triplets of shape [batch_size, 3] where:
            - [:, 0] is head entities
            - [:, 1] is tail entities
            - [:, 2] is relation types
    """
    # Combine edge_index and edge_type into triplets
    triplets = torch.cat([
        data.edge_index.t(),  # Shape: [num_edges, 2]
        data.edge_type.unsqueeze(-1)  # Shape: [num_edges, 1]
    ], dim=1)  # Final shape: [num_edges, 3]
    
    # Get total number of triplets and batches
    num_triplets = triplets.size(0)
    num_batches = (num_triplets + batch_size - 1) // batch_size  # Ceiling division
    
    # Create random permutation for shuffling
    indices = torch.randperm(num_triplets)
    
    for batch_idx in range(num_batches):
        # Get start and end indices for this batch
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_triplets)
        
        # Select batch indices using the random permutation
        batch_indices = indices[start_idx:end_idx]
        
        # Create the batch of triplets
        batch = triplets[batch_indices]
        
        # Move to device if specified
        if device is not None:
            batch = batch.to(device)
        
        yield batch

def get_training_batches_with_negatives(data, batch_size, num_negatives, device=None):
    """
    Creates batches of training triplets with corresponding negative samples.
    
    This enhanced version of get_training_batches also generates negative samples
    for each positive triplet in the batch, which is often needed for training
    knowledge graph embedding models.
    
    Args:
        data: PyG Data object containing edge_index and edge_type
        batch_size: Number of positive triplets per batch
        num_negatives: Number of negative samples per positive triplet
        device: torch device (optional)
    
    Returns:
        Iterator yielding tuples of (pos_batch, neg_batch) where:
            - pos_batch has shape [batch_size, 3]
            - neg_batch has shape [batch_size, num_negatives, 3]
    """
    # Get total number of entities for negative sampling
    num_entities = data.num_nodes
    
    # Create batches of positive triplets
    for pos_batch in get_training_batches(data, batch_size, device):
        batch_size_curr = pos_batch.size(0)  # Might be smaller for last batch
        
        # Generate negative samples by corrupting either head or tail
        neg_batch = []
        for _ in range(num_negatives):
            # Randomly decide whether to corrupt head or tail
            corrupt_head = torch.rand(batch_size_curr) < 0.5
            
            # Create corrupted triplets
            neg_triplets = pos_batch.clone()
            
            # Corrupt heads
            head_mask = corrupt_head
            neg_triplets[head_mask, 0] = torch.randint(
                num_entities, (head_mask.sum(),), device=device
            )
            
            # Corrupt tails
            tail_mask = ~corrupt_head
            neg_triplets[tail_mask, 1] = torch.randint(
                num_entities, (tail_mask.sum(),), device=device
            )
            
            neg_batch.append(neg_triplets)
        
        # Stack negative samples
        neg_batch = torch.stack(neg_batch, dim=1)  # [batch_size, num_negatives, 3]
        
        yield pos_batch, neg_batch


def save_model(model, model_file_name, state_dict_name, cfg, logger):

    # Save both the full model and just the state dict
    save_dir = cfg.fm_dir 
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the full model
    model_path = os.path.join(save_dir, f'{model_file_name}.pt')
    torch.save(model, model_path)
    logger.warning(f"Saved full model to {model_path}")
    
    # Save just the state dict (more portable)
    state_dict_path = os.path.join(save_dir, f'{state_dict_name}.pt')
    torch.save(model.state_dict(), state_dict_path)
    logger.warning(f"Saved model state dict to {state_dict_path}")

def load_model(cfg, logger, device, model_path):
    """
    Load a saved model onto the specified device.
    
    Args:
        cfg: Configuration object
        logger: Logger object for logging messages
        device: torch.device already specified from util.get_device(cfg)
        model_path: Direct path to the saved model file
        
    Returns:
        Loaded PyTorch model on the specified device
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
        
    # Load model to CPU first to avoid device mismatch
    model = torch.load(model_path, map_location='cpu')
    # Move the entire model to the specified device
    model = model.to(device)
    logger.info(f"Loaded model from {model_path} to {device}")
    
    # Double check that all model parameters are on the correct device
    for param in model.parameters():
        if param.device != device:
            logger.warning(f"Found parameter on {param.device}, moving to {device}")
            param.data = param.data.to(device)
    
    return model

def train_foundation_model(cfg, cfg_ultra, model, train_data_cpu, valid_data_cpu, model_file_name, state_dict_name, device, logger):
    """
    Trains the foundation model with memory-efficient implementation and progress tracking.
    Includes nested progress bars to monitor overall training progress and batch-level progress.
    """
    logger.warning("Starting foundation model training...")
    
    # Data movement and initialization remains the same...
    train_data_gpu = move_data_to_gpu(train_data_cpu, device)
    valid_data_gpu = move_data_to_gpu(valid_data_cpu, device)
    
    # Verify data location...
    for key, value in train_data_gpu:
        if torch.is_tensor(value):
            logger.warning(f"{key} is on: {value.device}")
    
    # Initialize training components...
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Initialize early stopping...
    best_val_mrr = float('-inf')
    best_model_state = None
    patience = 6
    epochs_no_improve = 0
    
    # Create training triplets with progress bar for data preparation
    logger.warning("Preparing training triplets...")
    chunk_size = min(50000, train_data_gpu.target_edge_index.size(1))
    train_triplets = []
    
    # Progress bar for triplet preparation
    chunks = range(0, train_data_gpu.target_edge_index.size(1), chunk_size)
    for start_idx in tqdm(chunks, desc="Preparing data chunks", unit="chunk"):
        end_idx = min(start_idx + chunk_size, train_data_gpu.target_edge_index.size(1))
        chunk_triplets = torch.cat([
            train_data_gpu.target_edge_index[:, start_idx:end_idx],
            train_data_gpu.target_edge_type[start_idx:end_idx].unsqueeze(0)
        ]).t().to(device)
        train_triplets.append(chunk_triplets)
    
    train_triplets = torch.cat(train_triplets, dim=0)
    
    # Get training parameters
    num_epochs = cfg.train.num_epoch #getattr(cfg, 'epochs', 5)
    batch_size = cfg.train.batch_size #getattr(cfg, 'batch_size', 64)
    batch_per_epoch = cfg.train.batch_per_epoch #getattr(cfg, 'batch_per_epoch', float('inf'))
    
    # Main training loop with epoch-level progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Training epochs", unit="epoch")
    for epoch in epoch_pbar:
        model.train()
        epoch_losses = []
        
        # Calculate number of batches for this epoch
        num_triplets = train_triplets.size(0)
        indices = torch.randperm(num_triplets, device=device)

        # This is the key change - we calculate total batches based on batch_per_epoch
        total_possible_batches = num_triplets // batch_size
        actual_batches = min(total_possible_batches, batch_per_epoch)

        print('actual_batches', actual_batches)

        batch_pbar = tqdm(
            range(0, actual_batches * batch_size, batch_size),  # Only iterate the number of batches we want
            desc=f"Epoch {epoch+1} batches",
            total=actual_batches,  # Set the total to our limited number
            leave=False,
            unit="batch"
        )
        
        for start_idx in batch_pbar:
            try:
                # Clear cache before processing each batch
                torch.cuda.empty_cache()
                
                # Process batch
                batch_idx = indices[start_idx:start_idx + batch_size]
                batch = train_triplets[batch_idx].unsqueeze(1)
                
                # Generate negative samples
                batch = tasks.negative_sampling(
                    train_data_gpu,
                    batch,
                    cfg.task.num_negative,
                    strict=cfg.task.strict_negative
                )
                
                # Forward pass and loss calculation
                pred = model(train_data_gpu, batch)
                target = torch.zeros_like(pred, device=device)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(pred, target)
                
                # Optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update progress bar with current loss
                batch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # Record loss and clear memory
                epoch_losses.append(loss.item())
                del pred, target, loss
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning("OOM occurred, clearing cache and reducing batch size")
                    torch.cuda.empty_cache()
                    batch_size = batch_size // 2
                    if batch_size < 1:
                        raise RuntimeError("Batch size too small, cannot continue")
                    continue
                else:
                    raise e
            
            if (start_idx // batch_size) >= batch_per_epoch:
                break
        
        # Update epoch progress bar with average loss
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})
        
        # Validation with its own progress bar
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                torch.cuda.empty_cache()
                logger.warning("\nRunning validation...")
                val_metrics = validate(cfg, model, valid_data_gpu, device=device, logger=logger) #val_metrics = test(cfg, model, valid_data_gpu, device=device, logger=logger)
                val_mrr = val_metrics['mrr'] if isinstance(val_metrics, dict) else val_metrics
                
                # Early stopping logic
                if val_mrr > best_val_mrr:
                    logger.warning(f"New best validation MRR: {val_mrr:.4f}")
                    best_val_mrr = val_mrr
                    best_model_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    logger.warning(f"No improvement for {epochs_no_improve} epochs")
                
                if epochs_no_improve >= patience:
                    logger.warning("Early stopping triggered")
                    break
    
    # Load best model and clean up
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    logger.warning(f"Training complete. Best validation MRR: {best_val_mrr:.4f}")
    
    # save the FM
    save_model(model, model_file_name, state_dict_name, cfg_ultra, logger)

    return model


###

def ultra_fast_negative_sampling(data, batch, num_negative, strict=False):
    """
    Ultra-fast negative sampling implementation that prioritizes speed over strict filtering.
    
    Args:
        data: The knowledge graph data
        batch: Batch of positive triples
        num_negative: Number of negative samples per positive triple
        strict: Whether to use strict negative sampling (ignored for speed)
        
    Returns:
        Tensor with positive and negative samples
    """
    # Reshape batch if needed
    if batch.dim() == 2:  # [batch_size, 3]
        batch = batch.unsqueeze(1)  # [batch_size, 1, 3]
    
    batch_size = batch.size(0)
    num_nodes = data.num_nodes
    
    # Extract the positive triples
    positive_triples = batch.view(batch_size, -1, 3)[:, 0].long()  # [batch_size, 3]
    
    # Create the result tensor [batch_size, 1 + num_negative, 3]
    result = torch.zeros(batch_size, 1 + num_negative, 3, dtype=torch.long, device=batch.device)
    
    # Set positive triples
    result[:, 0] = positive_triples
    
    # Repeat head and relation for all negative samples
    result[:, 1:, 0] = positive_triples[:, 0].view(-1, 1).expand(-1, num_negative)
    result[:, 1:, 2] = positive_triples[:, 2].view(-1, 1).expand(-1, num_negative)
    
    # Generate random tails for all negatives at once
    result[:, 1:, 1] = torch.randint(num_nodes, (batch_size, num_negative), 
                                    dtype=torch.long, device=batch.device)
    
    # No filtering for speed - accept some false negatives
    return result

def fine_tune_patient_specific(cfg, model, train_data, patient_data, model_name, state_dict_name, l1, l2, device, logger, batch_per_epoch=None):
    """
    Highly optimized patient-specific fine-tuning that freezes the base model
    and only trains the patient-specific weights.
    
    Args:
        cfg: Configuration object containing hyperparameters
        model: The pre-trained model to fine-tune
        train_data: Training data for the knowledge graph
        patient_data: Patient-specific data containing PRS values and expression data
        model_name: Name for saving the model checkpoint
        state_dict_name: Name for saving the model state dict
        device: Device to use for computation
        logger: Logger for outputting information
        batch_per_epoch: Number of batches per epoch (optional)
        
    Returns:
        The fine-tuned model with patient-specific adaptations
    """
    from tqdm import tqdm, trange
    import time
    
    # 1. Quick return if no training needed
    if cfg.train.num_epoch == 0:
        logger.warning("No fine-tuning epochs specified, returning original model")
        return model

    # 2. Initialize key variables and move data to device
    train_data = train_data.to(device)
    
    # Extract patient-specific data and move to device
    prs_values = torch.tensor(patient_data['prs_values'], device=device)
    expr_values = torch.tensor(patient_data['expr_values'], device=device)
    disease_names = patient_data['disease_names']
    protein_names = patient_data['protein_names']
    
    # 3. Set up early stopping
    best_val_loss = float('inf')
    best_model_state = None
    best_criterion_state = None  # We'll save criterion state separately
    epochs_no_improve = 0
    patience = 5
    stop_training = False

    # 4. Get distributed training information
    world_size = util.get_world_size()
    rank = util.get_rank()
    is_main_process = (rank == 0)

    # 5. Prepare data loader (keep data on CPU for DataLoader)
    # Create the triplets on CPU first
    train_triplets_cpu = torch.cat([
        train_data.target_edge_index.cpu(), 
        train_data.target_edge_type.cpu().unsqueeze(0)
    ]).t()
    
    sampler = torch_data.DistributedSampler(train_triplets_cpu, world_size, rank)
    
    # Use pin_memory only for CPU tensors going to GPU
    pin_memory = (device.type == 'cuda')
    
    # Increase batch size for faster processing
    batch_size = min(256, len(train_triplets_cpu) // world_size)  # Ensure it's divisible by world_size
    if batch_size < 1:
        batch_size = 1
    
    if is_main_process:
        logger.warning(f"Using batch size of {batch_size} for fine-tuning")
    
    train_loader = torch_data.DataLoader(
        train_triplets_cpu, 
        batch_size, 
        sampler=sampler,
        pin_memory=pin_memory,
        num_workers=0  # In-process loading for small datasets
    )

    batch_per_epoch = batch_per_epoch or len(train_loader)
    
    logger.warning(f"Ploss lambda1 for PRS: {l1}, lambda2 for biomarker exp: {l2} for fine-tuning")
    # 6. Initialize personalized loss
    criterion = PersonalizedRepurposingLoss(
        disease_names=disease_names,
        protein_names=protein_names,
        lambda1=l1,#0.01,
        lambda2=l2
    ).to(device)

    # 7. CRITICAL OPTIMIZATION: Freeze the base model parameters
    # This is the key to making fine-tuning much faster
    model.eval()  # Set model to evaluation mode
    for param in model.parameters():
        param.requires_grad = False
    
    if is_main_process:
        # Count number of frozen parameters
        frozen_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in criterion.parameters() if p.requires_grad)
        logger.warning(f"Frozen {frozen_params} base model parameters")
        logger.warning(f"Training only {trainable_params} patient-specific parameters")
        logger.warning(f"Reduction ratio: {frozen_params / (trainable_params or 1):.1f}x fewer parameters")

    # 8. Set up optimizer with only patient-specific parameters
    # Can use higher learning rate for fewer parameters
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, criterion.parameters()),
        lr=2e-3,  # Higher learning rate since we're training fewer parameters
        weight_decay=1e-5
    )
    
    # 9. Set up learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=2, verbose=is_main_process
    )

    # 10. Parallelize model if needed (though parameters are frozen)
    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model
    
    # 11. Log initial information
    if is_main_process:
        logger.warning("-" * 30)
        logger.warning(f"Starting fine-tuning for {cfg.train.num_epoch} epochs with frozen base model")
        logger.warning(f"Training only patient-specific weights")
    
    # 12. Training loop
    epoch_iterator = trange(cfg.train.num_epoch, desc="Epochs", disable=not is_main_process)
    
    # Track timing information for optimization
    sample_times = []
    forward_times = []
    loss_times = []
    backward_times = []
    
    for epoch in epoch_iterator:
        # Set criterion to training mode (model stays in eval mode)
        criterion.train()
        
        # Initialize metrics
        losses = []
        epoch_start_time = time.time()
        
        # Set sampler epoch
        sampler.set_epoch(epoch)
        
        # Batch iteration with tqdm
        batch_iterator = tqdm(
            train_loader, 
            desc=f"Epoch {epoch}", 
            disable=not is_main_process,
            leave=False
        )
        
        # 13. Process batches
        for batch_idx, batch in enumerate(batch_iterator):
            # Move batch to device - now safe to use non_blocking with pin_memory
            batch = batch.to(device, non_blocking=pin_memory)
            
            # Generate negative samples
            sample_start = time.time()
            batch = ultra_fast_negative_sampling(
                train_data, 
                batch, 
                cfg.task.num_negative,
                strict=False
            )
            sample_times.append(time.time() - sample_start)
            
            # Forward pass - with no gradient calculation for the frozen model
            forward_start = time.time()
            with torch.no_grad():
                pred = parallel_model(train_data, batch)
            forward_times.append(time.time() - forward_start)
            
            # Extract positive and negative scores
            pos_scores = pred[:, 0]
            neg_scores = pred[:, 1:]
            
            # Calculate loss - only this part needs gradients
            loss_start = time.time()
            loss = criterion(pos_scores, neg_scores, prs_values, expr_values)
            loss_times.append(time.time() - loss_start)
            
            # Backward pass and optimization - only for criterion parameters
            backward_start = time.time()
            loss.backward()
            
            # Skip gradient clipping since we have fewer parameters
            
            # Update weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            backward_times.append(time.time() - backward_start)

            # Update progress bar
            if len(sample_times) > 0 and len(forward_times) > 0 and len(backward_times) > 0:
                batch_iterator.set_postfix(
                    loss=f"{loss.item():.4f}",
                    smpl=f"{sample_times[-1]:.3f}s",
                    fwd=f"{forward_times[-1]:.3f}s",
                    bwd=f"{backward_times[-1]:.3f}s"
                )
            
            losses.append(loss.item())

            if batch_idx >= 100:  # Process only 100 batches per epoch
                logger.warning(f"Reached batch limit of 100. Breaking early.")
                break
        
        # 14. End of epoch processing
        if is_main_process:
            avg_loss = sum(losses) / len(losses)
            epoch_time = time.time() - epoch_start_time
            examples_per_sec = len(losses) * batch_size / epoch_time
            
            # Print epoch stats in tabular format
            logger.warning(f"Epoch {epoch} - Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s, Examples/s: {examples_per_sec:.1f}")
            
            # Report timing breakdown
            if len(sample_times) > 0 and len(forward_times) > 0 and len(loss_times) > 0 and len(backward_times) > 0:
                avg_sample = sum(sample_times) / len(sample_times)
                avg_forward = sum(forward_times) / len(forward_times)
                avg_loss_calc = sum(loss_times) / len(loss_times)
                avg_backward = sum(backward_times) / len(backward_times)
                total = avg_sample + avg_forward + avg_loss_calc + avg_backward
                
                logger.warning(f"Time breakdown - Sampling: {avg_sample:.4f}s ({avg_sample/total*100:.1f}%), "
                              f"Forward: {avg_forward:.4f}s ({avg_forward/total*100:.1f}%), "
                              f"Loss calc: {avg_loss_calc:.4f}s ({avg_loss_calc/total*100:.1f}%), "
                              f"Backward: {avg_backward:.4f}s ({avg_backward/total*100:.1f}%)")
            
            # Update learning rate scheduler
            scheduler.step(avg_loss)
            
            # Early stopping check
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                epochs_no_improve = 0
                # Save only the criterion state since the model is frozen
                best_criterion_state = copy.deepcopy(criterion.state_dict())
                best_epoch = epoch
                
                logger.warning(f"-> New best model! Saving checkpoint")
                # Save both model and criterion for completeness
                state = {
                    "model": model.state_dict(),  # Frozen model
                    "criterion": criterion.state_dict(),  # Patient-specific weights
                    "epoch": epoch,
                    "best_loss": best_val_loss
                }
                torch.save(state, f"{model_name}_{epoch}.pth")
            else:
                epochs_no_improve += 1
                logger.warning(f"No improvement for {epochs_no_improve}/{patience} epochs")
            
            # Early stopping
            if epochs_no_improve >= patience:
                logger.warning(f"Early stopping triggered after {epoch+1} epochs")
                # Only restore criterion state since model was frozen
                criterion.load_state_dict(best_criterion_state)
                stop_training = True
        
        # Synchronize processes
        util.synchronize()
        
        # Break if early stopping triggered
        if stop_training:
            break
    
    # 15. Final processing
    if is_main_process:
        # Save final state dictionary - focus on criterion which contains patient-specific weights
        logger.warning(f"Saving final state dictionary to {state_dict_name}.pth")
        state = {
            "model": model.state_dict(),
            "criterion": criterion.state_dict(),
            "patient_specific_only": True
        }
        torch.save(state, f"{state_dict_name}.pth")
        
        # Load best criterion if available
        if best_criterion_state is not None:
            logger.warning(f"Loading best patient weights from epoch {best_epoch}")
            criterion.load_state_dict(best_criterion_state)
        
        # Log personalized weights
        if hasattr(criterion, 'disease_weights') and criterion.disease_weights is not None:
            top_diseases = torch.argsort(criterion.disease_weights, descending=True)#[:5]
            logger.warning("Top disease weights:")
            for i in top_diseases:
                idx = i.item()
                logger.warning(f"  {disease_names[idx]}: {criterion.disease_weights[idx].item():.6f}")
        
        if hasattr(criterion, 'protein_weights') and criterion.protein_weights is not None:
            top_proteins = torch.argsort(criterion.protein_weights, descending=True)#[:5]
            logger.warning("Top protein weights:")
            for i in top_proteins:
                idx = i.item()
                logger.warning(f"  {protein_names[idx]}: {criterion.protein_weights[idx].item():.6f}")
    
    # Final synchronization
    util.synchronize()
    
    # Return both the model and the criterion for the caller to use
    return model, criterion
