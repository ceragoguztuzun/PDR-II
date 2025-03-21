#evaluation.py
import torch
import numpy as np
from utils import get_disease_relevant_drugs

##################
''' EVALUATION '''
##################

def evaluate_drug_repurposing(model, test_data, target_disease, patient_data, device):
    """
    Evaluate drug repurposing predictions for a specific disease
    """
    model.eval()
    with torch.no_grad():
        # Get candidate drugs for the disease
        candidate_drugs = get_disease_relevant_drugs(test_data, target_disease)
        
        # Generate scores for all candidates
        scores = []
        for drug in candidate_drugs:
            score = predict_drug_score(model, test_data, drug, patient_data)
            scores.append((drug, score))
            
        # Sort by score
        rankings = sorted(scores, key=lambda x: x[1], reverse=True)
        
        # Calculate metrics
        metrics = calculate_ranking_metrics(rankings)
        
        return rankings, metrics

def calculate_ranking_metrics(rankings, ground_truth=None):

    """
    Calculate standard ranking metrics
    """
    metrics = {}
    
    if ground_truth is not None:
        # Calculate precision@k
        for k in [5, 10, 20]:
            top_k = set([drug for drug, _ in rankings[:k]])
            precision = len(top_k.intersection(ground_truth)) / k
            metrics[f'precision@{k}'] = precision
        
        # Calculate MRR
        mrr = 0
        for i, (drug, _) in enumerate(rankings):
            if drug in ground_truth:
                mrr += 1.0 / (i + 1)
        metrics['mrr'] = mrr / len(ground_truth)
    
    return metrics

