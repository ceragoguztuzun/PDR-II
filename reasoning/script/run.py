#run.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.data import Data
import numpy as np
from collections import defaultdict
import copy
import time
from tqdm import tqdm
import argparse
from pathlib import Path
from config import load_config
import traceback

from model import initialize_model
from losses import DrugRepurposingLoss, StandardKGLoss, PersonalizedRepurposingLoss
from training import train_foundation_model, evaluate_model, evaluate_model_personalized, fine_tune_patient_specific, load_model, case_study_rank_drugs
from evaluation import evaluate_drug_repurposing, calculate_ranking_metrics
from utils import (
    load_patient_data,
    load_data,
    print_results,
    load_dataset_and_dictionaries,
    move_data_to_gpu
)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import tasks, util

############
''' MAIN '''
############

def run_drug_repurposing(cfg, cfg_ultra, patient_id, target_disease, logger, w_ploss=None, l1=None, l2=None):
    """Main workflow following the original code's data handling pattern."""
    # Load all data first, keeping on CPU
    train_data_cpu, valid_data_cpu, test_data_cpu, filtered_data_cpu, \
    entity_dict_f, relation_dict_f, entity_dict_i, relation_dict_i = \
        load_dataset_and_dictionaries(cfg, patient_id)
    
    # Initialize model from checkpoint (ultra or our FM)
    device = util.get_device(cfg)
    model = initialize_model(cfg, logger).to(device)

    if w_ploss: #Train with PLoss
        # Load patient data (Robin, Usopp, Zoro)
        patient_data = load_patient_data(f'/usr/homes/cxo147/ismb/KGIA/reasoning/patient_data/patient_{patient_id}_data.pkl')

        # Patient-specific fine-tuning
        model, criterion = fine_tune_patient_specific(
            cfg, model, train_data_cpu, patient_data, 
            f'p{patient_id}_wploss_{l1}_{l2}_model', f'p{patient_id}_wploss_{l1}_{l2}_state_dict', 
            l1,l2,
            device, logger
        )

        # Test with personalized rankings
        logger.warning("\nEvaluating model with personalization...")
        test_metrics = evaluate_model_personalized(cfg, patient_id, model, criterion, test_data_cpu, patient_data, device, logger)

        # case study (Rank drugs for AD)
        ##ranked_drugs = case_study_rank_drugs(cfg, patient_id, model, criterion, test_data_cpu, patient_data, device, logger)

    else: #Train without Ploss - model training (FM training or Pmodel fine-tuning)
        model = train_foundation_model(
            cfg, cfg_ultra, model, train_data_cpu, valid_data_cpu, 
            f'p{patient_id}_model', f'p{patient_id}_state_dict', #model_name and state_dict name to be saved
            device, logger
        )
        
        # test general LP performance
        logger.warning("\nEvaluating model on test set...")
        test_metrics = evaluate_model(cfg, model, test_data_cpu, device, logger)

    logger.warning(f"Final test metrics: {test_metrics}")

    
    return 0,0 #rankings, metrics
    
    
if __name__ == "__main__":
    try:
        # Initialize configuration and logging from command line args
        args, vars = util.parse_args()
        cfg = util.load_config(args.config, context=vars)
        cfg_ultra = load_config()
        logger = util.get_root_logger()

        rankings, metrics = run_drug_repurposing(
            cfg_ultra=cfg_ultra,
            cfg=cfg,
            patient_id=args.patient_id,
            target_disease="Alzheimer's",
            logger=logger,
            w_ploss=args.use_ploss, #whether we want to fine-tine pmodel with ploss
            l1=args.lambda1_prs, #lambda values
            l2=args.lambda2_biomarker
        )
        
        print_results(rankings, metrics)

    except Exception as e:
        logger.error(f"\nError during execution: {str(e)}")
        logger.error(traceback.format_exc())
        raise
        
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.warning("\nExperiment completed.")
