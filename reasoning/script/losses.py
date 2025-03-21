#losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

##############
''' LOSSES '''
##############

class DrugRepurposingLoss(nn.Module):
    """Base class for drug repurposing loss functions"""
    def __init__(self):
        super(DrugRepurposingLoss, self).__init__()
    
    def forward(self, pos_scores, neg_scores):
        raise NotImplementedError

class StandardKGLoss(DrugRepurposingLoss):
    """Standard knowledge graph loss for foundation model training"""
    def __init__(self, margin=1.0):
        super(StandardKGLoss, self).__init__()
        self.margin = margin
    
    def forward(self, pos_scores, neg_scores):
        # Reshape scores if needed
        pos_scores = pos_scores.view(-1, 1)
        neg_scores = neg_scores.view(pos_scores.size(0), -1)
        
        # Expand positive scores to match negative scores shape
        pos_scores = pos_scores.expand_as(neg_scores)
        
        # Calculate margin-based loss
        difference = pos_scores - neg_scores
        loss = torch.clamp(self.margin - difference, min=0)
        return loss.mean()

class PersonalizedRepurposingLoss(DrugRepurposingLoss):
    """Personalized loss incorporating patient data for fine-tuning"""
    def __init__(self, disease_names, protein_names, lambda1=0.01, lambda2=0.01):
        super(PersonalizedRepurposingLoss, self).__init__()
        self.eps = 1e-10
        
        # Initialize learnable weights for patient features
        if len(disease_names) > 0:
            self.disease_weights = nn.Parameter(torch.randn(len(disease_names)) * 0.01)
        else:
            self.disease_weights = None
            
        if len(protein_names) > 0:
            self.protein_weights = nn.Parameter(torch.randn(len(protein_names)) * 0.01)
        else:
            self.protein_weights = None
        
        self.disease_names = disease_names
        self.protein_names = protein_names
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.weight_decay = 0.001

    def forward(self, pos_scores, neg_scores, prs_values, expr_values):
        # Base KG loss
        base_loss = StandardKGLoss()(pos_scores, neg_scores)
        
        # Initialize personalization terms
        prs_term = torch.tensor(0.0, device=pos_scores.device)
        expr_term = torch.tensor(0.0, device=pos_scores.device)
        
        # PRS impact
        if self.disease_weights is not None and len(prs_values) > 0:
            disease_weights_norm = F.normalize(self.disease_weights, p=2, dim=0)
            prs_impact = torch.sum(disease_weights_norm * prs_values)
            prs_term = torch.tanh(prs_impact)
            
        # Expression impact
        if self.protein_weights is not None and len(expr_values) > 0:
            protein_weights_norm = F.normalize(self.protein_weights, p=2, dim=0)
            expr_impact = torch.sum(protein_weights_norm * expr_values)
            expr_term = torch.tanh(expr_impact)
        
        # Combine all terms
        total_loss = (
            base_loss + 
            self.lambda1 * prs_term +
            self.lambda2 * expr_term
        )
        
        return total_loss

