#utils.py
import torch
from torch_geometric.data import Data
import numpy as np
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import tasks, util

###############
''' HELPERZ '''
###############

def print_results(rankings, metrics):
    #to be implemented
    print(f'print_results: {rankings}, {metrics}')

def load_data():
    pass


def load_patient_data(filepath):
    """
    Load patient data from a pickle file.
    
    Args:
        filepath (str): Path to the pickle file containing patient data
        
    Returns:
        dict: A dictionary containing patient data with the following keys:
            - 'prs_values': Polygenic risk scores
            - 'expr_values': Expression values
            - 'disease_names': List of disease names
            - 'protein_names': List of protein names
    """
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Patient data file not found at: {filepath}")
    
    # Load the pickle file
    try:
        with open(filepath, 'rb') as f:
            patient_data = pickle.load(f)
    except Exception as e:
        raise Exception(f"Error loading patient data: {str(e)}")
    
    # Validate the required fields
    required_fields = ['prs_values', 'expr_values', 'disease_names', 'protein_names']
    for field in required_fields:
        if field not in patient_data:
            raise ValueError(f"Patient data is missing required field: {field}")
    
    return patient_data
    
def get_disease_relevant_drugs(test_data, target_disease):
    """Gets list of candidate drugs relevant to target disease."""
    pass

def get_drug_node_indices(data, relation_dict_f):
    """Gets indices of all drug nodes in the graph."""
    pass

def get_patient_node_id(data, relation_dict_f):
    """Gets the patient node index in the graph."""
    pass

def create_patient_drug_pairs(patient_id, drug_indices, device):
    """Creates positive (patient->drug) pairs for training."""
    pass

def load_relation_dicts(processed_dir):
    dicts_path = os.path.join(processed_dir, "entity_relation_dicts.pt")
    ##print(f"Loading entity and relation dictionaries from: {dicts_path}")
    
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

def move_data_to_gpu(data, device):
    """
    Moves all attributes of a PyG Data object to the specified device.
    This ensures all tensors, including nested ones in custom attributes,
    are properly moved to GPU.
    
    Args:
        data: PyG Data object containing graph information
        device: torch device to move the data to
    
    Returns:
        PyG Data object with all tensors moved to the specified device
    """
    # First, create a new Data object on the specified device
    data_gpu = data.clone()
    
    # Move all tensor attributes to GPU
    for key, value in data_gpu:
        if torch.is_tensor(value):
            data_gpu[key] = value.to(device)
        elif isinstance(value, dict):
            # Handle nested dictionaries
            data_gpu[key] = {k: v.to(device) if torch.is_tensor(v) else v 
                            for k, v in value.items()}
    
    # Specifically ensure relation_graph is moved to GPU if it exists
    if hasattr(data_gpu, 'relation_graph'):
        data_gpu.relation_graph = move_data_to_gpu(data_gpu.relation_graph, device)
    
    return data_gpu

def load_dataset_and_dictionaries(cfg, pid):
    """Load dataset and entity/relation dictionaries with the same pattern as the original code."""
    # Load the base dataset using the existing util function
    dataset = util.build_dataset(cfg)
    
    # Split into train/valid/test, keeping on CPU initially
    train_data_cpu, valid_data_cpu, test_data_cpu = dataset[0], dataset[1], dataset[2]
    
    # Load dictionaries from the processed directory
    processed_dir = f"/usr/homes/cxo147/git/ULTRA/kg-datasets/biomedical_KG_{pid}/processed/"
    entity_dict_f, relation_dict_f, entity_dict_i, relation_dict_i = load_relation_dicts(processed_dir)
    
    # Create filtered data using the same pattern as original code
    filtered_data_cpu = Data(
        edge_index=dataset._data.target_edge_index,
        edge_type=dataset._data.target_edge_type,
        num_nodes=dataset[0].num_nodes
    )
    
    return (train_data_cpu, valid_data_cpu, test_data_cpu, filtered_data_cpu,
            entity_dict_f, relation_dict_f, entity_dict_i, relation_dict_i)


##################

def extract_entity_disease_relationships(data, disease_entities, disease_name_to_idx, entity_dict_forward):
    """
    Extract relationships between entities and diseases.
    
    Args:
        data: The graph data object
        disease_entities: List of entity IDs that are diseases
        disease_name_to_idx: Mapping from disease names to indices in PRS array
        entity_dict_forward: Dictionary mapping entity IDs to names
        
    Returns:
        Dictionary mapping entity IDs to lists of (disease_idx, strength) pairs
    """
    relationships = {}
    
    # Get edge information
    edge_index = data.edge_index.t()
    
    # For each disease entity
    for disease_id in disease_entities:
        # Get disease name
        disease_name = entity_dict_forward.get(disease_id, "")
        
        # Skip if we can't map this disease to PRS data
        if disease_name not in disease_name_to_idx:
            continue
            
        disease_idx = disease_name_to_idx[disease_name]
        
        # Find all edges involving this disease
        for idx in range(edge_index.size(0)):
            h, t = edge_index[idx, 0].item(), edge_index[idx, 1].item()
            
            # If disease is the tail, add relationship to head entity
            if t == disease_id:
                if h not in relationships:
                    relationships[h] = []
                relationships[h].append((disease_idx, 1.0))
            
            # If disease is the head, add relationship to tail entity
            elif h == disease_id:
                if t not in relationships:
                    relationships[t] = []
                relationships[t].append((disease_idx, 1.0))
    
    return relationships

def extract_entity_protein_relationships(data, protein_entities, protein_name_to_idx, entity_dict_forward):
    """
    Extract relationships between entities and proteins.
    
    Args:
        data: The graph data object
        protein_entities: List of entity IDs that are proteins
        protein_name_to_idx: Mapping from protein names to indices in expression array
        entity_dict_forward: Dictionary mapping entity IDs to names
        
    Returns:
        Dictionary mapping entity IDs to lists of (protein_idx, strength) pairs
    """
    # Similar implementation to extract_entity_disease_relationships
    relationships = {}
    
    # Get edge information
    edge_index = data.edge_index.t()
    
    # For each protein entity
    for protein_id in protein_entities:
        # Get protein name
        protein_name = entity_dict_forward.get(protein_id, "")
        
        # Skip if we can't map this protein to expression data
        if protein_name not in protein_name_to_idx:
            continue
            
        protein_idx = protein_name_to_idx[protein_name]
        
        # Find all edges involving this protein
        for idx in range(edge_index.size(0)):
            h, t = edge_index[idx, 0].item(), edge_index[idx, 1].item()
            
            # If protein is the tail, add relationship to head entity
            if t == protein_id:
                if h not in relationships:
                    relationships[h] = []
                relationships[h].append((protein_idx, 1.0))
            
            # If protein is the head, add relationship to tail entity
            elif h == protein_id:
                if t not in relationships:
                    relationships[t] = []
                relationships[t].append((protein_idx, 1.0))
    
    return relationships

def get_entity_name(entity_id, patient_id, entity_dict_inverted=None):
    """
    Get the name of an entity from its ID.
    
    Args:
        entity_id: The ID of the entity to look up
        entity_dict_inverted: Optional dictionary mapping entity IDs to names.
                             If None, the function will attempt to load it.
    
    Returns:
        str: The name of the entity, or a placeholder if not found
    """
    # If dictionary not provided, try to load it
    if entity_dict_inverted is None:
        try:
            # Choose a default processed directory if none is specified
            processed_dir = f"/usr/homes/cxo147/git/ULTRA/kg-datasets/biomedical_KG_{patient_id}/processed/"
            _, _, entity_dict_inverted, _ = load_relation_dicts(processed_dir)
        except Exception as e:
            print(f"Error loading entity dictionary: {str(e)}")
            return f"Unknown Entity (ID: {entity_id})"
    
    # Look up the entity name in the dictionary
    entity_name = entity_dict_inverted.get(entity_id, f"Unknown Entity (ID: {entity_id})")
    ##print(f'matched: {entity_id} : {entity_name}')
    
    return entity_name

def get_entity_id(entity_name, patient_id, entity_dict_forward=None):
    """
    Get the ID of an entity from its name.
    
    Args:
        entity_name: The name of the entity to look up
        patient_id: The patient ID to determine which dataset to use
        entity_dict_forward: Optional dictionary mapping entity names to IDs.
                            If None, the function will attempt to load it.
    
    Returns:
        int: The ID of the entity, or None if not found
    """
    # If dictionary not provided, try to load it
    if entity_dict_forward is None:
        try:
            # Choose a default processed directory based on patient ID
            processed_dir = f"/usr/homes/cxo147/git/ULTRA/kg-datasets/biomedical_KG_{patient_id}/processed/"
            entity_dict_forward, _, _, _ = load_relation_dicts(processed_dir)
        except Exception as e:
            print(f"Error loading entity dictionary: {str(e)}")
            return None
    
    # Look up the entity ID in the dictionary
    entity_id = entity_dict_forward.get(entity_name, None)
    ##print(f'matched: {entity_name} : {entity_id}')
    
    return entity_id

def get_relation_id(relation_name, patient_id, relation_dict_forward=None):
    """
    Get the ID of a relation from its name.
    
    Args:
        relation_name: The name of the relation to look up
        patient_id: The patient ID to determine which dataset to use
        relation_dict_forward: Optional dictionary mapping relation names to IDs.
                              If None, the function will attempt to load it.
    
    Returns:
        int: The ID of the relation, or None if not found
    """
    # If dictionary not provided, try to load it
    if relation_dict_forward is None:
        try:
            # Choose a default processed directory based on patient ID
            processed_dir = f"/usr/homes/cxo147/git/ULTRA/kg-datasets/biomedical_KG_{patient_id}/processed/"
            _, relation_dict_forward, _, _ = load_relation_dicts(processed_dir)
        except Exception as e:
            print(f"Error loading relation dictionary: {str(e)}")
            return None
    
    # Look up the relation ID in the dictionary
    relation_id = relation_dict_forward.get(relation_name, None)
    ##print(f'matched relation: {relation_name} : {relation_id}')
    
    return relation_id