from functools import reduce
from torch_scatter import scatter_add
from torch_geometric.data import Data
import torch
import torch.nn.functional as F

def edge_match(edge_index, query_index):
    # O((n + q)logn) time
    # O(n) memory
    # edge_index: big underlying graph
    # query_index: edges to match

    # preparing unique hashing of edges, base: (max_node, max_relation) + 1
    base = edge_index.max(dim=1)[0] + 1
    # we will map edges to long ints, so we need to make sure the maximum product is less than MAX_LONG_INT
    # idea: max number of edges = num_nodes * num_relations
    # e.g. for a graph of 10 nodes / 5 relations, edge IDs 0...9 mean all possible outgoing edge types from node 0
    # given a tuple (h, r), we will search for all other existing edges starting from head h
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max 
    scale = base.cumprod(0)
    scale = scale[-1] // scale

    # hash both the original edge index and the query index to unique integers
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)

    # matched ranges: [start[i], end[i])
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    # num_match shows how many edges satisfy the (h, r) pattern for each query in the batch
    num_match = end - start

    # generate the corresponding ranges
    offset = num_match.cumsum(0) - num_match
    
    ## wrote this part 
    #if num_match.sum() == 0:
    #    # Handle the case with no matches, possibly by skipping certain operations
    #    # or setting variables to None or an empty tensor, depending on later usage
    #    print('(!!!) num_match.sum() == 0', edge_index.device)
    

    range = torch.arange(num_match.sum(), device=edge_index.device)
    range = range + (start - offset).repeat_interleave(num_match)

    return order[range], num_match

def negative_sampling(data, batch, num_negative, strict=True):
    """
    data:         PyG Data object or similar, with 'num_nodes' and other info.
    batch:        shape [B, 1, 3], where each row has exactly one positive triple at index 0.
    num_negative: number of negative triples per positive
    strict:       whether to do strict negative sampling or random negative sampling
    
    Returns:
        shape [B, 1 + num_negative, 3], 
        storing the original positive triple in column #0, plus negative triples in columns [1..num_negative].
    """

    # B = batch.size(0), i.e. the number of positive triples
    batch_size = batch.size(0)

    # 1) Extract the positive triple from dimension #1=0.
    #    Now we do 3D indexing: [B, 0, 2] => 2 is the last dimension
    pos_h_index = batch[:, 0, 0]  # shape [B]
    pos_t_index = batch[:, 0, 1]  # shape [B]
    pos_r_index = batch[:, 0, 2]  # shape [B]

    # 2) Based on 'strict' or not, gather negative tail/head indices
    if strict:
        # This function likely needs to handle a 3D batch or be given a 2D flatten.
        # Example: flatten it for mask:
        # flatten_batch = batch[:,0,:]  # shape [B, 3]
        # t_mask, h_mask = strict_negative_mask(data, flatten_batch)

        # For demonstration, let's say we do something like:
        flatten_batch = torch.stack([pos_h_index, pos_t_index, pos_r_index], dim=-1)  # shape [B, 3]
        t_mask, h_mask = strict_negative_mask(data, flatten_batch)

        t_mask = t_mask[:batch_size // 2]
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)

        rand = torch.rand(len(t_mask), num_negative, device=batch.device)
        index = (rand * num_t_candidate.unsqueeze(-1)).long()
        index = index + (num_t_candidate.cumsum(0) - num_t_candidate).unsqueeze(-1)
        neg_t_index = neg_t_candidate[index]

        h_mask = h_mask[batch_size // 2:]
        neg_h_candidate = h_mask.nonzero()[:, 1]
        num_h_candidate = h_mask.sum(dim=-1)

        rand = torch.rand(len(h_mask), num_negative, device=batch.device)
        index = (rand * num_h_candidate.unsqueeze(-1)).long()
        index = index + (num_h_candidate.cumsum(0) - num_h_candidate).unsqueeze(-1)
        neg_h_index = neg_h_candidate[index]
    else:
        # random negative sampling
        # shape [B, num_negative] of random nodes
        neg_index = torch.randint(data.num_nodes, (batch_size, num_negative), device=batch.device)
        # first half -> tail corruption, second half -> head corruption
        neg_t_index = neg_index[:batch_size // 2]
        neg_h_index = neg_index[batch_size // 2:]

    # 3) Expand the positive triple fields to shape [B, 1 + num_negative]
    #    The first column is the positive, the rest are negatives
    h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)  # [B, 1+num_neg]
    t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
    r_index = pos_r_index.unsqueeze(-1).repeat(1, num_negative + 1)

    # The first half of the batch does tail corruption => put the negative tails in [1..num_neg]
    t_index[:batch_size // 2, 1:] = neg_t_index
    # The second half of the batch does head corruption => put the negative heads in [1..num_neg]
    h_index[batch_size // 2:, 1:] = neg_h_index

    # 4) Stack into shape [B, 1 + num_negative, 3]
    #    dimension = -1 means the last dimension -> we get [B, 1+num_neg, 3]
    out = torch.stack([h_index, t_index, r_index], dim=-1)

    # Now out[:, 0, :] is the positive triple
    # out[:, 1:, :] are the negative triples
    return out


'''
def negative_sampling(data, batch, num_negative, strict=True):
    batch_size = len(batch)
    #pos_h_index, pos_t_index, pos_r_index = batch.t()

    B = batch.size(0)
    # batch[:, 0, :] => the single positive triple for each row
    # For example, you might do:
    pos_h_index = batch[:, 0, 0]  # shape [B]
    pos_t_index = batch[:, 0, 1]  # shape [B]
    pos_r_index = batch[:, 0, 2]  # shape [B]

    # strict negative sampling vs random negative sampling
    if strict:
        t_mask, h_mask = strict_negative_mask(data, batch)
        t_mask = t_mask[:batch_size // 2]
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)
        # draw samples for negative tails
        rand = torch.rand(len(t_mask), num_negative, device=batch.device)
        index = (rand * num_t_candidate.unsqueeze(-1)).long()
        index = index + (num_t_candidate.cumsum(0) - num_t_candidate).unsqueeze(-1)
        neg_t_index = neg_t_candidate[index]

        h_mask = h_mask[batch_size // 2:]
        neg_h_candidate = h_mask.nonzero()[:, 1]
        num_h_candidate = h_mask.sum(dim=-1)
        # draw samples for negative heads
        rand = torch.rand(len(h_mask), num_negative, device=batch.device)
        index = (rand * num_h_candidate.unsqueeze(-1)).long()
        index = index + (num_h_candidate.cumsum(0) - num_h_candidate).unsqueeze(-1)
        neg_h_index = neg_h_candidate[index]
    else:
        neg_index = torch.randint(data.num_nodes, (batch_size, num_negative), device=batch.device)
        neg_t_index, neg_h_index = neg_index[:batch_size // 2], neg_index[batch_size // 2:]

    h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
    r_index = pos_r_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index[:batch_size // 2, 1:] = neg_t_index
    h_index[batch_size // 2:, 1:] = neg_h_index

    return torch.stack([h_index, t_index, r_index], dim=-1)
'''

def all_negative(data, batch, no_of_triples=None):
    if no_of_triples is not None:
        length = no_of_triples
    else:
        length = data.num_nodes

    pos_h_index, pos_t_index, pos_r_index = batch.t()
    r_index = pos_r_index.unsqueeze(-1).expand(-1, length)

    # generate all negative tails for this batch
    all_index = torch.arange(length, device=batch.device)
    h_index, t_index = torch.meshgrid(pos_h_index, all_index, indexing="ij")  # indexing "xy" would return transposed
    t_batch = torch.stack([h_index, t_index, r_index], dim=-1)
    
    # generate all negative heads for this batch
    all_index = torch.arange(length, device=batch.device)
    t_index, h_index = torch.meshgrid(pos_t_index, all_index, indexing="ij")
    h_batch = torch.stack([h_index, t_index, r_index], dim=-1)

    return t_batch, h_batch

'''
def strict_negative_mask(data, batch):
    # this function makes sure that for a given (h, r) batch we will NOT sample true tails as random negatives
    # similarly, for a given (t, r) we will NOT sample existing true heads as random negatives

    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # part I: sample hard negative tails
    # edge index of all (head, relation) edges from the underlying graph
    edge_index = torch.stack([data.edge_index[0], data.edge_type])
    # edge index of current batch (head, relation) for which we will sample negatives
    query_index = torch.stack([pos_h_index, pos_r_index])
    # search for all true tails for the given (h, r) batch
    edge_id, num_t_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    t_truth_index = data.edge_index[1, edge_id]
    sample_id = torch.arange(len(num_t_truth), device=batch.device).repeat_interleave(num_t_truth)
    t_mask = torch.ones(len(num_t_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true tails
    t_mask[sample_id, t_truth_index] = 0
    t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)

    # part II: sample hard negative heads
    # edge_index[1] denotes tails, so the edge index becomes (t, r)
    edge_index = torch.stack([data.edge_index[1], data.edge_type])
    # edge index of current batch (tail, relation) for which we will sample heads
    query_index = torch.stack([pos_t_index, pos_r_index])
    # search for all true heads for the given (t, r) batch
    edge_id, num_h_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    h_truth_index = data.edge_index[0, edge_id]
    sample_id = torch.arange(len(num_h_truth), device=batch.device).repeat_interleave(num_h_truth)
    h_mask = torch.ones(len(num_h_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true heads
    h_mask[sample_id, h_truth_index] = 0
    h_mask.scatter_(1, pos_h_index.unsqueeze(-1), 0)

    return t_mask, h_mask
'''

def strict_negative_mask(data, batch, test_data=None):
    """
    Modified version of strict_negative_mask that handles different entity counts
    between test and filtered data.
    """
    # Use test_data's entity count if provided, otherwise use data's entity count
    num_nodes = test_data.num_nodes if test_data is not None else data.num_nodes
    
    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # Part I: sample hard negative tails
    edge_index = torch.stack([data.edge_index[0], data.edge_type])
    query_index = torch.stack([pos_h_index, pos_r_index])
    edge_id, num_t_truth = edge_match(edge_index, query_index)
    t_truth_index = data.edge_index[1, edge_id]
    
    # Generate sample_id first
    sample_id = torch.arange(len(num_t_truth), device=batch.device).repeat_interleave(num_t_truth)
    
    # Filter out any entity indices that are beyond the test data's entity count
    valid_t_mask = t_truth_index < num_nodes
    t_truth_index = t_truth_index[valid_t_mask]
    
    # Update num_t_truth to account for filtered indices
    if valid_t_mask.shape[0] > 0:  # only if we have any truth indices
        new_num_t_truth = torch.zeros_like(num_t_truth)
        unique_samples = torch.unique(sample_id[valid_t_mask])
        for i in unique_samples:
            new_num_t_truth[i] = (sample_id[valid_t_mask] == i).sum()
        num_t_truth = new_num_t_truth
    
    # Create the tail mask
    t_mask = torch.ones(len(num_t_truth), num_nodes, dtype=torch.bool, device=batch.device)
    
    if t_truth_index.numel() > 0:  # only if we have valid truth indices
        filtered_sample_id = sample_id[valid_t_mask]
        t_mask[filtered_sample_id, t_truth_index] = 0
    
    # Only mask pos_t_index that are within bounds
    valid_pos_t = pos_t_index < num_nodes
    if valid_pos_t.any():
        t_mask.scatter_(1, pos_t_index[valid_pos_t].unsqueeze(-1), 0)

    # Part II: sample hard negative heads
    edge_index = torch.stack([data.edge_index[1], data.edge_type])
    query_index = torch.stack([pos_t_index, pos_r_index])
    edge_id, num_h_truth = edge_match(edge_index, query_index)
    h_truth_index = data.edge_index[0, edge_id]
    
    # Generate sample_id for heads
    sample_id = torch.arange(len(num_h_truth), device=batch.device).repeat_interleave(num_h_truth)
    
    # Filter out any entity indices that are beyond the test data's entity count
    valid_h_mask = h_truth_index < num_nodes
    h_truth_index = h_truth_index[valid_h_mask]
    
    # Update num_h_truth to account for filtered indices
    if valid_h_mask.shape[0] > 0:  # only if we have any truth indices
        new_num_h_truth = torch.zeros_like(num_h_truth)
        unique_samples = torch.unique(sample_id[valid_h_mask])
        for i in unique_samples:
            new_num_h_truth[i] = (sample_id[valid_h_mask] == i).sum()
        num_h_truth = new_num_h_truth
    
    # Create the head mask
    h_mask = torch.ones(len(num_h_truth), num_nodes, dtype=torch.bool, device=batch.device)
    
    if h_truth_index.numel() > 0:  # only if we have valid truth indices
        filtered_sample_id = sample_id[valid_h_mask]
        h_mask[filtered_sample_id, h_truth_index] = 0
    
    # Only mask pos_h_index that are within bounds
    valid_pos_h = pos_h_index < num_nodes
    if valid_pos_h.any():
        h_mask.scatter_(1, pos_h_index[valid_pos_h].unsqueeze(-1), 0)

    return t_mask, h_mask

def compute_ranking(pred, target, mask=None):
    #print("target shape:", target.shape)
    #print("mask shape:", mask.shape)
   
    if mask is not None:
        #print("mask shape:", mask.shape)

        # Ensure mask matches pred's shape by clipping the mask
        if mask.size(1) > pred.size(1):
            mask_adjusted = mask[:, :pred.size(1)]
        else:
            mask_adjusted = mask
    else:
        mask_adjusted = None

    pos_pred = pred.gather(-1, target.unsqueeze(-1))

    if mask_adjusted is not None:
        # Calculate padding needed on the second dimension if any (should not be needed after adjustment)
        padding_needed = pred.size(1) - mask_adjusted.size(1)
        if padding_needed > 0:
            # This block should not be necessary after the initial adjustment, but kept for safety
            mask_adjusted = F.pad(mask_adjusted, (0, padding_needed), "constant", 0)
        
        # Filtered ranking
        ranking = torch.sum((pos_pred <= pred) & mask_adjusted, dim=-1) + 1
    else:
        # Unfiltered ranking
        ranking = torch.sum(pos_pred <= pred, dim=-1) + 1

    # Initialize a list to hold the most confident scores for triples with ranking 1
    most_confident_scores = []

    # Check for rankings that are 1 and append the corresponding confident scores
    for i, rank in enumerate(ranking):
        #if rank == 1:
        most_confident_score = pos_pred[i].item()
        #print(f"Most confident score for triple with ranking 1 at index {i}: {most_confident_score}")
        most_confident_scores.append(most_confident_score)

    # Return both the ranking and the list of most confident scores for ranking 1
    #print('in compute', ranking.shape, len(most_confident_scores))
    #print(most_confident_scores)
    return ranking, most_confident_scores


def build_relation_graph(graph):

    # expect the graph is already with inverse edges

    edge_index, edge_type = graph.edge_index, graph.edge_type
    num_nodes, num_rels = graph.num_nodes, graph.num_relations
    device = edge_index.device

    Eh = torch.vstack([edge_index[0], edge_type]).T.unique(dim=0)  # (num_edges, 2)
    Dh = scatter_add(torch.ones_like(Eh[:, 1]), Eh[:, 0])

    EhT = torch.sparse_coo_tensor(
        torch.flip(Eh, dims=[1]).T, 
        torch.ones(Eh.shape[0], device=device) / Dh[Eh[:, 0]], 
        (num_rels, num_nodes)
    )
    Eh = torch.sparse_coo_tensor(
        Eh.T, 
        torch.ones(Eh.shape[0], device=device), 
        (num_nodes, num_rels)
    )
    Et = torch.vstack([edge_index[1], edge_type]).T.unique(dim=0)  # (num_edges, 2)

    Dt = scatter_add(torch.ones_like(Et[:, 1]), Et[:, 0])
    assert not (Dt[Et[:, 0]] == 0).any()

    EtT = torch.sparse_coo_tensor(
        torch.flip(Et, dims=[1]).T, 
        torch.ones(Et.shape[0], device=device) / Dt[Et[:, 0]], 
        (num_rels, num_nodes)
    )
    Et = torch.sparse_coo_tensor(
        Et.T, 
        torch.ones(Et.shape[0], device=device), 
        (num_nodes, num_rels)
    )

    Ahh = torch.sparse.mm(EhT, Eh).coalesce()
    Att = torch.sparse.mm(EtT, Et).coalesce()
    Aht = torch.sparse.mm(EhT, Et).coalesce()
    Ath = torch.sparse.mm(EtT, Eh).coalesce()

    hh_edges = torch.cat([Ahh.indices().T, torch.zeros(Ahh.indices().T.shape[0], 1, dtype=torch.long).fill_(0)], dim=1)  # head to head
    tt_edges = torch.cat([Att.indices().T, torch.zeros(Att.indices().T.shape[0], 1, dtype=torch.long).fill_(1)], dim=1)  # tail to tail
    ht_edges = torch.cat([Aht.indices().T, torch.zeros(Aht.indices().T.shape[0], 1, dtype=torch.long).fill_(2)], dim=1)  # head to tail
    th_edges = torch.cat([Ath.indices().T, torch.zeros(Ath.indices().T.shape[0], 1, dtype=torch.long).fill_(3)], dim=1)  # tail to head
    
    rel_graph = Data(
        edge_index=torch.cat([hh_edges[:, [0, 1]].T, tt_edges[:, [0, 1]].T, ht_edges[:, [0, 1]].T, th_edges[:, [0, 1]].T], dim=1), 
        edge_type=torch.cat([hh_edges[:, 2], tt_edges[:, 2], ht_edges[:, 2], th_edges[:, 2]], dim=0),
        num_nodes=num_rels, 
        num_relations=4
    )

    graph.relation_graph = rel_graph
    return graph


