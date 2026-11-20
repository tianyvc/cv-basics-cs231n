import torch
import numpy as np


def sim(z_i, z_j):
    """Normalized dot product between two vectors.

    Inputs:
    - z_i: 1xD tensor.
    - z_j: 1xD tensor.
    
    Returns:
    - A scalar value that is the normalized dot product between z_i and z_j.
    """
    norm_dot_product = None
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################
    z_i_norm = torch.linalg.norm(z_i)
    z_j_norm = torch.linalg.norm(z_j)
    norm_dot_product = torch.dot(z_i, z_j) / (z_i_norm * z_j_norm)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """Compute the contrastive loss L over a batch (naive loop version).
    
    Input:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch. The same row in out_left and out_right form a positive pair. 
    In other words, (out_left[k], out_right[k]) form a positive pair for all k=0...N-1.
    - tau: scalar value, temperature parameter that determines how fast the exponential increases.
    
    Returns:
    - A scalar value; the total loss across all positive pairs in the batch. See notebook for definition.
    """
    N = out_left.shape[0]  # total number of training examples
    
     # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    total_loss = 0
    for k in range(N):  # loop through each positive pair (k, k+N)
        z_k, z_k_N = out[k], out[k+N]
        
        ##############################################################################
        # TODO: Start of your code.                                                  #
        #                                                                            #
        # Hint: Compute l(k, k+N) and l(k+N, k).                                     #
        ##############################################################################
        denoml = 0.0
        for i in range(2*N):
            if i != k:
                denoml += torch.exp(sim(z_k, out[i]) / tau)
        loss1 = - torch.log(torch.exp(sim(z_k, z_k_N) / tau) / denoml)

        exp_pos2 = torch.exp(sim(z_k_N, z_k) / tau)
        exp_neg2 = 0.0
        for i in range(2*N):
            if i != k+N:
                exp_neg2 += torch.exp(sim(z_k_N, out[i]) / tau)
        loss2 = - torch.log(exp_pos2 / exp_neg2)
        total_loss += loss1 + loss2
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
    
    # In the end, we need to divide the total loss by 2N, the number of samples in the batch.
    total_loss = total_loss / (2*N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """Normalized dot product between positive pairs.

    Inputs:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch.
    The same row in out_left and out_right form a positive pair.
    
    Returns:
    - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].
    """
    pos_pairs = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################
    # normalize each vector along feature dim
    left_norm = torch.linalg.norm(out_left, dim=1, keepdim=True)   # (N,1)
    right_norm = torch.linalg.norm(out_right, dim=1, keepdim=True) # (N,1)
    # element-wise dot product for each row
    dot = torch.sum(out_left * out_right, dim=1, keepdim=True)     # (N,1)
    pos_pairs = dot / (left_norm * right_norm)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return pos_pairs


def compute_sim_matrix(out):
    """Compute a 2N x 2N matrix of normalized dot products between all pairs of augmented examples in a batch.

    Inputs:
    - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single augmented example.
    There are a total of 2N augmented examples in the batch.
    
    Returns:
    - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot product between out[i] and out[j].
    """
    sim_matrix = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    ##############################################################################
    # normalize each row
    out_norm = torch.linalg.norm(out, dim=1, keepdim=True)  # (2N,1)
    out_normalized = out / out_norm
    # normalized cosine similarity = matmul of normalized vectors
    sim_matrix = out_normalized @ out_normalized.T
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device=None):
    """Compute the contrastive loss L over a batch (vectorized version). No loops are allowed.
    
    Inputs and output are the same as in simclr_loss_naive.
    """
    device = out_left.device if device is None else device
    N = out_left.shape[0]
    
    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # Compute similarity matrix between all pairs of augmented examples in the batch.
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    
    ##############################################################################
    # TODO: Start of your code. Follow the hints.                                #
    ##############################################################################
    
    # Step 1: Use sim_matrix to compute the denominator value for all augmented samples.
    # Hint: Compute e^{sim / tau} and store into exponential, which should have shape 2N x 2N.
    exponential = torch.exp(sim_matrix / tau)  # [2*N, 2*N]
    
    # This binary mask zeros out terms where k=i.
    mask = (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).to(device).bool()
    
    # We apply the binary mask.
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]
    
    # Hint: Compute the denominator values for all augmented samples. This should be a 2N x 1 vector.
    denom = torch.sum(exponential, dim=1, keepdim=True)  # [2*N, 1]

    # Step 2: Compute similarity between positive pairs.
    # You can do this in two ways: 
    # Option 1: Extract the corresponding indices from sim_matrix. 
    # Option 2: Use sim_positive_pairs().
    pos_sim = torch.zeros(2*N, 1, device=device)
    pos_sim[:N, 0] = sim_matrix[range(N), range(N, 2*N)]
    pos_sim[N:, 0] = sim_matrix[range(N, 2*N), range(N)]
    
    # Step 3: Compute the numerator value for all augmented samples.
    numerator = torch.exp(pos_sim / tau)  # [2*N, 1]
    
    
    # Step 4: Now that you have the numerator and denominator for all augmented samples, compute the total loss.
    loss = -torch.sum(torch.log(numerator / denom)) / (2 * N)
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return loss


def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))