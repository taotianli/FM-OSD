# TCGR: Topology-Constrained Graph Refinement Module
# 
# This module enforces anatomical topology constraints via graph-based
# joint landmark refinement, going beyond independent correspondence matching.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


# ============================================================================
# Anatomical Graph Definition for Cephalometric Landmarks
# ============================================================================

def get_cephalometric_adjacency(num_landmarks: int = 19) -> torch.Tensor:
    """
    Define the anatomical adjacency matrix for cephalometric landmarks.
    
    The 19 cephalometric landmarks have well-defined anatomical relationships.
    We encode these as edges in a graph based on:
    - Anatomical proximity
    - Structural relationships (e.g., landmarks on the same bone structure)
    - Clinical relevance (commonly used landmark pairs in analysis)
    
    Returns:
        Adjacency matrix of shape [num_landmarks, num_landmarks]
    """
    # Initialize adjacency matrix
    adj = torch.zeros(num_landmarks, num_landmarks)
    
    # Define anatomical connections based on cephalometric anatomy
    # These connections are based on standard cephalometric analysis:
    # 
    # Landmark indices (0-18) typically correspond to:
    # 0: Sella (S)           1: Nasion (N)          2: Orbitale (Or)
    # 3: Porion (Po)         4: A point             5: B point
    # 6: Pogonion (Pog)      7: Menton (Me)         8: Gnathion (Gn)
    # 9: Gonion (Go)         10: ANS               11: PNS
    # 12: Upper Incisor Edge 13: Lower Incisor Edge 14: Upper Lip
    # 15: Lower Lip          16: Soft tissue Pog   17: Subnasale
    # 18: Columella
    
    # Cranial base connections
    edges = [
        (0, 1),   # S-N: cranial base
        (1, 2),   # N-Or: orbital region
        (2, 3),   # Or-Po: Frankfort horizontal reference
        (0, 3),   # S-Po: posterior cranial base
        
        # Maxilla connections
        (1, 4),   # N-A: maxillary position
        (4, 10),  # A-ANS: maxillary plane
        (10, 11), # ANS-PNS: palatal plane
        (4, 12),  # A-Upper incisor
        
        # Mandible connections
        (4, 5),   # A-B: maxillomandibular relationship
        (5, 6),   # B-Pog: chin prominence
        (6, 7),   # Pog-Me: chin point
        (7, 8),   # Me-Gn: mentum
        (8, 9),   # Gn-Go: mandibular plane
        (5, 13),  # B-Lower incisor
        (9, 3),   # Go-Po: posterior reference
        
        # Dental/incisor connections
        (12, 13), # Upper-Lower incisor: dental relationship
        
        # Soft tissue connections
        (10, 17), # ANS-Subnasale
        (17, 18), # Subnasale-Columella
        (17, 14), # Subnasale-Upper lip
        (14, 15), # Upper-Lower lip
        (15, 16), # Lower lip-Soft tissue Pog
        (6, 16),  # Pog-Soft tissue Pog
        
        # Additional structural connections
        (0, 4),   # S-A: SNA angle
        (0, 5),   # S-B: SNB angle
        (1, 8),   # N-Gn: facial height
    ]
    
    # Fill adjacency matrix (symmetric)
    for i, j in edges:
        if i < num_landmarks and j < num_landmarks:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    
    # Add self-loops
    adj = adj + torch.eye(num_landmarks)
    
    return adj


def get_dense_adjacency(num_landmarks: int) -> torch.Tensor:
    """Fully-connected graph (including self-loops) for generic anatomy (e.g. Hand, 37 pts)."""
    return torch.ones(num_landmarks, num_landmarks)


def get_edge_index(adj: torch.Tensor) -> torch.Tensor:
    """Convert adjacency matrix to edge_index format for PyG-style processing."""
    edge_index = adj.nonzero(as_tuple=False).t().contiguous()
    return edge_index


# ============================================================================
# Graph Neural Network Layers
# ============================================================================

class GraphConvLayer(nn.Module):
    """
    Simple Graph Convolution Layer.
    Implements: H' = sigma(D^{-1/2} A D^{-1/2} H W)
    
    We use a simplified version that doesn't require PyG,
    making the code self-contained.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [B, N, F_in] or [N, F_in]
            adj: Adjacency matrix [N, N]
        Returns:
            Updated features [B, N, F_out] or [N, F_out]
        """
        # Normalize adjacency matrix
        # D^{-1/2} A D^{-1/2}
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        adj_norm = adj * deg_inv_sqrt * deg_inv_sqrt.t()
        
        # Apply transformation
        if x.dim() == 2:
            # Single graph: [N, F_in]
            support = torch.mm(x, self.weight)  # [N, F_out]
            output = torch.mm(adj_norm, support)  # [N, F_out]
        else:
            # Batched: [B, N, F_in]
            support = torch.matmul(x, self.weight)  # [B, N, F_out]
            output = torch.matmul(adj_norm.unsqueeze(0), support)  # [B, N, F_out]
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (simplified GAT).
    Learns attention weights for aggregating neighbor information.
    """
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, 
                 dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        
        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.a = nn.Parameter(torch.FloatTensor(num_heads, 2 * out_features))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [B, N, F_in]
            adj: Adjacency matrix [N, N]
        Returns:
            Updated features [B, N, F_out * num_heads] if concat else [B, N, F_out]
        """
        B, N, _ = x.shape
        
        # Linear transformation
        h = self.W(x)  # [B, N, out_features * num_heads]
        h = h.view(B, N, self.num_heads, self.out_features)  # [B, N, H, F]
        
        # Compute attention scores
        # For each pair (i, j), compute attention
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1, -1)  # [B, N, N, H, F]
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1, -1)  # [B, N, N, H, F]
        
        # Concatenate and compute attention
        concat_features = torch.cat([h_i, h_j], dim=-1)  # [B, N, N, H, 2F]
        e = torch.einsum('bijnf,hf->bijh', concat_features, self.a)  # [B, N, N, H]
        e = self.leaky_relu(e)
        
        # Mask with adjacency (only attend to neighbors)
        mask = (adj == 0).unsqueeze(0).unsqueeze(-1)  # [1, N, N, 1]
        e = e.masked_fill(mask, float('-inf'))
        
        # Softmax attention
        attention = F.softmax(e, dim=2)  # [B, N, N, H]
        attention = self.dropout(attention)
        
        # Aggregate
        h = h.permute(0, 2, 1, 3)  # [B, H, N, F]
        attention = attention.permute(0, 3, 1, 2)  # [B, H, N, N]
        output = torch.matmul(attention, h)  # [B, H, N, F]
        output = output.permute(0, 2, 1, 3)  # [B, N, H, F]
        
        if self.concat:
            output = output.reshape(B, N, -1)  # [B, N, H * F]
        else:
            output = output.mean(dim=2)  # [B, N, F]
        
        return output


# ============================================================================
# TCGR: Topology-Constrained Graph Refinement Module
# ============================================================================

class TCGRModule(nn.Module):
    """
    Topology-Constrained Graph Refinement (TCGR) Module.
    
    Takes initial landmark predictions and refines them using graph neural
    networks that model anatomical topology constraints.
    
    Input features per landmark:
    - Initial predicted coordinates (2D)
    - Cosine similarity score (1D)
    - Local feature vector (optional, configurable dim)
    
    Output:
    - Refined coordinate offsets (2D)
    """
    def __init__(
        self,
        num_landmarks: int = 19,
        coord_dim: int = 2,
        score_dim: int = 1,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        use_attention: bool = True,
        num_heads: int = 4,
        dropout: float = 0.1,
        adjacency: str = "ceph",
    ):
        """
        Args:
            num_landmarks: Number of anatomical landmarks
            coord_dim: Dimension of coordinates (2 for 2D)
            score_dim: Dimension of similarity score
            feature_dim: Dimension of local feature vector (0 to disable)
            hidden_dim: Hidden dimension of GNN layers
            num_layers: Number of GNN layers
            use_attention: Whether to use GAT (True) or GCN (False)
            num_heads: Number of attention heads (if using GAT)
            dropout: Dropout rate
            adjacency: \"ceph\" (19-point cephalometric edges), \"dense\" (fully connected)
        """
        super().__init__()
        self.num_landmarks = num_landmarks
        self.coord_dim = coord_dim
        self.feature_dim = feature_dim
        
        # Input dimension: coords + score + optional features
        input_dim = coord_dim + score_dim + feature_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Graph neural network layers
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if use_attention:
                # Use graph attention
                layer = GraphAttentionLayer(
                    hidden_dim, hidden_dim // num_heads, 
                    num_heads=num_heads, dropout=dropout, concat=True
                )
            else:
                # Use simple graph convolution
                layer = GraphConvLayer(hidden_dim, hidden_dim)
            
            self.gnn_layers.append(layer)
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Output head: predict coordinate offsets
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, coord_dim)
        )
        
        # Register anatomical adjacency matrix as buffer
        if adjacency == "ceph":
            if num_landmarks != 19:
                raise ValueError(
                    "adjacency='ceph' requires num_landmarks=19; use adjacency='dense' for other counts."
                )
            adj = get_cephalometric_adjacency(num_landmarks)
        else:
            adj = get_dense_adjacency(num_landmarks)
        self.register_buffer('adj', adj)
        
        # Learnable refinement scale (small initial value for stable training)
        self.refinement_scale = nn.Parameter(torch.tensor(0.1))
        
    def forward(
        self, 
        coords: torch.Tensor, 
        scores: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Refine landmark coordinates using graph-based topology constraints.
        
        Args:
            coords: Initial coordinates [B, N, 2] normalized to [0, 1]
            scores: Similarity scores [B, N, 1] or [B, N]
            features: Optional local features [B, N, F]
            
        Returns:
            refined_coords: Refined coordinates [B, N, 2]
            offsets: Predicted offsets [B, N, 2]
        """
        B, N, _ = coords.shape
        
        # Ensure scores have correct shape
        if scores.dim() == 2:
            scores = scores.unsqueeze(-1)
        
        # Build input features
        if features is not None and self.feature_dim > 0:
            # Truncate or pad features to match expected dim
            if features.shape[-1] > self.feature_dim:
                features = features[..., :self.feature_dim]
            elif features.shape[-1] < self.feature_dim:
                padding = torch.zeros(B, N, self.feature_dim - features.shape[-1], 
                                      device=features.device)
                features = torch.cat([features, padding], dim=-1)
            
            node_input = torch.cat([coords, scores, features], dim=-1)
        else:
            node_input = torch.cat([coords, scores], dim=-1)
            # Pad with zeros if feature_dim > 0 but no features provided
            if self.feature_dim > 0:
                padding = torch.zeros(B, N, self.feature_dim, device=coords.device)
                node_input = torch.cat([node_input, padding], dim=-1)
        
        # Input projection
        h = self.input_proj(node_input)  # [B, N, hidden_dim]
        
        # Graph neural network layers with residual connections
        for gnn, ln in zip(self.gnn_layers, self.layer_norms):
            h_new = gnn(h, self.adj)
            h_new = F.relu(h_new)
            h = ln(h + h_new)  # Residual connection
        
        # Predict offsets
        offsets = self.output_head(h)  # [B, N, 2]
        offsets = offsets * self.refinement_scale  # Scale offsets
        
        # Refine coordinates
        refined_coords = coords + offsets
        
        return refined_coords, offsets
    
    def compute_topology_loss(
        self, 
        pred_coords: torch.Tensor, 
        gt_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute topology-aware loss that penalizes violations of
        expected inter-landmark distances.
        
        Args:
            pred_coords: Predicted coordinates [B, N, 2]
            gt_coords: Ground truth coordinates [B, N, 2]
            
        Returns:
            Topology consistency loss
        """
        # Compute pairwise distances
        pred_dist = torch.cdist(pred_coords, pred_coords)  # [B, N, N]
        gt_dist = torch.cdist(gt_coords, gt_coords)  # [B, N, N]
        
        # Only penalize distance errors for connected landmarks
        adj_mask = (self.adj > 0).float().unsqueeze(0)  # [1, N, N]
        
        # L1 distance error weighted by adjacency
        dist_error = torch.abs(pred_dist - gt_dist) * adj_mask
        
        # Normalize by number of edges
        num_edges = adj_mask.sum()
        topology_loss = dist_error.sum() / (num_edges * pred_coords.shape[0])
        
        return topology_loss


class TCGRLoss(nn.Module):
    """
    Combined loss function for TCGR training.
    
    Combines:
    - Coordinate MSE loss
    - Topology consistency loss
    """
    def __init__(self, coord_weight: float = 1.0, topo_weight: float = 0.1):
        super().__init__()
        self.coord_weight = coord_weight
        self.topo_weight = topo_weight
        self.mse = nn.MSELoss()
    
    def forward(
        self, 
        pred_coords: torch.Tensor, 
        gt_coords: torch.Tensor,
        tcgr_module: TCGRModule
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute TCGR training loss.
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        coord_loss = self.mse(pred_coords, gt_coords)
        topo_loss = tcgr_module.compute_topology_loss(pred_coords, gt_coords)
        
        total_loss = self.coord_weight * coord_loss + self.topo_weight * topo_loss
        
        loss_dict = {
            'coord_loss': coord_loss.item(),
            'topo_loss': topo_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict


# ============================================================================
# Utility Functions
# ============================================================================

def normalize_coordinates(coords: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    """Normalize coordinates to [0, 1] range."""
    h, w = image_size
    normalized = coords.clone()
    normalized[..., 0] = coords[..., 0] / h
    normalized[..., 1] = coords[..., 1] / w
    return normalized


def denormalize_coordinates(coords: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    """Denormalize coordinates from [0, 1] to pixel space."""
    h, w = image_size
    denormalized = coords.clone()
    denormalized[..., 0] = coords[..., 0] * h
    denormalized[..., 1] = coords[..., 1] * w
    return denormalized


if __name__ == "__main__":
    # Test the TCGR module
    print("Testing TCGR Module...")
    
    # Create module
    tcgr = TCGRModule(
        num_landmarks=19,
        coord_dim=2,
        score_dim=1,
        feature_dim=64,
        hidden_dim=128,
        num_layers=2,
        use_attention=True
    )
    
    # Test forward pass
    batch_size = 4
    coords = torch.rand(batch_size, 19, 2)  # Random initial coords
    scores = torch.rand(batch_size, 19)     # Random similarity scores
    features = torch.rand(batch_size, 19, 64)  # Random local features
    
    refined_coords, offsets = tcgr(coords, scores, features)
    print(f"Input coords shape: {coords.shape}")
    print(f"Refined coords shape: {refined_coords.shape}")
    print(f"Offsets shape: {offsets.shape}")
    print(f"Mean offset magnitude: {offsets.norm(dim=-1).mean():.4f}")
    
    # Test topology loss
    gt_coords = torch.rand(batch_size, 19, 2)
    topo_loss = tcgr.compute_topology_loss(refined_coords, gt_coords)
    print(f"Topology loss: {topo_loss.item():.4f}")
    
    # Test combined loss
    loss_fn = TCGRLoss()
    total_loss, loss_dict = loss_fn(refined_coords, gt_coords, tcgr)
    print(f"Loss components: {loss_dict}")
    
    print("\nTCGR Module test passed!")
