# =============================================================================
# Observation Encoder
# =============================================================================
"""
Neural network encoders for MiniGrid observations.

This module provides:
- GridEncoder: CNN for grid observations
- SubgoalEncoder: Embedding for subgoals
- ObservationEncoder: Combined encoder

Why Separate Encoders?
----------------------
Modularity! Each encoder can be:
- Tested independently
- Replaced with different architectures
- Pre-trained separately

MiniGrid Observation Structure:
-------------------------------
The default observation is a 7x7x3 tensor:
- 7x7: Agent's field of view (what it "sees")
- Channel 0: Object type (0=empty, 1=wall, 2=floor, 3=door, 4=key, ...)
- Channel 1: Color (0=red, 1=green, 2=blue, 3=purple, 4=yellow, 5=grey)
- Channel 2: State (0=open, 1=closed, 2=locked for doors)

This is a PARTIAL observation - the agent can't see behind itself!
This is the difference between MiniGrid and problems with full observability. 
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    
    class GridEncoder(nn.Module):
        """
        CNN encoder for MiniGrid observations.
        
        Architecture:
        -------------
        Conv2D layers extract spatial features from the grid.
        
        Why CNN?
        - Grid is 2D spatial data (like images)
        - CNN captures local patterns (walls, objects, corridors)
        - Translation equivariance: same object looks same anywhere
        
        Why NOT ViT/Transformer?
        - Overkill for 7x7 grid
        - CNN is faster and simpler
        - Works great for this scale
        """
        
        def __init__(
            self,
            in_channels: int = 3,
            cnn_channels: List[int] = [32, 64, 64],
            output_dim: int = 128,
        ):
            """
            Initialize the grid encoder.
            
            Parameters:
            -----------
            in_channels : int
                Number of input channels (3 for MiniGrid)
            cnn_channels : list
                Channels for each conv layer
            output_dim : int
                Output embedding dimension
            """
            super().__init__()
            
            # Build CNN layers
            layers = []
            prev_channels = in_channels
            
            for i, out_ch in enumerate(cnn_channels):
                layers.append(nn.Conv2d(
                    prev_channels, out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ))
                layers.append(nn.ReLU())
                # Only pool if we have enough spatial dimensions
                if i < len(cnn_channels) - 1:
                    layers.append(nn.MaxPool2d(2))
                prev_channels = out_ch
            
            self.cnn = nn.Sequential(*layers)
            
            # Calculate flattened size after CNN
            # For 7x7 input with 2 pooling layers: 7 -> 3 -> 1
            # Actually depends on padding, let's compute dynamically
            self._cnn_output_size = self._get_cnn_output_size((7, 7), in_channels)
            
            # Final projection
            self.fc = nn.Linear(self._cnn_output_size, output_dim)
        
        def _get_cnn_output_size(self, input_shape: Tuple[int, int], in_channels: int) -> int:
            """Calculate CNN output size dynamically."""
            with torch.no_grad():
                dummy = torch.zeros(1, in_channels, *input_shape)
                output = self.cnn(dummy)
                return int(np.prod(output.shape[1:]))
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Encode grid observation.
            
            Parameters:
            -----------
            x : torch.Tensor
                Grid observation of shape (batch, height, width, channels)
                Note: MiniGrid uses HWC format, we convert to CHW
                
            Returns:
            --------
            torch.Tensor
                Encoded features of shape (batch, output_dim)
            """
            # Convert HWC to CHW if needed
            if x.dim() == 4 and x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)
            elif x.dim() == 3 and x.shape[-1] == 3:
                x = x.permute(2, 0, 1).unsqueeze(0)
            
            # Normalize to [0, 1] if int
            if x.dtype in [torch.int32, torch.int64, torch.uint8]:
                x = x.float() / 10.0  # MiniGrid values are small ints
            
            # CNN forward
            features = self.cnn(x)
            features = features.view(features.size(0), -1)
            
            # Final projection
            return self.fc(features)
    
    
    class SubgoalEncoder(nn.Module):
        """
        Encoder for subgoal conditioning.
        
        This encodes the current subgoal (from the LLM planner) into
        a vector that conditions the policy.
        
        Options for encoding:
        1. ONE-HOT: Simple, but limited vocabulary
        2. EMBEDDING: Learned, handles more subgoals
        3. LANGUAGE MODEL: Most flexible, but slower
        
        We use option 2 (embedding) for balance.
        """
        
        # Vocabulary for subgoal encoding
        ACTIONS = ["navigate_to", "pickup", "drop", "toggle"]
        COLORS = ["red", "green", "blue", "purple", "yellow", "grey"]
        OBJECTS = ["ball", "key", "box", "door", "goal"]
        
        def __init__(
            self,
            embedding_dim: int = 32,
        ):
            """
            Initialize subgoal encoder.
            
            Parameters:
            -----------
            embedding_dim : int
                Dimension of subgoal embedding
            """
            super().__init__()
            
            # Create vocabularies
            self.action_to_idx = {a: i for i, a in enumerate(self.ACTIONS)}
            self.color_to_idx = {c: i for i, c in enumerate(self.COLORS)}
            self.object_to_idx = {o: i for i, o in enumerate(self.OBJECTS)}
            
            # Embedding layers
            self.action_embed = nn.Embedding(len(self.ACTIONS), embedding_dim // 2)
            self.color_embed = nn.Embedding(len(self.COLORS) + 1, embedding_dim // 4)  # +1 for unknown
            self.object_embed = nn.Embedding(len(self.OBJECTS) + 1, embedding_dim // 4)  # +1 for unknown
            
            self.output_dim = embedding_dim
        
        def encode_subgoal(self, subgoal_str: str) -> Tuple[int, int, int]:
            """
            Convert subgoal string to indices.
            
            Example: "navigate_to(red ball)" -> (0, 0, 0)
            """
            # Parse action
            action_idx = 0
            for action, idx in self.action_to_idx.items():
                if action in subgoal_str:
                    action_idx = idx
                    break
            
            # Parse color
            color_idx = len(self.COLORS)  # Unknown
            for color, idx in self.color_to_idx.items():
                if color in subgoal_str:
                    color_idx = idx
                    break
            
            # Parse object
            object_idx = len(self.OBJECTS)  # Unknown
            for obj, idx in self.object_to_idx.items():
                if obj in subgoal_str:
                    object_idx = idx
                    break
            
            return action_idx, color_idx, object_idx
        
        def forward(
            self,
            action_idx: torch.Tensor,
            color_idx: torch.Tensor,
            object_idx: torch.Tensor,
        ) -> torch.Tensor:
            """
            Encode subgoal components.
            
            Parameters:
            -----------
            action_idx : torch.Tensor
                Action indices (batch,)
            color_idx : torch.Tensor
                Color indices (batch,)
            object_idx : torch.Tensor
                Object indices (batch,)
                
            Returns:
            --------
            torch.Tensor
                Subgoal embedding (batch, embedding_dim)
            """
            action_emb = self.action_embed(action_idx)
            color_emb = self.color_embed(color_idx)
            object_emb = self.object_embed(object_idx)
            
            return torch.cat([action_emb, color_emb, object_emb], dim=-1)
    
    
    class ObservationEncoder(nn.Module):
        """
        Combined encoder for full observations.
        
        Encodes:
        - Grid image (CNN)
        - Agent direction (one-hot)
        - Current subgoal (embedding)
        
        All encodings are concatenated and projected to final representation.
        """
        
        def __init__(
            self,
            grid_output_dim: int = 128,
            subgoal_dim: int = 32,
            direction_dim: int = 4,
            hidden_dim: int = 256,
            output_dim: int = 256,
        ):
            """
            Initialize combined encoder.
            
            Parameters:
            -----------
            grid_output_dim : int
                Output dimension of grid CNN
            subgoal_dim : int
                Dimension of subgoal embedding
            direction_dim : int
                Number of directions (4)
            hidden_dim : int
                Hidden layer dimension
            output_dim : int
                Final output dimension
            """
            super().__init__()
            
            self.grid_encoder = GridEncoder(output_dim=grid_output_dim)
            self.subgoal_encoder = SubgoalEncoder(embedding_dim=subgoal_dim)
            
            # Direction is one-hot encoded
            self.direction_dim = direction_dim
            
            # Combined dimension
            combined_dim = grid_output_dim + subgoal_dim + direction_dim
            
            # MLP to combine features
            self.mlp = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.ReLU(),
            )
            
            self.output_dim = output_dim
        
        def forward(
            self,
            grid: torch.Tensor,
            direction: torch.Tensor,
            subgoal_action: torch.Tensor,
            subgoal_color: torch.Tensor,
            subgoal_object: torch.Tensor,
        ) -> torch.Tensor:
            """
            Encode full observation.
            
            Parameters:
            -----------
            grid : torch.Tensor
                Grid observation (batch, 7, 7, 3)
            direction : torch.Tensor
                Agent direction (batch,) as int
            subgoal_action : torch.Tensor
                Subgoal action index (batch,)
            subgoal_color : torch.Tensor
                Subgoal color index (batch,)
            subgoal_object : torch.Tensor
                Subgoal object index (batch,)
                
            Returns:
            --------
            torch.Tensor
                Encoded observation (batch, output_dim)
            """
            # Encode grid
            grid_features = self.grid_encoder(grid)
            
            # Encode direction as one-hot
            direction_onehot = F.one_hot(
                direction.long(),
                num_classes=self.direction_dim
            ).float()
            
            # Encode subgoal
            subgoal_features = self.subgoal_encoder(
                subgoal_action, subgoal_color, subgoal_object
            )
            
            # Concatenate all features
            combined = torch.cat([
                grid_features,
                direction_onehot,
                subgoal_features,
            ], dim=-1)
            
            # MLP
            return self.mlp(combined)


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping test.")
    else:
        print("Testing observation encoders...")
        print()
        
        # Test GridEncoder
        print("=== GridEncoder ===")
        grid_encoder = GridEncoder()
        dummy_grid = torch.randn(2, 7, 7, 3)  # batch=2, HWC format
        grid_out = grid_encoder(dummy_grid)
        print(f"Input shape: {dummy_grid.shape}")
        print(f"Output shape: {grid_out.shape}")
        print()
        
        # Test SubgoalEncoder
        print("=== SubgoalEncoder ===")
        subgoal_encoder = SubgoalEncoder()
        indices = subgoal_encoder.encode_subgoal("navigate_to(red ball)")
        print(f"Subgoal: navigate_to(red ball)")
        print(f"Indices: action={indices[0]}, color={indices[1]}, object={indices[2]}")
        
        action_idx = torch.tensor([indices[0], indices[0]])
        color_idx = torch.tensor([indices[1], indices[1]])
        object_idx = torch.tensor([indices[2], indices[2]])
        subgoal_out = subgoal_encoder(action_idx, color_idx, object_idx)
        print(f"Output shape: {subgoal_out.shape}")
        print()
        
        # Test full ObservationEncoder
        print("=== ObservationEncoder ===")
        obs_encoder = ObservationEncoder()
        direction = torch.tensor([0, 1])  # right, down
        full_out = obs_encoder(dummy_grid, direction, action_idx, color_idx, object_idx)
        print(f"Output shape: {full_out.shape}")
        print()
        
        print("✓ Observation encoder test passed!")
