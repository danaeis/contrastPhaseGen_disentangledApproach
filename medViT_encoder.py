import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from pathlib import Path

# Add MedViT to Python path
medvit_path = Path(__file__).parent 
sys.path.insert(0, str(medvit_path))

try:
    import MedViT
except ImportError:
    print("Warning: Could not import MedViT. Make sure MedViT repository is cloned in the current directory.")
    print("Falling back to a dummy implementation.")
    
    class MedViT(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            self.num_features = 64
            
        def forward(self, x):
            return self.features(x).flatten(1)

try:
    from MedViT.MedViT import MedViT_small  # Adjusted import for direct file structure
except ImportError:
    raise ImportError("Failed to import MedViT. Ensure the MedViT repository is cloned in the project directory and the path is correct.")


class MedViTEncoder3D(nn.Module):
    """
    3D Medical Volume Encoder using MedViT backbone
    Processes 3D volumes slice-by-slice and aggregates features
    """
    
    def __init__(self, 
                 model_size='small',
                 pretrained_path=None,
                 latent_dim=512,
                 aggregation_method='lstm',
                 slice_sampling='uniform',
                 max_slices=32):
        """
        Args:
            model_size: 'tiny', 'small', 'base' (MedViT model size)
            pretrained_path: Path to pretrained MedViT weights
            latent_dim: Final latent representation dimension
            aggregation_method: 'lstm', 'attention', 'mean', 'max'
            slice_sampling: 'uniform', 'all' (how to sample slices from 3D volume)
            max_slices: Maximum number of slices to process
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.aggregation_method = aggregation_method
        self.slice_sampling = slice_sampling
        self.max_slices = max_slices
        
        print(f"Initializing MedViT encoder (size: {model_size}, latent_dim: {latent_dim})")
        
        # Initialize MedViT backbone without fallback
        self.medvit = self._create_medvit_model(model_size)
        
        # Load pretrained weights if available
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained_weights(pretrained_path)
        else:
            print(f"Warning: Pretrained weights not found at {pretrained_path}")
            print("Training from scratch...")
        
        # # Remove classification head and get feature dimension
        # self.medvit.head = nn.Identity()
        # medvit_features = self._get_medvit_feature_dim()
        
        # Get actual MedViT feature dimension
        medvit_features = self._get_medvit_feature_dim()
        print(f"MedViT feature dimension: {medvit_features}")

        # Feature projection layer
        self.feature_projection = nn.Linear(medvit_features, latent_dim)
        
        # Aggregation layers
        if aggregation_method == 'lstm':
            self.aggregator = nn.LSTM(
                latent_dim, latent_dim, 
                num_layers=2, batch_first=True, bidirectional=True
            )
            self.final_projection = nn.Linear(latent_dim * 2, latent_dim)
        elif aggregation_method == 'attention':
            self.aggregator = nn.MultiheadAttention(
                latent_dim, num_heads=8, batch_first=True
            )
            self.attention_weights = nn.Linear(latent_dim, 1)
            self.final_projection = nn.Identity()
        elif aggregation_method in ['mean', 'max']:
            self.aggregator = None
            self.final_projection = nn.Identity()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        # Normalization layers
        self.slice_norm = nn.LayerNorm(latent_dim)
        self.final_norm = nn.LayerNorm(latent_dim)
        
    def _create_medvit_model(self, model_size):
        """Create MedViT model based on size with fallback"""
        """Create MedViT model based on size"""
        if model_size == 'small':
            return MedViT_small()  # Use the specific constructor from the repo
        elif model_size == 'base':
            # Add similar for base if needed
            pass
        elif model_size == 'large':
            # Add similar for large if needed
            pass
        else:
            raise ValueError(f"Unsupported model size: {model_size}")

    #     try:
    #         # Try to create MedViT model with different possible configurations
    #         configs = {
    #             'tiny': {
    #                 'img_size': 224,
    #                 'patch_size': 16,
    #                 'num_classes': 1000,
    #                 'embed_dims': [64, 128, 256, 512],
    #                 'num_heads': [1, 2, 4, 8],
    #                 'mlp_ratios': [8, 8, 4, 4],
    #                 'depths': [2, 2, 2, 2]
    #             },
    #             'small': {
    #                 'img_size': 224,
    #                 'patch_size': 16,
    #                 'num_classes': 1000,
    #                 'embed_dims': [64, 128, 256, 512],
    #                 'num_heads': [1, 2, 4, 8],
    #                 'mlp_ratios': [8, 8, 4, 4],
    #                 'depths': [3, 3, 5, 2]
    #             },
    #             'base': {
    #                 'img_size': 224,
    #                 'patch_size': 16,
    #                 'num_classes': 1000,
    #                 'embed_dims': [96, 192, 384, 768],
    #                 'num_heads': [1, 2, 4, 8],
    #                 'mlp_ratios': [8, 8, 4, 4],
    #                 'depths': [3, 3, 5, 2]
    #             }
    #         }
            
    #         config = configs.get(model_size, configs['small'])
            
    #         # Try different MedViT constructor signatures
    #         try:
    #             return MedViT(**config)
    #         except TypeError:
    #             # Fallback to simpler constructor
    #             return MedViT(
    #                 img_size=config['img_size'],
    #                 patch_size=config['patch_size'],
    #                 num_classes=config['num_classes']
    #             )
                
    #     except Exception as e:
    #         print(f"Error creating MedViT: {e}")
    #         print("Using fallback CNN model...")
    #         return self._create_fallback_model()
    
    # def _create_fallback_model(self):
    #     """Create a fallback CNN model if MedViT fails"""
    #     return nn.Sequential(
    #         nn.Conv2d(3, 64, 7, 2, 3),
    #         nn.BatchNorm2d(64),
    #         nn.ReLU(),
    #         nn.AdaptiveAvgPool2d(1),
    #         nn.Flatten(),
    #         nn.Linear(64, 512)
    #     )
    
    def _get_medvit_feature_dim(self):
        # """Get the feature dimension of MedViT model"""
        # self.medvit.eval()
        # with torch.no_grad():
            # try:
            #     dummy_input = torch.randn(1, 3, 224, 224)
            #     features = self.medvit(dummy_input)
                
            #     # Handle different output formats
            #     if isinstance(features, tuple):
            #         features = features[0]
            #     if isinstance(features, dict):
            #         # Try common keys
            #         for key in ['features', 'logits', 'out']:
            #             if key in features:
            #                 features = features[key]
            #                 break
            #         else:
            #             features = list(features.values())[0]
                
            #     # Ensure 2D tensor
            #     while features.dim() > 2:
            #         features = features.mean(dim=-1)
                
            #     feature_dim = features.shape[-1]
            #     print(f"Detected MedViT feature dimension: {feature_dim}")
            #     return feature_dim
                
            # except Exception as e:
            #     print(f"Error detecting feature dimension: {e}")
            #     return 512  # Fallback dimension
            # finally:
            #     self.medvit.train()
        """Get the feature dimension of MedViT model"""
        original_mode = self.medvit.training
        self.medvit.eval()
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                features = self.medvit(dummy_input)
                # Assuming MedViT outputs features directly
                feature_dim = features.shape[-1]
                print(f"Detected MedViT feature dimension: {feature_dim}")
                return feature_dim
        finally:
            self.medvit.train(mode=original_mode)
    
    def _load_pretrained_weights(self, checkpoint_path):
        # """Load pretrained MedViT weights with better error handling"""
        # try:
        #     print(f"Loading pretrained weights from {checkpoint_path}")
        #     checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
        #     # Handle different checkpoint formats
        #     if 'model' in checkpoint:
        #         state_dict = checkpoint['model']
        #     elif 'state_dict' in checkpoint:
        #         state_dict = checkpoint['state_dict']
        #     elif 'model_state_dict' in checkpoint:
        #         state_dict = checkpoint['model_state_dict']
        #     else:
        #         state_dict = checkpoint
            
        #     # Remap keys (common fixes: remove 'module.' or 'backbone.')
        #     remapped_state_dict = {}
        #     for key, value in state_dict.items():
        #         new_key = key
        #         if new_key.startswith('module.'):
        #             new_key = new_key[7:]  # Remove 'module.'
        #         if new_key.startswith('backbone.'):
        #             new_key = new_key[9:]  # Remove 'backbone.'
        #         # Add more remaps if needed (inspect checkpoint keys with print(list(state_dict.keys())[:5]))
        #         remapped_state_dict[new_key] = value

        #     # Get current model state dict
        #     print("state_dict keys:", list(state_dict.keys())[:10])
        #     model_state_dict = self.medvit.state_dict()
        #     print("model_state_dict keys:", list(model_state_dict.keys())[:10])

        #     # Filter compatible weights
        #     compatible_weights = {}
        #     incompatible_keys = []
            
        #     for key, value in state_dict.items():
        #         if key in model_state_dict:
        #             if model_state_dict[key].shape == value.shape:
        #                 compatible_weights[key] = value
        #             else:
        #                 incompatible_keys.append(f"{key}: {value.shape} vs {model_state_dict[key].shape}")
        #         else:
        #             incompatible_keys.append(f"{key}: not found in model")
            
        #     # Load compatible weights
        #     missing_keys, unexpected_keys = self.medvit.load_state_dict(compatible_weights, strict=False)
            
        #     print(f"Loaded {len(compatible_weights)} compatible weights")
        #     if missing_keys:
        #         print(f"Missing keys: {missing_keys[:5]}...")
        #     if unexpected_keys:
        #         print(f"Unexpected keys: {unexpected_keys[:5]}...")
        #     if incompatible_keys:
        #         print(f"Incompatible shapes: {len(incompatible_keys)} keys")
                
        # except Exception as e:
        #     print(f"Error loading pretrained weights: {e}")
        #     print("Continuing with random initialization...")

        """Load pretrained MedViT weights with key mapping if needed"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint if 'state_dict' not in checkpoint else checkpoint['state_dict']
        
        # If keys don't match directly, add mapping (e.g., remove 'module.' prefix if present)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')  # Adjust based on common mismatches
            new_state_dict[name] = v
        
        missing_keys, unexpected_keys = self.medvit.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded weights from {checkpoint_path}")
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
    
    def _sample_slices(self, volume_3d):
        """Sample slices from 3D volume"""
        batch_size, channels, depth, height, width = volume_3d.shape
        
        if self.slice_sampling == 'all' and depth <= self.max_slices:
            slice_indices = list(range(depth))
        else:
            if depth <= self.max_slices:
                slice_indices = list(range(depth))
            else:
                # Uniform sampling
                slice_indices = torch.linspace(0, depth-1, self.max_slices).long().tolist()
        
        return slice_indices
    
    def _preprocess_slice(self, slice_2d):
        """Preprocess 2D slice for MedViT input"""
        batch_size = slice_2d.shape[0]
        
        # Resize to 224x224 if needed
        if slice_2d.shape[2] != 224 or slice_2d.shape[3] != 224:
            slice_2d = F.interpolate(
                slice_2d, size=(224, 224), 
                mode='bilinear', align_corners=False
            )
        
        # Convert grayscale to RGB (MedViT expects 3 channels)
        if slice_2d.shape[1] == 1:
            slice_2d = slice_2d.repeat(1, 3, 1, 1)
        
        return slice_2d
    
    def forward(self, volume_3d):
        """
        Forward pass through 3D MedViT encoder
        
        Args:
            volume_3d: (batch, 1, D, H, W) - 3D medical volume
            
        Returns:
            latent: (batch, latent_dim) - latent representation
        """
        batch_size, channels, depth, height, width = volume_3d.shape
        device = volume_3d.device
        
        # Sample slices to process
        slice_indices = self._sample_slices(volume_3d)
        
        # Process each slice
        slice_features = []
        
        for idx in slice_indices:
            # Extract slice: (batch, 1, H, W)
            slice_2d = volume_3d[:, :, idx, :, :]
            
            # Preprocess for MedViT
            slice_2d = self._preprocess_slice(slice_2d)
            
            # Get features from MedViT without independent autocast
            # (rely on outer context or set enabled=False if not using mixed precision)
            with torch.amp.autocast('cuda', enabled=False):  # Disable here or pass a flag
                medvit_features = self.medvit(slice_2d)
                
                # Handle different output formats
                if isinstance(medvit_features, tuple):
                    medvit_features = medvit_features[0]
                if isinstance(medvit_features, dict):
                    medvit_features = list(medvit_features.values())[0]
                
                # Ensure correct shape
                while medvit_features.dim() > 2:
                    medvit_features = medvit_features.mean(dim=-1)
            
            # Project to latent dimension (now dtypes match)
            slice_latent = self.feature_projection(medvit_features.float())  # Explicit cast if needed
            slice_latent = self.slice_norm(slice_latent)
            
            slice_features.append(slice_latent)
        
        # Stack slice features: (batch, num_slices, latent_dim)
        volume_features = torch.stack(slice_features, dim=1)
        
        # Aggregate features across slices
        if self.aggregation_method == 'lstm':
            lstm_out, (hidden, _) = self.aggregator(volume_features)
            # Concatenate final states from both directions
            volume_latent = self.final_projection(lstm_out[:, -1, :])  # Use last output
            
        elif self.aggregation_method == 'attention':
            # Self-attention across slices
            attended_features, _ = self.aggregator(
                volume_features, volume_features, volume_features
            )
            
            # Weighted average using attention weights
            attention_scores = self.attention_weights(attended_features)
            attention_weights = F.softmax(attention_scores, dim=1)
            volume_latent = torch.sum(attended_features * attention_weights, dim=1)
            
        elif self.aggregation_method == 'mean':
            volume_latent = torch.mean(volume_features, dim=1)
            
        elif self.aggregation_method == 'max':
            volume_latent, _ = torch.max(volume_features, dim=1)
        
        # Final normalization
        volume_latent = self.final_norm(volume_latent)
        
        return volume_latent


def create_medvit_encoder(config):
    """Factory function to create MedViT encoder with error handling"""
    try:
        return MedViTEncoder3D(
            model_size=config.get('model_size', 'small'),
            pretrained_path=config.get('pretrained_path', None),
            latent_dim=config.get('latent_dim', 512),
            aggregation_method=config.get('aggregation_method', 'lstm'),
            slice_sampling=config.get('slice_sampling', 'uniform'),
            max_slices=config.get('max_slices', 32)
        )
    except Exception as e:
        print(f"Error creating MedViT encoder: {e}")
        print("Falling back to simple CNN encoder...")
        
        # Import fallback encoder
        from models import Simple3DCNNEncoder
        return Simple3DCNNEncoder(
            in_channels=1,
            latent_dim=config.get('latent_dim', 512),
            img_size=(128, 128, 128)
        )

# Test function
def test_medvit_encoder():
    """Test the MedViT encoder implementation"""
    
    config = {
        'model_size': 'small',
        'latent_dim': 256,
        'aggregation_method': 'lstm',
        'max_slices': 16
    }
    
    try:
        encoder = create_medvit_encoder(config)
        
        # Test with dummy data
        dummy_volume = torch.randn(1, 1, 32, 128, 128)
        
        print(f"Testing MedViT encoder...")
        print(f"Input shape: {dummy_volume.shape}")
        
        with torch.no_grad():
            latent = encoder(dummy_volume)
            print(f"Output shape: {latent.shape}")
            print(f"Expected latent_dim: {config['latent_dim']}")
            
        print("✓ MedViT encoder test passed!")
        return True
        
    except Exception as e:
        print(f"✗ MedViT encoder test failed: {e}")
        return False


if __name__ == "__main__":
    test_medvit_encoder()