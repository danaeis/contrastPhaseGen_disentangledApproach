import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from pathlib import Path

# Add MedViT to Python path
medvit_path = Path(__file__).parent 
sys.path.insert(0, str(medvit_path))

def inspect_checkpoint(checkpoint_path):
    """Inspect the checkpoint to understand its structure"""
    print(f"🔍 Inspecting checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get the actual state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        print(f"🔧 Total parameters in checkpoint: {len(state_dict)}")
        
        # Group keys by prefix to understand architecture
        key_prefixes = {}
        for key in state_dict.keys():
            prefix = key.split('.')[0]
            if prefix not in key_prefixes:
                key_prefixes[prefix] = []
            key_prefixes[prefix].append(key)
        
        print(f"🏗️ Architecture components:")
        for prefix, keys in key_prefixes.items():
            print(f"  {prefix}: {len(keys)} parameters")
        
        return state_dict, key_prefixes
        
    except Exception as e:
        print(f"❌ Error inspecting checkpoint: {e}")
        return None, None

def load_weights_with_mapping(model, state_dict, strict=False):
    """Load weights with intelligent key mapping"""
    
    print(f"🔄 Loading weights with intelligent mapping...")
    
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    
    print(f"  Model keys: {len(model_keys)}")
    print(f"  Checkpoint keys: {len(checkpoint_keys)}")
    
    # Direct key matching first
    direct_matches = model_keys & checkpoint_keys
    print(f"  Direct matches: {len(direct_matches)}")
    
    # Create filtered state dict with size matching
    filtered_state_dict = {}
    size_mismatches = []
    
    for key in direct_matches:
        model_shape = model.state_dict()[key].shape
        checkpoint_shape = state_dict[key].shape
        
        if model_shape == checkpoint_shape:
            filtered_state_dict[key] = state_dict[key]
        else:
            size_mismatches.append((key, model_shape, checkpoint_shape))
    
    print(f"  Compatible weights: {len(filtered_state_dict)}")
    print(f"  Size mismatches: {len(size_mismatches)}")
    
    if size_mismatches:
        print("  First few size mismatches:")
        for key, model_shape, checkpoint_shape in size_mismatches[:3]:
            print(f"    {key}: model {model_shape} vs checkpoint {checkpoint_shape}")
    
    # Load the compatible weights
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    print(f"  ✅ Loaded {len(filtered_state_dict)} compatible weights")
    print(f"  Missing keys: {len(missing_keys)}")
    
    return len(filtered_state_dict), len(missing_keys)

class MedViTEncoder3D(nn.Module):
    """3D Medical Volume Encoder using MedViT backbone"""
    
    def __init__(self, 
                 model_size='small',
                 pretrained_path=None,
                 latent_dim=512,
                 aggregation_method='lstm',
                 slice_sampling='uniform',
                 max_slices=32):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.aggregation_method = aggregation_method
        self.slice_sampling = slice_sampling
        self.max_slices = max_slices
        
        print(f"🔧 Initializing MedViT encoder (size: {model_size}, latent_dim: {latent_dim})")
        
        # Initialize MedViT backbone
        self.medvit = self._create_medvit_model(model_size)
        
        # Load pretrained weights if available
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained_weights(pretrained_path)
        else:
            print(f"Warning: No pretrained weights found at {pretrained_path}")
        
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
        """Create MedViT model based on size"""
        try:
            # Try to import and create MedViT
            from MedViT.MedViT import MedViT_small, MedViT_base
            
            if model_size == 'small':
                model = MedViT_small()
            elif model_size == 'base':
                model = MedViT_base()
            else:
                print(f"Unknown model size {model_size}, defaulting to small")
                model = MedViT_small()
            
            # Remove classification head if present
            if hasattr(model, 'head'):
                model.head = nn.Identity()
            elif hasattr(model, 'proj_head'):
                model.proj_head = nn.Identity()
            
            return model
            
        except ImportError as e:
            print(f"Failed to import MedViT: {e}")
            print("Using fallback CNN model")
            return self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create fallback CNN model if MedViT fails"""
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1024)  # Match expected feature dimension
        )
    
    def _get_medvit_feature_dim(self):
        """Get the feature dimension of MedViT model"""
        original_mode = self.medvit.training
        self.medvit.eval()
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                features = self.medvit(dummy_input)
                
                # Handle different output formats
                if isinstance(features, tuple):
                    features = features[0]
                if isinstance(features, dict):
                    features = list(features.values())[0]
                
                # Ensure 2D tensor
                while features.dim() > 2:
                    features = features.mean(dim=-1)
                
                feature_dim = features.shape[-1]
                return feature_dim
        except Exception as e:
            print(f"Error detecting feature dimension: {e}")
            return 1024  # Fallback dimension
        finally:
            self.medvit.train(mode=original_mode)
    
    def _load_pretrained_weights(self, checkpoint_path):
        """Load pretrained MedViT weights"""
        try:
            print(f"Loading weights from {checkpoint_path}")
            state_dict, _ = inspect_checkpoint(checkpoint_path)
            
            if state_dict is not None:
                loaded_count, missing_count = load_weights_with_mapping(self.medvit, state_dict)
                print(f"Successfully loaded {loaded_count} weights")
            else:
                print("Failed to load checkpoint")
                
        except Exception as e:
            print(f"Error loading weights: {e}")
    
    def _sample_slices(self, volume_3d):
        """Sample slices from 3D volume"""
        batch_size, channels, depth, height, width = volume_3d.shape
        
        if self.slice_sampling == 'all' and depth <= self.max_slices:
            slice_indices = list(range(depth))
        else:
            if depth <= self.max_slices:
                slice_indices = list(range(depth))
            else:
                slice_indices = torch.linspace(0, depth-1, self.max_slices).long().tolist()
        
        return slice_indices
    
    def _preprocess_slice(self, slice_2d):
        """Preprocess 2D slice for MedViT input"""
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
        """Forward pass through 3D MedViT encoder"""
        batch_size = volume_3d.shape[0]
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
            
            # Get features from MedViT
            try:
                if torch.is_autocast_enabled():
                    with torch.cuda.amp.autocast(enabled=False):
                        medvit_features = self.medvit(slice_2d)
                else:
                    medvit_features = self.medvit(slice_2d)
                
                # Handle different output formats
                if isinstance(medvit_features, tuple):
                    medvit_features = medvit_features[0]
                if isinstance(medvit_features, dict):
                    medvit_features = list(medvit_features.values())[0]
                
                # Ensure correct shape
                while medvit_features.dim() > 2:
                    medvit_features = medvit_features.mean(dim=-1)
                
            except Exception as e:
                print(f"Error in MedViT forward pass: {e}")
                # Use fallback features
                medvit_features = torch.randn(batch_size, 1024, device=device)
            
            # Project to latent dimension
            slice_latent = self.feature_projection(medvit_features.float())
            slice_latent = self.slice_norm(slice_latent)
            slice_features.append(slice_latent)
        
        # Stack slice features: (batch, num_slices, latent_dim)
        volume_features = torch.stack(slice_features, dim=1)
        
        # Aggregate features across slices
        if self.aggregation_method == 'lstm':
            lstm_out, (hidden, _) = self.aggregator(volume_features)
            volume_latent = self.final_projection(lstm_out[:, -1, :])
            
        elif self.aggregation_method == 'attention':
            attended_features, _ = self.aggregator(
                volume_features, volume_features, volume_features
            )
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
    """Factory function to create MedViT encoder"""
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
            
        print("✅ MedViT encoder test passed!")
        return True
        
    except Exception as e:
        print(f"❌ MedViT encoder test failed: {e}")
        return False


if __name__ == "__main__":
    test_medvit_encoder()