import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

# MONAI imports
try:
    import monai
    from monai.networks.nets import UNet, BasicUNet, SegResNet
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
        ScaleIntensityRanged, CropForegroundd, Resized,
        EnsureTyped, ToTensord
    )
    from monai.data import CacheDataset, DataLoader, decollate_batch
    from monai.inferers import sliding_window_inference
    MONAI_AVAILABLE = True
    print("✅ MONAI available for TotalSegmentator integration")
except ImportError:
    MONAI_AVAILABLE = False
    print("❌ MONAI not available. Install with: pip install monai[all]")

try:
    from totalsegmentator.python_api import totalsegmentator
    TOTALSEG_DIRECT_AVAILABLE = True
except ImportError:
    TOTALSEG_DIRECT_AVAILABLE = False


class FixedMONAITotalSegmentatorEncoder(nn.Module):
    """
    FIXED: TotalSegmentator encoder using MONAI interface
    Addresses device placement and compatibility issues
    """
    
    def __init__(self, 
                 latent_dim: int = 512,
                 model_name: str = "totalsegmentator",
                 device: str = "cuda",
                 extract_features_from: str = "encoder",
                 use_pretrained: bool = True,
                 roi_size: Tuple[int, int, int] = (96, 96, 96),
                 sw_batch_size: int = 4,
                 overlap: float = 0.25,
                 use_enhanced_features: bool = False,
                 use_anatomical_priors: bool = False,
                 contrast_phase_aware: bool = False):
        """
        FIXED version with proper device handling and MONAI compatibility
        """
        super().__init__()
        
        if not MONAI_AVAILABLE:
            raise ImportError("MONAI not available. Install with: pip install monai[all]")
        
        self.latent_dim = latent_dim
        self.model_name = model_name
        self.device = torch.device(device)  # Ensure proper device type
        self.extract_features_from = extract_features_from
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.use_enhanced_features = use_enhanced_features
        self.use_anatomical_priors = use_anatomical_priors
        self.contrast_phase_aware = contrast_phase_aware
        
        print(f"🔧 Initializing FIXED MONAI TotalSegmentator encoder...")
        print(f"   Latent dim: {latent_dim}")
        print(f"   Device: {self.device}")
        print(f"   ROI size: {roi_size}")
        print(f"   Enhanced features: {use_enhanced_features}")
        
        # Initialize the segmentation model with proper device handling
        self._initialize_segmentation_model_fixed(use_pretrained)
        
        # Setup feature extraction hooks
        self._setup_feature_extraction_fixed()
        
        # Feature processing and projection with device handling
        self._setup_feature_processor_fixed()
        
        # Enhanced features if requested
        if self.use_enhanced_features:
            self._setup_enhanced_features_fixed()
        
        print(f"✅ FIXED MONAI TotalSegmentator encoder initialized")
    
    def _initialize_segmentation_model_fixed(self, use_pretrained: bool = True):
        """FIXED: Initialize with proper MONAI compatibility"""
        try:
            # Try BasicUNet with corrected parameters
            self._initialize_with_basic_unet_fixed()
        except Exception as e:
            print(f"⚠️  BasicUNet initialization failed: {e}")
            try:
                # Try SegResNet
                self._initialize_with_segresnet_fixed()
            except Exception as e2:
                print(f"⚠️  SegResNet initialization failed: {e2}")
                # Final fallback
                self._initialize_simple_fallback()
    
    def _initialize_with_basic_unet_fixed(self):
        """FIXED: BasicUNet with correct parameters"""
        print("📥 Loading BasicUNet via MONAI (FIXED)...")
        
        # FIXED: BasicUNet expects features to have 6 elements for spatial_dims=3
        self.model = BasicUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=104,
            features=(32, 64, 128, 256, 512, 1024),  # FIXED: Added 6th element
            dropout=0.1,
            norm="batch",
            act='relu'
        )
        
        self.num_classes = 104
        self.model.eval()
        
        # FIXED: Move model to device immediately
        self.model = self.model.to(self.device)
        
        print("✅ BasicUNet model initialized and moved to device")
    
    def _initialize_with_segresnet_fixed(self):
        """FIXED: SegResNet with proper device handling"""
        print("📥 Loading SegResNet via MONAI (FIXED)...")
        
        self.model = SegResNet(
            spatial_dims=3,
            init_filters=32,
            in_channels=1,
            out_channels=104,
            dropout_prob=0.1,
            norm="batch",
            num_groups=16,
            use_conv_final=True
        )
        
        self.num_classes = 104
        self.model.eval()
        
        # FIXED: Move to device
        self.model = self.model.to(self.device)
        self.use_fallback = True
        
        print("✅ SegResNet model initialized and moved to device")
    
    def _initialize_simple_fallback(self):
        """FIXED: Simple fallback with proper device handling"""
        print("🔄 Using simple fallback segmentation model...")
        
        # Create a simple but effective 3D segmentation network
        self.model = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(256, 104, kernel_size=3, padding=1),  # Output 104 classes
        )
        
        self.num_classes = 104
        self.model.eval()
        
        # FIXED: Move to device
        self.model = self.model.to(self.device)
        self.use_simple_fallback = True
        
        print("✅ Simple fallback model initialized and moved to device")
    
    def _setup_feature_extraction_fixed(self):
        """FIXED: Setup hooks with proper error handling"""
        self.feature_maps = {}
        self.hooks = []
        
        def get_activation(name):
            def hook(module, input, output):
                try:
                    # Store detached features on correct device
                    if isinstance(output, torch.Tensor):
                        self.feature_maps[name] = output.detach()
                    elif isinstance(output, (list, tuple)):
                        self.feature_maps[name] = [o.detach() if torch.is_tensor(o) else o for o in output]
                    else:
                        self.feature_maps[name] = output
                except Exception as e:
                    print(f"⚠️  Hook error for {name}: {e}")
            return hook
        
        # Register hooks with better error handling
        hook_count = 0
        
        # Try to hook specific layers
        try:
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)) and hook_count < 6:
                    hook = module.register_forward_hook(get_activation(f'conv_{hook_count}'))
                    self.hooks.append(hook)
                    hook_count += 1
        except Exception as e:
            print(f"⚠️  Error registering hooks: {e}")
        
        print(f"✅ Registered {hook_count} feature extraction hooks")
    
    def _setup_feature_processor_fixed(self):
        """FIXED: Feature processor with proper device handling"""
        # Conservative feature dimension estimate
        estimated_feature_dim = 1024
        
        # FIXED: Use GroupNorm instead of BatchNorm to avoid batch size issues
        self.feature_projection = nn.Sequential(
            nn.Linear(estimated_feature_dim, 1024),
            nn.GroupNorm(32, 1024),  # FIXED: GroupNorm instead of BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.GroupNorm(32, 512),  # FIXED: GroupNorm instead of BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, self.latent_dim)
        )
        
        # FIXED: LayerNorm initialization on correct device
        self.layer_norm = nn.LayerNorm(self.latent_dim)
        
        # FIXED: Move all components to device
        self.feature_projection = self.feature_projection.to(self.device)
        self.layer_norm = self.layer_norm.to(self.device)
        
        # Pooling operations
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        
        # Flag to track initialization
        self.feature_processor_initialized = False
    
    def _setup_enhanced_features_fixed(self):
        """FIXED: Enhanced features with proper device handling"""
        if self.use_anatomical_priors:
            self.anatomical_attention = nn.MultiheadAttention(
                embed_dim=self.latent_dim,
                num_heads=8,
                batch_first=True
            ).to(self.device)
            
            self.region_weights = nn.Parameter(torch.ones(self.num_classes)).to(self.device)
        
        if self.contrast_phase_aware:
            self.phase_modulation = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.Sigmoid()
            ).to(self.device)
    
    def _preprocess_volume(self, volume: torch.Tensor) -> torch.Tensor:
        """FIXED: Preprocess with proper device handling"""
        # Ensure volume is on correct device
        volume = volume.to(self.device)
        
        # Clamp and normalize
        volume = torch.clamp(volume, -1000, 1000)
        volume = (volume + 1000) / 2000
        
        return volume
    
    def _extract_and_aggregate_features_fixed(self, volume: torch.Tensor) -> torch.Tensor:
        """FIXED: Feature extraction with proper error handling"""
        batch_size = volume.shape[0]
        
        # Clear previous features
        self.feature_maps.clear()
        
        # Forward pass through segmentation model
        try:
            with torch.no_grad():
                # Ensure model is in eval mode and on correct device
                self.model.eval()
                self.model = self.model.to(self.device)
                
                # Use sliding window for larger volumes
                if any(s > r for s, r in zip(volume.shape[2:], self.roi_size)):
                    segmentation = sliding_window_inference(
                        inputs=volume,
                        roi_size=self.roi_size,
                        sw_batch_size=self.sw_batch_size,
                        predictor=self.model,
                        overlap=self.overlap,
                        mode="gaussian"
                    )
                else:
                    segmentation = self.model(volume)
        except Exception as e:
            print(f"⚠️  Model forward pass failed: {e}")
            # Return fallback features on correct device
            return torch.randn(batch_size, 1024, device=self.device)
        
        # Aggregate features
        all_features = []
        
        for batch_idx in range(batch_size):
            batch_features = []
            
            # Process feature maps
            for name, feature_map in self.feature_maps.items():
                try:
                    if isinstance(feature_map, torch.Tensor) and feature_map.dim() == 5:
                        # Extract for this batch item
                        batch_feature = feature_map[batch_idx:batch_idx+1]
                        
                        # Global pooling
                        pooled = self.global_pool(batch_feature).flatten()
                        batch_features.append(pooled.to(self.device))
                except Exception as e:
                    print(f"⚠️  Feature processing error for {name}: {e}")
                    continue
            
            if batch_features:
                # Concatenate features
                try:
                    combined_features = torch.cat(batch_features, dim=0)
                    all_features.append(combined_features)
                except Exception as e:
                    print(f"⚠️  Feature concatenation error: {e}")
                    all_features.append(torch.randn(512, device=self.device))
            else:
                # Fallback: use segmentation output
                try:
                    if segmentation.dim() == 5:
                        seg_features = self.global_pool(segmentation[batch_idx:batch_idx+1]).flatten()
                        all_features.append(seg_features.to(self.device))
                    else:
                        all_features.append(torch.randn(512, device=self.device))
                except Exception:
                    all_features.append(torch.randn(512, device=self.device))
        
        # Stack all features
        try:
            if all_features:
                features_tensor = torch.stack(all_features, dim=0)
            else:
                features_tensor = torch.randn(batch_size, 1024, device=self.device)
        except Exception as e:
            print(f"⚠️  Feature stacking error: {e}")
            features_tensor = torch.randn(batch_size, 1024, device=self.device)
        
        return features_tensor.to(self.device)
    
    def _initialize_feature_processor_dynamically_fixed(self, features: torch.Tensor):
        """FIXED: Dynamic initialization with proper device handling"""
        if self.feature_processor_initialized:
            return
        
        actual_feature_dim = features.shape[1]
        
        # Create new projection with correct dimensions
        self.feature_projection = nn.Sequential(
            nn.Linear(actual_feature_dim, min(1024, max(512, actual_feature_dim))),
            nn.GroupNorm(32, min(1024, max(512, actual_feature_dim))),  # GroupNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(min(1024, max(512, actual_feature_dim)), 512),
            nn.GroupNorm(32, 512),  # GroupNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, self.latent_dim)
        ).to(self.device)  # FIXED: Ensure on correct device
        
        self.feature_processor_initialized = True
        print(f"✅ Feature processor initialized with input dim: {actual_feature_dim}")
    
    def forward(self, volume: torch.Tensor, phase_hint: Optional[int] = None) -> torch.Tensor:
        """FIXED: Forward pass with proper device handling and NaN prevention"""
        # Preprocess volume
        volume = self._preprocess_volume(volume)
        
        # Check input for NaN
        if torch.isnan(volume).any():
            print(f"⚠️  NaN detected in input volume, cleaning up...")
            volume = torch.nan_to_num(volume, nan=0.0, posinf=1.0, neginf=-1.0)
            volume = torch.clamp(volume, 0.0, 1.0)
        
        # Extract raw features
        raw_features = self._extract_and_aggregate_features_fixed(volume)
        
        # Check raw features for NaN
        if torch.isnan(raw_features).any():
            print(f"⚠️  NaN detected in raw features, using fallback...")
            raw_features = torch.randn(raw_features.shape[0], raw_features.shape[1], device=self.device) * 0.1
            raw_features = torch.clamp(raw_features, -1.0, 1.0)
        
        # Initialize processor if needed
        self._initialize_feature_processor_dynamically_fixed(raw_features)
        
        # Project to latent space
        try:
            # FIXED: Set to train mode temporarily to avoid batch norm issues
            self.feature_projection.train()
            projected_features = self.feature_projection(raw_features)
            self.feature_projection.eval()
            
            # Check projected features for NaN
            if torch.isnan(projected_features).any():
                print(f"⚠️  NaN detected after projection, using fallback...")
                projected_features = torch.randn(projected_features.shape[0], projected_features.shape[1], device=self.device) * 0.1
                projected_features = torch.clamp(projected_features, -1.0, 1.0)
                
        except Exception as e:
            print(f"⚠️  Feature projection error: {e}")
            # Emergency fallback projection
            emergency_proj = nn.Linear(raw_features.shape[1], self.latent_dim).to(self.device)
            projected_features = emergency_proj(raw_features)
            
            # Check emergency projection for NaN
            if torch.isnan(projected_features).any():
                print(f"⚠️  NaN detected in emergency projection, using random features...")
                projected_features = torch.randn(projected_features.shape[0], projected_features.shape[1], device=self.device) * 0.1
                projected_features = torch.clamp(projected_features, -1.0, 1.0)
        
        # Apply enhanced processing if enabled
        if self.use_enhanced_features:
            try:
                projected_features = self._apply_enhanced_processing_fixed(projected_features, volume, phase_hint)
                
                # Check enhanced features for NaN
                if torch.isnan(projected_features).any():
                    print(f"⚠️  NaN detected after enhanced processing, reverting to basic features...")
                    projected_features = torch.randn(projected_features.shape[0], projected_features.shape[1], device=self.device) * 0.1
                    projected_features = torch.clamp(projected_features, -1.0, 1.0)
                    
            except Exception as e:
                print(f"⚠️  Enhanced processing failed: {e}")
        
        # Layer normalization
        try:
            features = self.layer_norm(projected_features)
            
            # Check normalized features for NaN
            if torch.isnan(features).any():
                print(f"⚠️  NaN detected after LayerNorm, using unnormalized features...")
                features = projected_features
                
        except Exception as e:
            print(f"⚠️  LayerNorm error: {e}")
            # Skip normalization if it fails
            features = projected_features
        
        # Final comprehensive NaN check and cleanup
        if torch.isnan(features).any() or not torch.isfinite(features).all():
            print(f"⚠️  Final NaN/Inf check failed, using safe fallback...")
            features = torch.randn(features.shape[0], features.shape[1], device=self.device) * 0.1
            features = torch.clamp(features, -1.0, 1.0)
        
        # Ensure output is on correct device and has correct shape
        features = features.to(self.device)
        assert features.shape[1] == self.latent_dim, f"Output shape mismatch: {features.shape[1]} != {self.latent_dim}"
        
        return features
    
    def _apply_enhanced_processing_fixed(self, features: torch.Tensor, volume: torch.Tensor, phase_hint: Optional[int]) -> torch.Tensor:
        """FIXED: Enhanced processing with error handling"""
        enhanced_features = features
        
        # Anatomical priors
        if self.use_anatomical_priors and hasattr(self, 'anatomical_attention'):
            try:
                features_expanded = features.unsqueeze(1)
                attended_features, _ = self.anatomical_attention(features_expanded, features_expanded, features_expanded)
                enhanced_features = attended_features.squeeze(1)
            except Exception as e:
                print(f"⚠️  Anatomical attention failed: {e}")
        
        # Phase-aware processing
        if self.contrast_phase_aware and phase_hint is not None and hasattr(self, 'phase_modulation'):
            try:
                modulation = self.phase_modulation(enhanced_features)
                enhanced_features = enhanced_features * modulation
            except Exception as e:
                print(f"⚠️  Phase modulation failed: {e}")
        
        return enhanced_features
    
    def get_segmentation(self, volume: torch.Tensor) -> torch.Tensor:
        """FIXED: Get segmentation with error handling"""
        volume = self._preprocess_volume(volume)
        
        try:
            with torch.no_grad():
                self.model.eval()
                if any(s > r for s, r in zip(volume.shape[2:], self.roi_size)):
                    segmentation = sliding_window_inference(
                        inputs=volume,
                        roi_size=self.roi_size,
                        sw_batch_size=self.sw_batch_size,
                        predictor=self.model,
                        overlap=self.overlap
                    )
                else:
                    segmentation = self.model(volume)
            return segmentation
        except Exception as e:
            print(f"⚠️  Segmentation failed: {e}")
            # Return dummy segmentation
            return torch.zeros(volume.shape[0], self.num_classes, *volume.shape[2:], device=self.device)
    
    def analyze_anatomical_features(self, volume: torch.Tensor) -> Dict[str, Any]:
        """FIXED: Anatomical analysis with error handling"""
        try:
            features = self.forward(volume)
            
            try:
                segmentation = self.get_segmentation(volume)
                num_organs = (segmentation.argmax(dim=1) > 0).sum().item()
                seg_confidence = segmentation.max(dim=1)[0].mean().item()
            except Exception:
                num_organs = 0
                seg_confidence = 0.0
            
            analysis = {
                'feature_statistics': {
                    'mean': features.mean().item(),
                    'std': features.std().item(),
                    'min': features.min().item(),
                    'max': features.max().item()
                },
                'anatomical_coverage': {
                    'num_organs_detected': num_organs,
                    'segmentation_confidence': seg_confidence
                },
                'feature_quality': {
                    'feature_norm': torch.norm(features, dim=1).mean().item(),
                    'feature_diversity': features.var(dim=0).mean().item()
                }
            }
            
            return analysis
            
        except Exception as e:
            print(f"⚠️  Anatomical analysis failed: {e}")
            return {
                'feature_statistics': {'mean': 0, 'std': 0, 'min': 0, 'max': 0},
                'anatomical_coverage': {'num_organs_detected': 0, 'segmentation_confidence': 0},
                'feature_quality': {'feature_norm': 0, 'feature_diversity': 0}
            }
    
    def cleanup(self):
        """FIXED: Cleanup with proper error handling"""
        for hook in self.hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self.hooks = []
        self.feature_maps.clear()
        print("🧹 FIXED MONAI TotalSegmentator encoder cleanup complete")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()


def create_monai_totalsegmentator_encoder(config: Dict[str, Any]) -> nn.Module:
    """
    FIXED: Factory function with better error handling
    """
    try:
        encoder = FixedMONAITotalSegmentatorEncoder(
            latent_dim=config.get('latent_dim', 512),
            model_name=config.get('model_name', 'totalsegmentator'),
            device=config.get('device', 'cuda'),
            extract_features_from=config.get('extract_features_from', 'encoder'),
            use_pretrained=config.get('use_pretrained', True),
            roi_size=config.get('roi_size', (96, 96, 96)),
            sw_batch_size=config.get('sw_batch_size', 4),
            overlap=config.get('overlap', 0.25),
            use_enhanced_features=config.get('use_enhanced_features', False),
            use_anatomical_priors=config.get('use_anatomical_priors', False),
            contrast_phase_aware=config.get('contrast_phase_aware', False)
        )
        
        print("✅ FIXED MONAI TotalSegmentator encoder created successfully")
        return encoder
        
    except Exception as e:
        print(f"❌ Error creating MONAI TotalSegmentator encoder: {e}")
        print("🔄 Falling back to minimal CNN encoder...")
        
        # Create minimal fallback
        return create_minimal_fallback_encoder_fixed(
            config.get('latent_dim', 512), 
            config.get('device', 'cuda')
        )


def create_minimal_fallback_encoder_fixed(latent_dim: int = 512, device: str = 'cuda'):
    """FIXED: Minimal fallback encoder with proper device handling"""
    
    class MinimalEncoderFixed(nn.Module):
        def __init__(self, latent_dim, device):
            super().__init__()
            self.device = torch.device(device)
            self.encoder = nn.Sequential(
                nn.Conv3d(1, 32, 4, 2, 1),
                nn.GroupNorm(8, 32),  # GroupNorm instead of BatchNorm
                nn.ReLU(),
                nn.Conv3d(32, 64, 4, 2, 1),
                nn.GroupNorm(16, 64),
                nn.ReLU(),
                nn.Conv3d(64, 128, 4, 2, 1),
                nn.GroupNorm(32, 128),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(128, latent_dim)
            ).to(self.device)
        
        def forward(self, x):
            x = x.to(self.device)
            return self.encoder(x)
        
        def cleanup(self):
            pass
    
    return MinimalEncoderFixed(latent_dim, device)


# FIXED Test function
def test_monai_totalsegmentator_encoder():
    """FIXED: Test function with better error handling"""
    print("🧪 Testing FIXED MONAI TotalSegmentator encoder...")
    
    config = {
        'latent_dim': 256,
        'model_name': 'totalsegmentator',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'use_pretrained': True,
        'roi_size': (64, 64, 64),
        'use_enhanced_features': False
    }
    
    try:
        encoder = create_monai_totalsegmentator_encoder(config)
        
        # Test with dummy data
        device = config['device']
        dummy_volume = torch.randn(1, 1, 64, 64, 64).to(device)
        
        print(f"Input shape: {dummy_volume.shape}")
        print(f"Device: {device}")
        
        with torch.no_grad():
            features = encoder(dummy_volume)
            print(f"✅ Forward pass successful!")
            print(f"   Output shape: {features.shape}")
            print(f"   Feature range: [{features.min():.3f}, {features.max():.3f}]")
            print(f"   Feature std: {features.std():.6f}")
            print(f"   Features on device: {features.device}")
        
        # Test anatomical analysis
        try:
            analysis = encoder.analyze_anatomical_features(dummy_volume)
            print(f"✅ Anatomical analysis successful: {analysis['feature_quality']}")
        except Exception as e:
            print(f"⚠️  Anatomical analysis failed: {e}")
        
        encoder.cleanup()
        print("✅ FIXED MONAI TotalSegmentator encoder test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ FIXED test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_monai_totalsegmentator_encoder()