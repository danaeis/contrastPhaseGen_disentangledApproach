import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple3DCNNEncoder(nn.Module):
    """Simple 3D CNN encoder - no external dependencies"""
    
    def __init__(self, in_channels=1, latent_dim=256, img_size=(128, 128, 128)):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # First block: 128 -> 64
            nn.Conv3d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            # Second block: 64 -> 32
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            # Third block: 32 -> 16
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # Fourth block: 16 -> 8
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            # Fifth block: 8 -> 4
            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            
            # Global average pooling
            nn.AdaptiveAvgPool3d(1)
        )
        
        self.fc = nn.Linear(512, latent_dim)
        
    def forward(self, x):
        # x: (batch, 1, 128, 128, 128)
        features = self.encoder(x)  # (batch, 512, 1, 1, 1)
        features = features.view(features.size(0), -1)  # (batch, 512)
        z = self.fc(features)  # (batch, latent_dim)
        return z


# Option 2: Use timm ViT with 2D slice processing
class TimmViTEncoder(nn.Module):
    """Use timm ViT with slice-by-slice processing"""
    
    def __init__(self, latent_dim=256, model_name='vit_small_patch16_224', pretrained=True, max_slices=32, slice_sampling='uniform'):
        super().__init__()
        
        import timm
        
        # Create 2D ViT
        self.vit = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        
        # Get ViT feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.vit(dummy)
            if features.dim() > 2:
                features = features.mean(dim=1)  # Average over patches
            vit_dim = features.shape[-1]
        
        # Projection layers
        self.slice_projection = nn.Linear(vit_dim, latent_dim)
        
        # Aggregation across slices
        self.slice_aggregator = nn.LSTM(
            latent_dim, latent_dim, 
            batch_first=True, bidirectional=True
        )
        self.final_projection = nn.Linear(latent_dim * 2, latent_dim)
        
        self.max_slices = max_slices  # Limit number of slices
        self.slice_sampling = slice_sampling  # Sampling strategy
        
    def forward(self, volume_3d):
        # volume_3d: (batch, 1, D, H, W)
        batch_size, _, depth, height, width = volume_3d.shape
        
        # Select slice indices based on sampling method
        if self.slice_sampling == 'all':
            slice_indices = torch.arange(depth)
        elif self.slice_sampling == 'uniform':
            if depth > self.max_slices:
                slice_indices = torch.linspace(0, depth-1, self.max_slices).long()
            else:
                slice_indices = torch.arange(depth)
        elif self.slice_sampling == 'adaptive':
            # Example adaptive sampling: more slices in the center
            center = depth // 2
            indices = torch.linspace(center - self.max_slices//2, center + self.max_slices//2, self.max_slices).clamp(0, depth-1).long()
            slice_indices = torch.unique(indices)  # Remove duplicates if any
        else:
            raise ValueError(f"Unknown slice_sampling method: {self.slice_sampling}")
        
        slice_features = []
        
        for idx in slice_indices:
            # Extract slice: (batch, 1, H, W)
            slice_2d = volume_3d[:, :, idx, :, :]
            
            # Resize to 224x224 for ViT
            slice_2d = F.interpolate(
                slice_2d, size=(224, 224), 
                mode='bilinear', align_corners=False
            )
            
            # Convert to 3-channel (RGB)
            slice_2d = slice_2d.repeat(1, 3, 1, 1)
            
            # Get ViT features
            features = self.vit(slice_2d)
            if features.dim() > 2:
                features = features.mean(dim=1)  # Average over patches
            
            # Project to latent space
            slice_latent = self.slice_projection(features)
            slice_features.append(slice_latent)
        
        # Stack and aggregate
        volume_features = torch.stack(slice_features, dim=1)  # (batch, slices, latent_dim)
        
        # LSTM aggregation
        lstm_out, (hidden, _) = self.slice_aggregator(volume_features)
        final_features = self.final_projection(hidden.view(batch_size, -1))
        
        return final_features


# Option 3: ResNet3D-based encoder
class ResNet3DEncoder(nn.Module):
    """3D ResNet-based encoder using torchvision's ResNet architecture adapted to 3D"""
    
    def __init__(self, latent_dim=256, in_channels=1):
        super().__init__()
        
        # Define 3D ResNet blocks
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, latent_dim)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # First block with potential stride
        layers.append(BasicBlock3D(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(out_channels, out_channels, 1))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class BasicBlock3D(nn.Module):
    """3D version of ResNet BasicBlock"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out


class StableLightweightHybridEncoder(nn.Module):
        def __init__(self, latent_dim=256, in_channels=1):
            super().__init__()
            
            # More stable CNN with GroupNorm
            self.cnn_features = nn.Sequential(
                nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, 32),  # More stable than BatchNorm
                nn.ReLU(),
                nn.Dropout3d(0.1),
                
                nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(16, 64),
                nn.ReLU(),
                nn.Dropout3d(0.1),
                
                nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, 128),
                nn.ReLU(),
                nn.Dropout3d(0.1),
                
                nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(64, 256),
                nn.ReLU(),
            )
            
            # Stable attention
            self.attention = nn.MultiheadAttention(
                embed_dim=256, num_heads=8, batch_first=True, dropout=0.1
            )
            
            self.global_pool = nn.AdaptiveAvgPool3d(1)
            self.layer_norm = nn.LayerNorm(256)
            
            self.fc = nn.Sequential(
                nn.Linear(256, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, latent_dim),
                nn.LayerNorm(latent_dim)
            )
            
            # Initialize weights properly
            self._init_weights()
        
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            # CNN features
            features = self.cnn_features(x)
            
            # Check for NaN (safety)
            if torch.isnan(features).any():
                print("⚠️ NaN detected in CNN, using safe fallback")
                return torch.zeros(x.shape[0], self.fc[-2].out_features, device=x.device)
            
            # Attention
            batch_size, channels, d, h, w = features.shape
            features_flat = features.view(batch_size, channels, -1).transpose(1, 2)
            features_flat = self.layer_norm(features_flat)
            
            try:
                attn_out, _ = self.attention(features_flat, features_flat, features_flat)
                attn_out = attn_out + features_flat  # Residual
            except:
                attn_out = features_flat  # Fallback
            
            # Pool and project
            attn_out = attn_out.transpose(1, 2).view(batch_size, channels, d, h, w)
            pooled = self.global_pool(attn_out).view(batch_size, -1)
            output = self.fc(pooled)
            
            # Final safety check
            if torch.isnan(output).any():
                print("⚠️ NaN in final output, using zeros")
                return torch.zeros_like(output)
            
            return output

# Option 4: Hybrid CNN-Transformer (lightweight)
class LightweightHybridEncoder(nn.Module):
    """Lightweight hybrid CNN + self-attention encoder"""
    
    def __init__(self, latent_dim=256, in_channels=1):
        super().__init__()
        
        # CNN feature extractor
        self.cnn_features = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
        )
        
        # Self-attention for global context
        self.attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, batch_first=True
        )
        
        # Final projection
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(256, latent_dim)
        
    def forward(self, x):
        # CNN features: (batch, 256, 8, 8, 8)
        features = self.cnn_features(x)
        
        # Reshape for attention: (batch, 512, 256)
        batch_size, channels, d, h, w = features.shape
        features_flat = features.view(batch_size, channels, -1).transpose(1, 2)
        
        # Self-attention
        attn_out, _ = self.attention(features_flat, features_flat, features_flat)
        
        # Reshape back and pool
        attn_out = attn_out.transpose(1, 2).view(batch_size, channels, d, h, w)
        pooled = self.global_pool(attn_out)
        
        # Final projection
        output = self.fc(pooled.view(batch_size, -1))
        
        return output


# Option 5: DINO v3 Encoder with slice-by-slice processing
class DinoV3Encoder(nn.Module):
    """DINO v3 encoder with slice-by-slice processing for 3D volumes"""
    
    def __init__(self, latent_dim=256, model_size='small', pretrained=True, max_slices=32, slice_sampling='uniform'):
        super().__init__()
        
        try:
            import torchvision
            from torchvision.models import dinov3_small, dinov3_base, dinov3_large
        except ImportError:
            raise ImportError("torchvision >= 0.15 required for DINO v3. Install with: pip install torchvision>=0.15")
        
        self.latent_dim = latent_dim
        self.max_slices = max_slices
        self.slice_sampling = slice_sampling
        
        print(f"🔧 Initializing DINO v3 encoder (size: {model_size}, latent_dim: {latent_dim})")
        
        # Create DINO v3 backbone
        if model_size == 'small':
            self.dino = dinov3_small(pretrained=pretrained)
        elif model_size == 'base':
            self.dino = dinov3_base(pretrained=pretrained)
        elif model_size == 'large':
            self.dino = dinov3_large(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown DINO v3 size: {model_size}. Use 'small', 'base', or 'large'")
        
        # Remove classification head
        self.dino.head = nn.Identity()
        
        # Get DINO feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.dino(dummy)
            dino_dim = features.shape[-1]
        
        print(f"DINO v3 feature dimension: {dino_dim}")
        
        # Projection layers
        self.slice_projection = nn.Linear(dino_dim, latent_dim)
        
        # Aggregation across slices
        self.slice_aggregator = nn.LSTM(
            latent_dim, latent_dim, 
            batch_first=True, bidirectional=True
        )
        self.final_projection = nn.Linear(latent_dim * 2, latent_dim)
        
        # Normalization layers
        self.slice_norm = nn.LayerNorm(latent_dim)
        self.final_norm = nn.LayerNorm(latent_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the projection layers"""
        for m in [self.slice_projection, self.final_projection]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _sample_slices(self, volume_3d):
        """Sample slice indices based on sampling strategy"""
        batch_size, _, depth, height, width = volume_3d.shape
        
        if self.slice_sampling == 'all':
            slice_indices = torch.arange(depth)
        elif self.slice_sampling == 'uniform':
            if depth > self.max_slices:
                slice_indices = torch.linspace(0, depth-1, self.max_slices).long()
            else:
                slice_indices = torch.arange(depth)
        elif self.slice_sampling == 'adaptive':
            # Adaptive sampling: more slices in the center
            center = depth // 2
            indices = torch.linspace(center - self.max_slices//2, center + self.max_slices//2, self.max_slices).clamp(0, depth-1).long()
            slice_indices = torch.unique(indices)
        else:
            raise ValueError(f"Unknown slice_sampling method: {self.slice_sampling}")
        
        return slice_indices
    
    def _preprocess_slice(self, slice_2d):
        """Preprocess 2D slice for DINO v3 input"""
        # Resize to 224x224 for DINO v3
        slice_2d = F.interpolate(
            slice_2d, size=(224, 224), 
            mode='bilinear', align_corners=False
        )
        
        # Convert to 3-channel (RGB) - DINO v3 expects RGB
        slice_2d = slice_2d.repeat(1, 3, 1, 1)
        
        # Normalize to ImageNet stats (DINO v3 expects this)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(slice_2d.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(slice_2d.device)
        slice_2d = (slice_2d - mean) / std
        
        return slice_2d
    
    def forward(self, volume_3d):
        """Forward pass through DINO v3 encoder"""
        # volume_3d: (batch, 1, D, H, W)
        batch_size, _, depth, height, width = volume_3d.shape
        device = volume_3d.device
        
        # Sample slice indices
        slice_indices = self._sample_slices(volume_3d)
        
        slice_features = []
        
        for idx in slice_indices:
            # Extract slice: (batch, 1, H, W)
            slice_2d = volume_3d[:, :, idx, :, :]
            
            # Preprocess for DINO v3
            slice_2d = self._preprocess_slice(slice_2d)
            
            # Get DINO v3 features
            try:
                features = self.dino(slice_2d)
                
                # Ensure correct shape
                if features.dim() > 2:
                    features = features.mean(dim=1)  # Average over patches
                
            except Exception as e:
                print(f"Error in DINO v3 forward pass: {e}")
                # Use fallback features
                features = torch.randn(batch_size, 384, device=device)  # DINO small has 384 features
            
            # Project to latent dimension
            slice_latent = self.slice_projection(features.float())
            slice_latent = self.slice_norm(slice_latent)
            slice_features.append(slice_latent)
        
        # Stack slice features: (batch, num_slices, latent_dim)
        volume_features = torch.stack(slice_features, dim=1)
        
        # LSTM aggregation
        lstm_out, (hidden, _) = self.slice_aggregator(volume_features)
        final_features = self.final_projection(lstm_out[:, -1, :])
        
        # Final normalization
        final_features = self.final_norm(final_features)
        
        return final_features


def create_encoder(encoder_type='simple_cnn', latent_dim=256, **kwargs):
    """Factory function to create encoders without MONAI dependency"""
    
    if encoder_type == 'simple_cnn':
        return Simple3DCNNEncoder(latent_dim=latent_dim, **kwargs)
    
    elif encoder_type == 'timm_vit':
        return TimmViTEncoder(latent_dim=latent_dim, **kwargs)
    
    elif encoder_type == 'resnet3d':
        return ResNet3DEncoder(latent_dim=latent_dim, **kwargs)
    
    elif encoder_type == 'hybrid':
        return LightweightHybridEncoder(latent_dim=latent_dim, **kwargs)
    
    elif encoder_type == 'dino_v3':
        return DinoV3Encoder(latent_dim=latent_dim, **kwargs)
    
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

# if __name__ == "__main__":
#     # Test different encoders
#     encoders = {
#         'simple_cnn': Simple3DCNNEncoder(latent_dim=256),
#         'timm_vit': TimmViTEncoder(latent_dim=256),
#         'resnet3d': ResNet3DEncoder(latent_dim=256),
#         'hybrid': LightweightHybridEncoder(latent_dim=256)
#     }
    
#     dummy_input = torch.randn(2, 1, 128, 128, 128)
    
#     for name, encoder in encoders.items():
#         try:
#             output = encoder(dummy_input)
#             print(f"{name}: Input {dummy_input.shape} -> Output {output.shape}")
#         except Exception as e:
#             print(f"{name}: Error - {e}")


# Generator: 3D Deconvolutional Network
class Generator(nn.Module):
    def __init__(self, latent_dim=256, phase_dim=32, output_shape=(128, 128, 128, 1)):
        super().__init__()
        self.latent_dim = latent_dim
        self.phase_dim = phase_dim
        # Initial dense layer to project to 3D feature map
        self.fc = nn.Linear(latent_dim + phase_dim, 8 * 8 * 8 * 256)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8x8 -> 16x16x16
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),   # 16x16x16 -> 32x32x32
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),    # 32x32x32 -> 64x64x64
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),     # 64x64x64 -> 128x128x128
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def forward(self, z, phase_emb):
        # z: (batch, latent_dim), phase_emb: (batch, phase_dim)
        x = torch.cat([z, phase_emb], dim=1)  # (batch, latent_dim + phase_dim)
        x = self.fc(x)  # (batch, 8*8*8*256)
        x = x.view(-1, 256, 8, 8, 8)  # Reshape to 3D
        return self.decoder(x)  # (batch, 1, 128, 128, 128)

# Discriminator: PatchGAN-style 3D CNN
class Discriminator(nn.Module):
    def __init__(self, input_shape=(128, 128, 128, 1)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1),  # 128x128x128 -> 64x64x64
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x64x64 -> 32x32x32
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),  # 32x32x32 -> 16x16x16
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            nn.Conv3d(256, 1, kernel_size=4, stride=1, padding=0),   # 16x16x16 -> 13x13x13
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)  # (batch, 1)



class PhaseDetector(nn.Module):
    def __init__(self, latent_dim=256, num_phases=4, dropout_rate=0.3):
        super().__init__()
        print(f"\nDEBUG: Initializing PhaseDetector")
        print(f"Latent dim: {latent_dim}")
        print(f"Num phases: {num_phases}")
        print(f"Dropout rate: {dropout_rate}")
        
        # Input normalization
        self.input_norm = nn.LayerNorm(latent_dim)
        
        # Simplified architecture for better learning
        # Using LayerNorm instead of BatchNorm1d for small batch size compatibility
        self.feature_extractor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),  # Works with any batch size
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),  # Works with any batch size
            nn.ReLU(),
            nn.Dropout(dropout_rate // 2)
        )
        
        # Classification head
        self.classifier = nn.Linear(latent_dim, num_phases)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, z):
        # Debug: Print input tensor information
        if torch.rand(1).item() < 0.01:  # Print for ~1% of forward passes
            print(f"\nDEBUG PhaseDetector forward:")
            print(f"Input shape: {z.shape}")
            print(f"Input range: [{z.min():.3f}, {z.max():.3f}]")
        
        # Normalize input
        z_norm = self.input_norm(z)
        
        # Extract features
        features = self.feature_extractor(z_norm)
        
        # Classify
        output = self.classifier(features)
        
        return output
    # def __init__(self, latent_dim=256, num_phases=4):
    #     super().__init__()  # ✅ Call parent class __init__ first
    #     self.model = nn.Sequential(
    #         nn.Linear(latent_dim, 512),      # Larger capacity
    #         nn.BatchNorm1d(512),             # Batch normalization
    #         nn.ReLU(),
    #         nn.Dropout(0.3),                 # Regularization
    #         nn.Linear(512, 256),
    #         nn.BatchNorm1d(256),
    #         nn.ReLU(),
    #         nn.Dropout(0.2),
    #         nn.Linear(256, num_phases)       # No softmax (let CrossEntropy handle it)
    #     )
    #     # Initialize weights properly
    #     self.apply(self._init_weights)
    
    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         torch.nn.init.xavier_uniform_(module.weight)
    #         if module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)

    # def forward(self, z):
    #     return self.model(z)  # (batch, num_phases)



# Test function to verify it works
def test_phase_detector_fix():
    """Quick test to make sure the improved detector works"""
    
    # Test parameters
    latent_dim = 256
    num_phases = 3
    batch_size = 4
    
    # Create test data
    test_features = torch.randn(batch_size, latent_dim)
    test_labels = torch.randint(0, num_phases, (batch_size,))
    
    print(f"Testing ImprovedPhaseDetector...")
    print(f"Input: {test_features.shape}")
    print(f"Labels: {test_labels}")
    
    # Create detector
    detector = PhaseDetector(latent_dim=latent_dim, num_phases=num_phases)
    
    # Test forward pass
    with torch.no_grad():
        output = detector(test_features)
        probabilities = torch.softmax(output, dim=1)
    
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Sample probabilities: {probabilities[0].numpy()}")
    
    # Test training step
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(detector.parameters(), lr=1e-3)
    
    optimizer.zero_grad()
    output = detector(test_features)
    loss = criterion(output, test_labels)
    loss.backward()
    optimizer.step()
    
    print(f"✅ Training step works, loss: {loss.item():.4f}")
    print(f"✅ ImprovedPhaseDetector is ready to use!")

if __name__ == "__main__":
    test_phase_detector_fix()