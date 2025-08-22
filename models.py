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
            nn.Sigmoid()  # PatchGAN output
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)  # (batch, 1)


class PhaseDetector(nn.Module):
    def __init__(self, latent_dim=256, num_phases=4, dropout_rate=0.3):
        super().__init__()
        
        # Input normalization
        self.input_norm = nn.LayerNorm(latent_dim)
        
        # Three processing blocks
        self.block1 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.block3 = nn.Sequential(
            nn.Linear(latent_dim // 2, latent_dim // 4),
            nn.LayerNorm(latent_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate // 2)
        )
        
        # Three classification heads
        self.head1 = nn.Linear(latent_dim // 4, num_phases)
        self.head2 = nn.Linear(latent_dim // 2, num_phases)  
        self.head3 = nn.Linear(latent_dim, num_phases)
        
        # Final fusion
        self.fusion = nn.Linear(num_phases * 3, num_phases)
        
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
        # Normalize input
        z_norm = self.input_norm(z)
        
        # Process through blocks
        x1 = self.block1(z_norm)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        
        # Three predictions
        pred1 = self.head1(x3)    # From final block
        pred2 = self.head2(x2)    # From middle block
        pred3 = self.head3(z_norm) # Direct from input
        
        # Combine predictions
        combined = torch.cat([pred1, pred2, pred3], dim=1)
        final_output = self.fusion(combined)
        
        return final_output
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