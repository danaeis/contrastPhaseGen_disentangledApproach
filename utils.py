import torch
import torch.nn as nn
import numpy as np
import math
import nibabel as nib

class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for domain adaptation and disentanglement
    
    During forward pass: output = input
    During backward pass: gradient = -lambda * input_gradient
    """
    
    @staticmethod
    def forward(ctx, x, lambda_):
        """
        Forward pass - identity function
        
        Args:
            x: Input tensor
            lambda_: Reversal strength coefficient
        """
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass - reverse gradients
        
        Args:
            grad_output: Gradients from upstream
            
        Returns:
            Reversed gradients scaled by lambda
        """
        output = grad_output.neg() * ctx.lambda_
        return output, None


def get_phase_embedding(phase, dim=32, device='cuda'):
    """
    Create sinusoidal position embedding for contrast phase
    
    Args:
        phase (int): Phase index (0=arterial, 1=venous, 2=delayed, 3=non-contrast)
        dim (int): Embedding dimension
        device (str or torch.device): Target device
        
    Returns:
        torch.Tensor: Phase embedding of shape (dim,)
    """
    # Create position encoding similar to transformer positional encoding
    position = phase
    div_term = torch.exp(torch.arange(0, dim, 2).float() * 
                        -(math.log(10000.0) / dim))
    
    embedding = torch.zeros(dim)
    embedding[0::2] = torch.sin(position * div_term)
    embedding[1::2] = torch.cos(position * div_term)
    
    return embedding.to(device)


def get_learnable_phase_embedding(num_phases=4, dim=32):
    """
    Create learnable phase embeddings
    
    Args:
        num_phases (int): Number of contrast phases
        dim (int): Embedding dimension
        
    Returns:
        nn.Embedding: Learnable phase embedding layer
    """
    return nn.Embedding(num_phases, dim)


class AdaptiveLossWeights:
    """
    Adaptive loss weighting for multi-objective training
    Dynamically adjusts loss weights based on training progress
    """
    
    def __init__(self, initial_weights=None, adaptation_rate=0.1):
        """
        Args:
            initial_weights (dict): Initial loss weights
            adaptation_rate (float): Rate of weight adaptation
        """
        self.weights = initial_weights or {
            'reconstruction': 10.0,
            'adversarial': 1.0,
            'phase': 1.0,
            'disentanglement': 0.1
        }
        self.adaptation_rate = adaptation_rate
        self.loss_history = {key: [] for key in self.weights.keys()}
    
    def update_weights(self, current_losses):
        """
        Update loss weights based on current loss values
        
        Args:
            current_losses (dict): Current loss values
        """
        # Store current losses
        for key, value in current_losses.items():
            if key in self.loss_history:
                self.loss_history[key].append(value)
        
        # Adapt weights based on loss trends
        if len(self.loss_history['reconstruction']) > 10:
            for loss_name in self.weights.keys():
                if loss_name in current_losses:
                    recent_losses = self.loss_history[loss_name][-10:]
                    loss_trend = np.mean(recent_losses[-5:]) / np.mean(recent_losses[:5])
                    
                    # If loss is increasing, increase weight
                    if loss_trend > 1.1:
                        self.weights[loss_name] *= (1 + self.adaptation_rate)
                    # If loss is decreasing rapidly, decrease weight
                    elif loss_trend < 0.9:
                        self.weights[loss_name] *= (1 - self.adaptation_rate)
    
    def get_weights(self):
        """Get current loss weights"""
        return self.weights


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting
    """
    
    def __init__(self, patience=10, min_delta=1e-6, restore_best_weights=True):
        """
        Args:
            patience (int): Number of epochs to wait for improvement
            min_delta (float): Minimum change to qualify as improvement
            restore_best_weights (bool): Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model=None):
        """
        Check if training should stop
        
        Args:
            val_loss (float): Current validation loss
            model (nn.Module): Model to save weights from
            
        Returns:
            bool: True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            
        return self.counter >= self.patience
    
    def restore_weights(self, model):
        """Restore best weights to model"""
        if self.best_weights is not None:
            model.load_state_dict({k: v.to(next(model.parameters()).device) 
                                 for k, v in self.best_weights.items()})


class LearningRateScheduler:
    """
    Custom learning rate scheduler for sequential training phases
    """
    
    def __init__(self, optimizer, phase_configs):
        """
        Args:
            optimizer: PyTorch optimizer
            phase_configs (dict): Configuration for each training phase
        """
        self.optimizer = optimizer
        self.phase_configs = phase_configs
        self.current_phase = None
        
    def set_phase(self, phase_name, epoch=0):
        """
        Set current training phase and adjust learning rate
        
        Args:
            phase_name (str): Name of training phase
            epoch (int): Current epoch within phase
        """
        if phase_name != self.current_phase:
            self.current_phase = phase_name
            config = self.phase_configs.get(phase_name, {})
            
            # Set base learning rate
            base_lr = config.get('lr', 1e-4)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = base_lr
        
        # Apply any epoch-based scheduling within phase
        config = self.phase_configs.get(phase_name, {})
        if 'schedule' in config:
            schedule_type = config['schedule']['type']
            
            if schedule_type == 'cosine':
                total_epochs = config['schedule']['total_epochs']
                min_lr = config['schedule'].get('min_lr', 1e-6)
                base_lr = config.get('lr', 1e-4)
                
                lr = min_lr + (base_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs)) / 2
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                    
            elif schedule_type == 'step':
                step_size = config['schedule']['step_size']
                gamma = config['schedule']['gamma']
                base_lr = config.get('lr', 1e-4)
                
                lr = base_lr * (gamma ** (epoch // step_size))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """
    Compute gradient penalty for WGAN-GP
    
    Args:
        discriminator: Discriminator network
        real_samples: Real samples
        fake_samples: Generated samples
        device: Device to run on
        
    Returns:
        Gradient penalty loss
    """
    batch_size = real_samples.shape[0]
    alpha = torch.rand(batch_size, 1, 1, 1, 1).to(device)
    
    # Interpolate between real and fake samples
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated = interpolated.requires_grad_(True)
    
    # Get discriminator output
    d_interpolated = discriminator(interpolated)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Compute gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss for improved GAN training
    """
    
    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator
        
    def forward(self, real_samples, fake_samples):
        """
        Compute feature matching loss
        
        Args:
            real_samples: Real samples
            fake_samples: Generated samples
            
        Returns:
            Feature matching loss
        """
        # Get intermediate features from discriminator
        real_features = []
        fake_features = []
        
        # Hook to capture intermediate features
        def hook_fn(module, input, output):
            return output
        
        # Register hooks on discriminator layers
        hooks = []
        for layer in self.discriminator.modules():
            if isinstance(layer, (nn.Conv3d, nn.ConvTranspose3d)):
                hooks.append(layer.register_forward_hook(hook_fn))
        
        # Forward pass
        with torch.no_grad():
            _ = self.discriminator(real_samples)
            real_features = [h.stored for h in hooks]
        
        _ = self.discriminator(fake_samples)
        fake_features = [h.stored for h in hooks]
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute L2 loss between features
        feature_loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            feature_loss += nn.functional.mse_loss(fake_feat, real_feat)
        
        return feature_loss / len(real_features)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pretrained features for better image quality
    """
    
    def __init__(self, feature_extractor=None, layers=None):
        super().__init__()
        
        if feature_extractor is None:
            # Use a simple 3D CNN as feature extractor
            self.feature_extractor = nn.Sequential(
                nn.Conv3d(1, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Conv3d(32, 64, 3, 1, 1),
                nn.ReLU(),
                nn.Conv3d(64, 128, 3, 1, 1),
                nn.ReLU()
            )
        else:
            self.feature_extractor = feature_extractor
        
        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.layers = layers or ['0', '2', '4']  # Layer indices to extract features from
        
    def forward(self, pred, target):
        """
        Compute perceptual loss
        
        Args:
            pred: Predicted images
            target: Target images
            
        Returns:
            Perceptual loss
        """
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += nn.functional.mse_loss(pred_feat, target_feat)
            
        return loss
    
    def extract_features(self, x):
        """Extract features from specified layers"""
        features = []
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            if str(i) in self.layers:
                features.append(x)
        return features


class MetricsTracker:
    """
    Utility class to track and visualize training metrics
    """
    
    def __init__(self, metrics_names):
        """
        Args:
            metrics_names (list): List of metric names to track
        """
        self.metrics = {name: [] for name in metrics_names}
        self.epoch_metrics = {name: 0.0 for name in metrics_names}
        self.batch_count = 0
        
    def update(self, **kwargs):
        """Update metrics with new values"""
        for name, value in kwargs.items():
            if name in self.epoch_metrics:
                self.epoch_metrics[name] += value
        self.batch_count += 1
        
    def finalize_epoch(self):
        """Finalize epoch by computing averages and storing"""
        for name in self.metrics.keys():
            if self.batch_count > 0:
                avg_value = self.epoch_metrics[name] / self.batch_count
                self.metrics[name].append(avg_value)
                self.epoch_metrics[name] = 0.0
        self.batch_count = 0
        
    def get_latest(self, metric_name):
        """Get latest value for a metric"""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return self.metrics[metric_name][-1]
        return 0.0
        
    def get_history(self, metric_name):
        """Get full history for a metric"""
        return self.metrics.get(metric_name, [])
        
    def save_to_file(self, filepath):
        """Save metrics to CSV file"""
        import csv
        
        # Get maximum length
        max_len = max(len(values) for values in self.metrics.values()) if self.metrics else 0
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['epoch'] + list(self.metrics.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i in range(max_len):
                row = {'epoch': i + 1}
                for name, values in self.metrics.items():
                    row[name] = values[i] if i < len(values) else ''
                writer.writerow(row)


def calculate_fid_score(real_features, generated_features):
    """
    Calculate Fréchet Inception Distance (FID) score
    
    Args:
        real_features: Features from real images
        generated_features: Features from generated images
        
    Returns:
        FID score
    """
    # Calculate mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(generated_features, axis=0)
    
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(generated_features, rowvar=False)
    
    # Calculate FID
    diff = mu_real - mu_gen
    covmean = scipy.linalg.sqrtm(sigma_real.dot(sigma_gen))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid


def calculate_ssim_3d(img1, img2, data_range=1.0):
    """
    Calculate 3D SSIM between two volumes
    
    Args:
        img1: First volume
        img2: Second volume
        data_range: Data range of images
        
    Returns:
        SSIM score
    """
    try:
        from skimage.metrics import structural_similarity
        
        # Convert to numpy if needed
        if torch.is_tensor(img1):
            img1 = img1.cpu().numpy()
        if torch.is_tensor(img2):
            img2 = img2.cpu().numpy()
            
        # Remove batch and channel dimensions if present
        if img1.ndim == 5:  # (B, C, D, H, W)
            img1 = img1[0, 0]
            img2 = img2[0, 0]
        elif img1.ndim == 4:  # (C, D, H, W)
            img1 = img1[0]
            img2 = img2[0]
            
        return structural_similarity(img1, img2, data_range=data_range)
        
    except ImportError:
        print("Warning: scikit-image not available, using MSE-based similarity")
        mse = np.mean((img1 - img2) ** 2)
        return 1 / (1 + mse)  # Convert MSE to similarity score


def calculate_psnr_3d(img1, img2, data_range=1.0):
    """
    Calculate 3D PSNR between two volumes
    
    Args:
        img1: First volume
        img2: Second volume
        data_range: Data range of images
        
    Returns:
        PSNR score
    """
    # Convert to numpy if needed
    if torch.is_tensor(img1):
        img1 = img1.cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.cpu().numpy()
        
    # Remove batch and channel dimensions if present
    if img1.ndim == 5:  # (B, C, D, H, W)
        img1 = img1[0, 0]
        img2 = img2[0, 0]
    elif img1.ndim == 4:  # (C, D, H, W)
        img1 = img1[0]
        img2 = img2[0]
        
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
        
    psnr = 20 * np.log10(data_range / np.sqrt(mse))
    return psnr


class WarmupScheduler:
    """
    Learning rate warmup scheduler
    """
    
    def __init__(self, optimizer, warmup_epochs, target_lr):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            target_lr: Target learning rate after warmup
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch):
        """Update learning rate based on epoch"""
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = (epoch + 1) / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * lr_scale
        else:
            # Set to target learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.target_lr


def visualize_3d_volume(volume, save_path=None, num_slices=9):
    """
    Create a visualization of 3D volume by showing multiple slices
    
    Args:
        volume: 3D volume tensor or numpy array
        save_path: Path to save visualization
        num_slices: Number of slices to show
    """
    try:
        import matplotlib.pyplot as plt
        
        # Convert to numpy if needed
        if torch.is_tensor(volume):
            volume = volume.cpu().numpy()
            
        # Remove batch and channel dimensions if present
        if volume.ndim == 5:  # (B, C, D, H, W)
            volume = volume[0, 0]
        elif volume.ndim == 4:  # (C, D, H, W)
            volume = volume[0]
            
        depth = volume.shape[0]
        slice_indices = np.linspace(0, depth-1, num_slices, dtype=int)
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()
        
        for i, slice_idx in enumerate(slice_indices):
            axes[i].imshow(volume[slice_idx], cmap='gray')
            axes[i].set_title(f'Slice {slice_idx}')
            axes[i].axis('off')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    except ImportError:
        print("Warning: matplotlib not available, skipping visualization")


def print_model_summary(model, input_shape):
    """
    Print model summary including parameter count
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (without batch dimension)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Test forward pass
    try:
        dummy_input = torch.randn(1, *input_shape)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"  Input shape: {input_shape}")
        print(f"  Output shape: {output.shape[1:]}")  # Remove batch dimension
    except Exception as e:
        print(f"  Could not determine output shape: {e}")


# Phase-specific utilities
def create_phase_schedule(total_epochs, phase_ratios):
    """
    Create epoch schedule for different training phases
    
    Args:
        total_epochs: Total number of training epochs
        phase_ratios: Dictionary with phase names and their ratios
        
    Returns:
        Dictionary with phase names and epoch ranges
    """
    schedule = {}
    current_epoch = 0
    
    for phase_name, ratio in phase_ratios.items():
        phase_epochs = int(total_epochs * ratio)
        schedule[phase_name] = {
            'start': current_epoch,
            'end': current_epoch + phase_epochs,
            'epochs': phase_epochs
        }
        current_epoch += phase_epochs
        
    return schedule


def save_training_config(config_dict, save_path):
    """
    Save training configuration to JSON file
    
    Args:
        config_dict: Configuration dictionary
        save_path: Path to save configuration
    """
    import json
    
    # Convert any non-serializable objects to strings
    serializable_config = {}
    for key, value in config_dict.items():
        if isinstance(value, (torch.device, type)):
            serializable_config[key] = str(value)
        else:
            serializable_config[key] = value
            
    with open(save_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)
        
    print(f"Training configuration saved to: {save_path}")


def load_training_config(config_path):
    """
    Load training configuration from JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    import json
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    return config


# Save volume as NIfTI file
def save_volume(volume, path):
    """Save a PyTorch tensor volume as a NIfTI file."""
    volume_np = volume.cpu().numpy().squeeze()
    nii_img = nib.Nifti1Image(volume_np, affine=np.eye(4))
    nib.save(nii_img, path)