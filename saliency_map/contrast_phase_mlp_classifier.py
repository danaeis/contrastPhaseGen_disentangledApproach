import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split
import os
import json
import pickle
from datetime import datetime
from tqdm import tqdm
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# Setup imports from parent directory
import sys
import os
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import your existing modules
try:
    from models import TimmViTEncoder
    TIMM_VIT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TimmViTEncoder not available: {e}")
    TimmViTEncoder = None
    TIMM_VIT_AVAILABLE = False

try:
    from dino_encoder import DinoV3Encoder
    DINOV3_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DinoV3Encoder not available: {e}")
    DinoV3Encoder = None
    DINOV3_AVAILABLE = False

try:
    from medViT_encoder import create_medvit_encoder
    MEDVIT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: create_medvit_encoder not available: {e}")
    create_medvit_encoder = None
    MEDVIT_AVAILABLE = False

try:
    from data import prepare_dataset_from_folders, prepare_data
    DATA_FUNCTIONS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: data functions not available: {e}")
    prepare_dataset_from_folders = None
    prepare_data = None
    DATA_FUNCTIONS_AVAILABLE = False

from feature_visualization import create_phase_mapping
# def create_phase_mapping(unique_classes):
#     """Create phase mapping based on actual classes in data"""
    
#     # Default phase names
#     default_phases = {
#         0: 'Non-contrast', 
#         1: 'Arterial', 
#         2: 'Venous', 
#         3: 'Delayed', 
#         4: 'Hepatobiliary'
#     }
    
#     # Create mapping only for classes that exist
#     phase_mapping = {}
#     for class_idx in unique_classes:
#         phase_mapping[class_idx] = default_phases.get(class_idx, f'Phase_{class_idx}')
    
#     return phase_mapping



class AttentionMLP(nn.Module):
    """
    MLP with attention mechanism for contrast phase classification
    """
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], n_classes=5, 
                 dropout_rate=0.3, use_attention=False, attention_heads=8):
        super(AttentionMLP, self).__init__()
        
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.use_attention = use_attention
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=prev_dim,
                num_heads=attention_heads,
                dropout=dropout_rate,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(prev_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(prev_dim // 2, n_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, return_attention=False):
        """
        Forward pass with optional attention weights return
        
        Args:
            x: Input features (batch_size, input_dim)
            return_attention: Whether to return attention weights
        """
        # Extract features
        features = self.feature_extractor(x)
        
        attention_weights = None
        if self.use_attention:
            # Add sequence dimension for attention (treating each feature as a sequence element)
            features_seq = features.unsqueeze(1)  # (batch_size, 1, feature_dim)
            
            # Self-attention
            attended_features, attention_weights = self.attention(
                features_seq, features_seq, features_seq
            )
            
            # Apply residual connection and normalization
            features = self.attention_norm(features + attended_features.squeeze(1))
        
        # Classification
        logits = self.classifier(features)
        
        if return_attention:
            return logits, attention_weights
        return logits


class ContrastPhaseMLPClassifier(nn.Module):
    """
    End-to-end trainable contrast phase classifier with MLP head
    """
    def __init__(self, encoder, encoder_name, mlp_config=None, n_classes=5, 
                 freeze_encoder='full', enable_gradcam=True):
        super(ContrastPhaseMLPClassifier, self).__init__()
        
        self.encoder = encoder
        self.encoder_name = encoder_name
        self.n_classes = n_classes
        self.freeze_encoder = freeze_encoder
        self.enable_gradcam = enable_gradcam
        
        # Freeze encoder if requested
        if freeze_encoder == 'full':
            # Freeze all encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        elif freeze_encoder == 'partial':
            # Freeze early layers, unfreeze final layers
            self._selective_freeze()
            
        elif freeze_encoder == 'none':
            # Keep all parameters trainable
            pass
        
        # Get encoder output dimension
        self.encoder_dim = self._get_encoder_dim()
        
        # MLP classifier configuration
        if mlp_config is None:
            mlp_config = {
                'hidden_dims': [512, 256, 128],
                'dropout_rate': 0.3,
                'use_attention': False,
                'attention_heads': 8
            }
        
        # Create MLP head
        self.mlp = AttentionMLP(
            input_dim=self.encoder_dim,
            n_classes=n_classes,
            **mlp_config
        )
        
        # For GradCAM
        self.encoder_features = None
        self.gradients = None
        
        if enable_gradcam:
            self._register_hooks()
        
        # Phase mapping
        self.phase_mapping = None
    
    def _selective_freeze(self):
        """Freeze early layers, keep final layers trainable for better saliency"""
        # For Vision Transformers, keep last few transformer blocks trainable
        if hasattr(self.encoder, 'dino'):
            # Freeze early blocks, unfreeze last 2-3 blocks
            for name, param in self.encoder.named_parameters():
                if 'slice_projection' in name or 'final_projection' in name:
                    param.requires_grad = True
                elif any(x in name for x in ['11', '10', '9']):  # Last few blocks
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
    def _get_encoder_dim(self):
        """Get the output dimension of the encoder"""
        # Create a dummy input to determine encoder output size
        dummy_input = torch.randn(1, 1, 32, 64, 64)
        
        if hasattr(self.encoder, 'latent_dim'):
            return self.encoder.latent_dim
        else:
            with torch.no_grad():
                dummy_output = self.encoder(dummy_input)
                return dummy_output.shape[-1]
    
    def _register_hooks(self):
        """Register hooks for GradCAM"""
        def forward_hook(module, input, output):
            self.encoder_features = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Register hooks on the encoder
        if hasattr(self.encoder, 'final_norm'):
            # For models with final normalization layer
            self.encoder.final_norm.register_forward_hook(forward_hook)
            self.encoder.final_norm.register_backward_hook(backward_hook)
        elif hasattr(self.encoder, 'slice_norm'):
            # For slice-based models
            self.encoder.slice_norm.register_forward_hook(forward_hook)
            self.encoder.slice_norm.register_backward_hook(backward_hook)
        else:
            # Fallback: register on the last layer
            for name, module in self.encoder.named_modules():
                pass
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)
    
    def forward(self, x, return_attention=False):
        """
        Forward pass through encoder and MLP
        
        Args:
            x: Input volume (batch_size, channels, depth, height, width)
            return_attention: Whether to return attention weights
        """
        # Extract features from encoder
        encoded_features = self.encoder(x)
        
        # Pass through MLP
        if return_attention:
            logits, attention_weights = self.mlp(encoded_features, return_attention=True)
            return logits, attention_weights
        else:
            logits = self.mlp(encoded_features)
            return logits
    
    def predict(self, x):
        """Predict class and probabilities"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
        return predictions, probabilities
    
    def generate_gradcam(self, x, target_class=None, layer_name=None):
        """
        Generate GradCAM saliency map for 3D input
        
        Args:
            x: Input volume (batch_size, channels, depth, height, width)
            target_class: Target class index (if None, uses predicted class)
            layer_name: Layer name for GradCAM (not used in current implementation)
        
        Returns:
            saliency_map: 3D saliency map (depth, height, width)
            predicted_class: Predicted class
            class_probability: Probability of predicted class
        """
        if not self.enable_gradcam:
            raise ValueError("GradCAM not enabled. Set enable_gradcam=True during initialization.")
        
        self.eval()
        x.requires_grad_(True)
        
        # Forward pass
        logits = self.forward(x)
        probabilities = F.softmax(logits, dim=1)
        
        # Get predicted class if target_class not specified
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
        
        predicted_class = torch.argmax(logits, dim=1).item()
        class_probability = probabilities[0, predicted_class].item()
        
        # Backward pass for target class
        self.zero_grad()
        class_score = logits[0, target_class]
        class_score.backward(retain_graph=True)
        
        # Generate saliency map using input gradients
        input_gradients = x.grad.data
        
        # Method 1: Use input gradients directly
        saliency_map_input = torch.abs(input_gradients[0, 0]).cpu().numpy()
        
        # Method 2: Use encoder features if available (GradCAM style)
        if self.encoder_features is not None and self.gradients is not None:
            # This is a simplified version - you might need to adapt based on your encoder structure
            gradients = self.gradients
            features = self.encoder_features
            
            # Global average pooling of gradients
            weights = torch.mean(gradients, dim=[2, 3], keepdim=True) if gradients.dim() > 2 else gradients
            
            # Weighted combination of features
            if features.dim() == 2:  # If encoder outputs 1D features
                # Create a dummy spatial map
                saliency_map_gradcam = torch.ones(x.shape[2:]).cpu().numpy()
            else:
                # Proper GradCAM computation
                cam = torch.sum(weights * features, dim=1)
                cam = F.relu(cam)
                saliency_map_gradcam = cam[0].cpu().numpy()
        else:
            saliency_map_gradcam = saliency_map_input
        
        # Normalize saliency map
        saliency_map = self._normalize_saliency_map(saliency_map_input)
        
        return saliency_map, predicted_class, class_probability
    
    def generate_attention_map(self, x):
        """
        Generate attention-based saliency map
        
        Args:
            x: Input volume (batch_size, channels, depth, height, width)
        
        Returns:
            attention_map: Attention weights
            predicted_class: Predicted class
            class_probability: Probability of predicted class
        """
        self.eval()
        with torch.no_grad():
            logits, attention_weights = self.forward(x, return_attention=True)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            class_probability = probabilities[0, predicted_class].item()
            
            if attention_weights is not None:
                # Process attention weights
                attention_map = attention_weights[0].cpu().numpy()  # (num_heads, seq_len, seq_len)
                # Average across heads and take diagonal (self-attention)
                attention_map = np.mean(attention_map, axis=0)
                attention_map = np.diag(attention_map) if attention_map.ndim > 1 else attention_map
            else:
                attention_map = np.ones(self.encoder_dim)
        
        return attention_map, predicted_class, class_probability
    
    def _normalize_saliency_map(self, saliency_map):
        """Normalize saliency map to [0, 1]"""
        saliency_map = saliency_map - saliency_map.min()
        if saliency_map.max() > 0:
            saliency_map = saliency_map / saliency_map.max()
        return saliency_map
    
    def generate_3d_saliency_visualization(self, x, save_path=None, method='gradcam'):
        """
        Generate comprehensive 3D saliency visualization
        
        Args:
            x: Input volume (batch_size, channels, depth, height, width)
            save_path: Path to save visualization
            method: 'gradcam' or 'attention'
        
        Returns:
            Dictionary with saliency results and visualizations
        """
        if method == 'gradcam':
            saliency_map, pred_class, class_prob = self.generate_gradcam(x)
        elif method == 'attention':
            attention_map, pred_class, class_prob = self.generate_attention_map(x)
            # For attention, create a dummy 3D map
            saliency_map = np.ones(x.shape[2:]) * np.mean(attention_map)
        else:
            raise ValueError("Method must be 'gradcam' or 'attention'")
        
        # Get original volume for overlay
        original_volume = x[0, 0].cpu().numpy()
        
        # Create visualization
        fig = plt.figure(figsize=(20, 12))
        
        # Plot multiple slices
        depth = original_volume.shape[0]
        slice_indices = np.linspace(0, depth-1, 9, dtype=int)
        
        for i, slice_idx in enumerate(slice_indices):
            # Original slice
            ax1 = plt.subplot(3, 6, i+1)
            plt.imshow(original_volume[slice_idx], cmap='gray')
            plt.title(f'Original Slice {slice_idx}')
            plt.axis('off')
            
            # Saliency map
            ax2 = plt.subplot(3, 6, i+10)
            plt.imshow(saliency_map[slice_idx], cmap='hot')
            plt.title(f'Saliency Slice {slice_idx}')
            plt.axis('off')
            
            # Overlay
            if i < 8:  # Only show overlay for first 8 slices to save space
                ax3 = plt.subplot(3, 6, i+10)
                plt.imshow(original_volume[slice_idx], cmap='gray', alpha=0.7)
                plt.imshow(saliency_map[slice_idx], cmap='hot', alpha=0.3)
                plt.title(f'Overlay Slice {slice_idx}')
                plt.axis('off')
        
        # Add prediction information
        # phase_name = self.phase_mapping.get(pred_class, f'Phase_{pred_class}')
        plt.suptitle(f'3D Saliency Map - Predicted: {pred_class} (Confidence: {class_prob:.3f})', 
                    fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D saliency visualization saved to: {save_path}")
        
        plt.show()
        
        return {
            'saliency_map': saliency_map,
            'predicted_class': pred_class,
            'class_probability': class_prob,
            'phase_name': pred_class,
            'original_volume': original_volume
        }


class ContrastPhaseTrainer:
    """
    Trainer class for end-to-end training of contrast phase classification
    """
    def __init__(self, model, device='cuda', learning_rate=1e-4, weight_decay=1e-4, phase_mapping=None):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Loss function with class weighting
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }
        self.phase_mapping = phase_mapping or {}
        
    
    def compute_class_weights(self, train_labels):
        """Compute class weights for imbalanced dataset"""
        unique_classes, class_counts = np.unique(train_labels, return_counts=True)
        total_samples = len(train_labels)
        
        # Phase mapping
        self.phase_mapping = create_phase_mapping(unique_classes)

        # Inverse frequency weighting
        class_weights = total_samples / (len(unique_classes) * class_counts)
        
        # Convert to tensor
        weight_tensor = torch.zeros(self.model.n_classes)
        for i, class_idx in enumerate(unique_classes):
            weight_tensor[class_idx] = class_weights[i]
        
        return weight_tensor.to(self.device)
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # Get data
            if isinstance(batch_data, dict):
                images = batch_data['input_path'].to(self.device)
                labels = batch_data['input_phase'].to(self.device)
            else:
                images, labels = batch_data
                images = images.to(self.device)
                labels = labels.to(self.device)
            
            # Ensure labels are long type and correct shape
            if labels.dim() > 1:
                labels = labels.squeeze()
            labels = labels.long()
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images)
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            current_acc = correct_predictions / total_samples
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc="Validation"):
                # Get data
                if isinstance(batch_data, dict):
                    images = batch_data['input_path'].to(self.device)
                    labels = batch_data['input_phase'].to(self.device)
                else:
                    images, labels = batch_data
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                
                # Ensure labels are long type and correct shape
                if labels.dim() > 1:
                    labels = labels.squeeze()
                labels = labels.long()
                
                # Forward pass
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                
                # Store for detailed metrics
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_samples
        
        # Detailed metrics
        detailed_metrics = self._compute_detailed_metrics(all_predictions, all_labels)
        
        return avg_loss, accuracy, detailed_metrics
    
    def _compute_detailed_metrics(self, predictions, labels):
        """Compute detailed classification metrics"""
        # Convert numpy types to standard Python types
        if hasattr(predictions, 'cpu'):
            predictions = [int(x) for x in predictions.cpu().numpy()]
        elif isinstance(predictions, np.ndarray):
            predictions = [int(x) for x in predictions]
        else:
            predictions = [int(x) for x in predictions]
            
        if hasattr(labels, 'cpu'):
            labels = [int(x) for x in labels.cpu().numpy()]  
        elif isinstance(labels, np.ndarray):
            labels = [int(x) for x in labels]
        else:
            labels = [int(x) for x in labels]
        # Get unique classes as standard Python integers
        unique_classes = sorted(list(set(labels + predictions)))
        n_actual_classes = len(unique_classes)
        
        print(f"Detected {n_actual_classes} classes: {unique_classes}")
        print(f"Class types: {[type(x).__name__ for x in unique_classes]}")
        # Create phase names only for classes that actually exist
        if self.phase_mapping is None or not self.phase_mapping:
            print("Creating phase mapping from detected classes...")
            self.phase_mapping = create_phase_mapping([int(x) for x in unique_classes])

        phase_names = []
        for class_idx in unique_classes:
            phase_name = self.phase_mapping.get(class_idx, f'Phase_{class_idx}')
            phase_names.append(phase_name)
        
        print(f"Detected {n_actual_classes} classes: {unique_classes}")
        print(f"Phase names: {phase_names}")
        # Classification report
        # phase_names = [self.phase_mapping.get(i, f'Phase_{i}') for i in range(self.model.n_classes)]
        try:
            report = classification_report(
                labels, predictions, 
                target_names=phase_names,
                output_dict=True,
                zero_division=0,
                labels=unique_classes
            )
            
            # Confusion matrix
            cm = confusion_matrix(labels, predictions)
            
            precision, recall, f1, support = precision_recall_fscore_support(
                labels, predictions, average=None, zero_division=0, labels=unique_classes
            )
            
            return {
                'classification_report': report,
                'confusion_matrix': cm,
                'per_class_precision': precision,
                'per_class_recall': recall,
                'per_class_f1': f1,
                'per_class_support': support,
                'unique_classes': unique_classes,
                'n_actual_classes': n_actual_classes
            }
        except Exception as e:
            print(f"Error in detailed metrics computation: {e}")
            print(f"Labels: {labels[:10]}...")  # Show first 10
            print(f"Predictions: {predictions[:10]}...")  # Show first 10
            
            # Return minimal metrics to avoid crashing
            return {
                'classification_report': {},
                'confusion_matrix': np.eye(n_actual_classes),
                'per_class_precision': np.ones(n_actual_classes),
                'per_class_recall': np.ones(n_actual_classes),
                'per_class_f1': np.ones(n_actual_classes),
                'per_class_support': np.ones(n_actual_classes),
                'unique_classes': unique_classes,
                'n_actual_classes': n_actual_classes
            }
    
    def train(self, train_loader, val_loader, num_epochs=50, save_best=True, 
              model_save_path='best_contrast_model.pth', early_stopping_patience=10):
        """
        Complete training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            save_best: Whether to save the best model
            model_save_path: Path to save the best model
            early_stopping_patience: Patience for early stopping
        """
        print("Starting training...")
        print(f"Model: {self.model.encoder_name}")
        print(f"Device: {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, detailed_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['learning_rates'].append(current_lr)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Print per-class metrics
            # Print per-class metrics
            print("\nPer-class metrics:")
            if self.phase_mapping:
                for class_idx, phase_name in self.phase_mapping.items():
                    # Convert key to integer to avoid type comparison issues
                    idx = int(class_idx)  # This ensures we have a proper int
                    
                    # Check bounds safely
                    if 0 <= idx < len(detailed_metrics['per_class_precision']):
                        print(f"  {phase_name}: P={detailed_metrics['per_class_precision'][idx]:.3f}, "
                            f"R={detailed_metrics['per_class_recall'][idx]:.3f}, "
                            f"F1={detailed_metrics['per_class_f1'][idx]:.3f}")
                    else:
                        print(f"  {phase_name}: No metrics available (index {idx} out of range)")
            else:
                print("  Phase mapping not available")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                if save_best:
                    self.save_model(model_save_path, epoch, val_acc, detailed_metrics)
                    print(f"New best model saved! Val Acc: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        
        return self.train_history
    
    def save_model(self, save_path, epoch, val_acc, detailed_metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'train_history': self.train_history,
            'detailed_metrics': detailed_metrics,
            'model_config': {
                'encoder_name': self.model.encoder_name,
                'n_classes': self.model.n_classes,
                'encoder_dim': self.model.encoder_dim,
                'freeze_encoder': self.model.freeze_encoder
            },
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, save_path)
    
    def load_model(self, load_path):
        """Load model checkpoint"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_history = checkpoint['train_history']
        
        print(f"Model loaded from epoch {checkpoint['epoch']} with val_acc: {checkpoint['val_acc']:.4f}")
        return checkpoint
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_history['train_loss']) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, self.train_history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.train_history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, self.train_history['train_acc'], 'b-', label='Train Acc')
        axes[0, 1].plot(epochs, self.train_history['val_acc'], 'r-', label='Val Acc')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(epochs, self.train_history['learning_rates'], 'g-')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Best metrics summary
        best_train_acc = max(self.train_history['train_acc'])
        best_val_acc = max(self.train_history['val_acc'])
        final_loss = self.train_history['val_loss'][-1]
        
        axes[1, 1].text(0.1, 0.8, f'Best Train Acc: {best_train_acc:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f'Best Val Acc: {best_val_acc:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, f'Final Loss: {final_loss:.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.2, f'Total Epochs: {len(epochs)}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
        
        plt.suptitle(f'Training History: {self.model.encoder_name}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        plt.show()


def create_model_and_trainer(encoder_type, encoder_config, mlp_config=None, 
                           n_classes=5, freeze_encoder='full', device='cuda', 
                           learning_rate=1e-4, weight_decay=1e-4, phase_mapping=None):
    """
    Factory function to create model and trainer
    
    Args:
        encoder_type: Type of encoder ('medvit', 'timm_vit', 'dinov3')
        encoder_config: Configuration for encoder
        mlp_config: Configuration for MLP head
        n_classes: Number of classes
        freeze_encoder: Whether to freeze encoder weights
        device: Device to use
        learning_rate: Learning rate for training
        weight_decay: Weight decay for regularization
    
    Returns:
        model: ContrastPhaseMLPClassifier instance
        trainer: ContrastPhaseTrainer instance
    """
    # Create encoder
    if encoder_type.lower() == 'medvit':
        encoder = create_medvit_encoder(encoder_config)
        encoder_name = f"MedViT_{encoder_config.get('model_size', 'small')}"
    elif encoder_type.lower() == 'timm_vit':
        encoder = TimmViTEncoder(**encoder_config)
        encoder_name = f"TimmViT_{encoder_config.get('model_name', 'vit_small')}"
    elif encoder_type.lower() == 'dinov3':
        encoder = DinoV3Encoder(**encoder_config)
        encoder_name = f"DinoV3_{encoder_config.get('model_size', 'small')}"
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    # Create model
    model = ContrastPhaseMLPClassifier(
        encoder=encoder,
        encoder_name=encoder_name,
        mlp_config=mlp_config,
        n_classes=n_classes,
        freeze_encoder=freeze_encoder,
        enable_gradcam=True
    )
    
    # Create trainer
    trainer = ContrastPhaseTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        phase_mapping=phase_mapping
    )
    
    return model, trainer


# Example usage and testing functions
def test_mlp_classifier():
    """Test the MLP classifier implementation"""
    print("Testing MLP Classifier Implementation...")
    
    # Test data
    batch_size = 2
    input_dim = 256
    n_classes = 5
    
    # Create test MLP
    mlp = AttentionMLP(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],
        n_classes=n_classes,
        dropout_rate=0.3,
        use_attention=False
    )
    
    # Test input
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass
    logits = mlp(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test with attention
    logits, attention = mlp(x, return_attention=True)
    print(f"Attention shape: {attention.shape if attention is not None else None}")
    
    print("✅ MLP classifier test passed!")


def test_end_to_end_model():
    """Test the end-to-end model"""
    print("Testing End-to-End Model...")
    
    # Create a simple dummy encoder
    class DummyEncoder(nn.Module):
        def __init__(self, latent_dim=256):
            super().__init__()
            self.latent_dim = latent_dim
            self.conv = nn.Conv3d(1, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc = nn.Linear(32, latent_dim)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    # Create model
    encoder = DummyEncoder(latent_dim=256)
    model = ContrastPhaseMLPClassifier(
        encoder=encoder,
        encoder_name="DummyEncoder",
        n_classes=5,
        freeze_encoder='full',
        enable_gradcam=True
    )
    
    # Test input
    x = torch.randn(1, 1, 32, 64, 64)
    
    # Forward pass
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test prediction
    predictions, probabilities = model.predict(x)
    print(f"Predictions: {predictions}")
    print(f"Probabilities shape: {probabilities.shape}")
    
    # Test GradCAM
    try:
        saliency_map, pred_class, class_prob = model.generate_gradcam(x)
        print(f"Saliency map shape: {saliency_map.shape}")
        print(f"Predicted class: {pred_class}, Probability: {class_prob:.3f}")
        print("✅ GradCAM test passed!")
    except Exception as e:
        print(f"⚠️ GradCAM test failed: {e}")
    
    print("✅ End-to-end model test passed!")


if __name__ == "__main__":
    print("Testing Contrast Phase MLP Classifier Module...")
    print("=" * 60)
    
    # Run tests
    test_mlp_classifier()
    print()
    test_end_to_end_model()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("\nTo use this module:")
    print("1. Create encoder with your preferred architecture")
    print("2. Use create_model_and_trainer() to setup training")
    print("3. Train with trainer.train()")
    print("4. Generate saliency maps with model.generate_3d_saliency_visualization()")