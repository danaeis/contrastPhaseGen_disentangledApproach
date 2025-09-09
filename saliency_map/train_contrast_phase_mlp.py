#!/usr/bin/env python3
"""
Comprehensive training script for contrast phase classification using MLP classifiers
with saliency map generation capabilities.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
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
    from dino_encoder import DinoV3Encoder
    from medViT_encoder import create_medvit_encoder
    from data import prepare_dataset_from_folders, prepare_data
    from feature_visualization import create_phase_mapping
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure the parent directory contains the required modules:")
    print("- models.py (with TimmViTEncoder)")
    print("- dino_encoder.py (with DinoV3Encoder)")
    print("- medViT_encoder.py (with create_medvit_encoder)")
    print("- data.py (with prepare_dataset_from_folders, prepare_data)")
    print("- feature_visualization.py (with create_phase_mapping)")
    sys.exit(1)

# Import the new MLP classifier module
from contrast_phase_mlp_classifier import (
    ContrastPhaseMLPClassifier, 
    ContrastPhaseTrainer, 
    create_model_and_trainer
)


class ContrastPhaseMLPExperiment:
    """
    Main experiment class for contrast phase classification with MLP and saliency maps
    """
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        (self.output_dir / 'saliency_maps').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.phase_mapping = create_phase_mapping()
        
        # Store experiment results
        self.results = {}
        
        print(f"Experiment initialized")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
    
    def setup_data(self):
        """Setup training and validation data loaders"""
        print("Setting up data loaders...")
        
        # if not DATA_FUNCTIONS_AVAILABLE:
        #     raise ImportError("Data functions not available. Please check data.py")
        
        # Prepare dataset from folders
        data_path = self.config['data_path']
        labels_csv = self.config.get('labels_csv')
        
        if labels_csv is None:
            labels_csv = os.path.join(data_path, 'labels.csv')
        
        if not os.path.exists(labels_csv):
            raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
        
        # Create train/val split
        train_data_dicts, val_data_dicts = prepare_dataset_from_folders(
            data_path, 
            labels_csv, 
            validation_split=self.config.get('validation_split', 0.2),
            skip_prep=True
        )
        
        # Limit data for debugging if requested
        max_samples = self.config.get('max_samples_debug')
        if max_samples:
            train_data_dicts = train_data_dicts[:max_samples]
            val_data_dicts = val_data_dicts[:max_samples//4] if val_data_dicts else []
            print(f"DEBUG: Limited to {len(train_data_dicts)} train, {len(val_data_dicts)} val samples")
        
        # Create data loaders
        spatial_size = tuple(self.config.get('spatial_size', [128, 128, 128]))
        batch_size = self.config.get('batch_size', 4)
        
        self.train_loader = prepare_data(
            train_data_dicts, 
            batch_size=batch_size,
            augmentation=self.config.get('use_augmentation', True),
            spatial_size=spatial_size
        )
        
        self.val_loader = prepare_data(
            val_data_dicts,
            batch_size=batch_size,
            augmentation=False,
            spatial_size=spatial_size
        ) if val_data_dicts else None
        
        print(f"Training samples: {len(train_data_dicts)}")
        print(f"Validation samples: {len(val_data_dicts) if val_data_dicts else 0}")
        print(f"Spatial size: {spatial_size}")
        print(f"Batch size: {batch_size}")
        
        # Store data info
        self.data_info = {
            'train_samples': len(train_data_dicts),
            'val_samples': len(val_data_dicts) if val_data_dicts else 0,
            'spatial_size': spatial_size,
            'batch_size': batch_size
        }
        
        return self.train_loader, self.val_loader
    
    def create_encoder_configs(self):
        """Create encoder configurations based on config"""
        encoder_configs = {}
        
        # Common parameters
        latent_dim = self.config.get('latent_dim', 256)
        max_slices = self.config.get('max_slices', 32)
        
        # MedViT configuration
        if self.config.get('use_medvit', True):
            medvit_config = {
                'model_size': self.config.get('medvit_size', 'small'),
                'pretrained_path': self.config.get('medvit_pretrained_path'),
                'latent_dim': latent_dim,
                'aggregation_method': self.config.get('aggregation_method', 'lstm'),
                'slice_sampling': 'uniform',
                'max_slices': max_slices
            }
            encoder_configs['MedViT'] = {'type': 'medvit', 'config': medvit_config}
        
        # Timm ViT configuration
        if self.config.get('use_timm_vit', True):
            timm_config = {
                'latent_dim': latent_dim,
                'model_name': self.config.get('timm_model_name', 'vit_small_patch16_224'),
                'pretrained': self.config.get('timm_pretrained', True),
                'max_slices': max_slices,
                'slice_sampling': 'uniform'
            }
            encoder_configs['TimmViT'] = {'type': 'timm_vit', 'config': timm_config}
        
        # DINO v3 configuration
        if self.config.get('use_dinov3', True):
            dinov3_config = {
                'latent_dim': latent_dim,
                'model_size': self.config.get('dinov3_size', 'small'),
                'pretrained': self.config.get('dinov3_pretrained', True),
                'max_slices': max_slices,
                'slice_sampling': 'uniform'
            }
            encoder_configs['DinoV3'] = {'type': 'dinov3', 'config': dinov3_config}
        
        return encoder_configs
    
    def create_mlp_config(self):
        """Create MLP configuration"""
        return {
            'hidden_dims': self.config.get('mlp_hidden_dims', [512, 256, 128]),
            'dropout_rate': self.config.get('mlp_dropout', 0.3),
            'use_attention': self.config.get('mlp_use_attention', True),
            'attention_heads': self.config.get('mlp_attention_heads', 8)
        }
    
    def train_single_model(self, encoder_name, encoder_config, mlp_config):
        """Train a single model"""
        print(f"\n{'='*60}")
        print(f"Training {encoder_name} Model")
        print(f"{'='*60}")
        
        try:
            # Create model and trainer
            model, trainer = create_model_and_trainer(
                encoder_type=encoder_config['type'],
                encoder_config=encoder_config['config'],
                mlp_config=mlp_config,
                n_classes=self.config.get('n_classes', 5),
                freeze_encoder=self.config.get('freeze_encoder', False),
                device=self.device,
                learning_rate=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
            
            print(f"Model created: {model.encoder_name}")
            print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            
            # Training parameters
            num_epochs = self.config.get('num_epochs', 50)
            model_save_path = self.output_dir / 'models' / f'{encoder_name}_best_model.pth'
            early_stopping_patience = self.config.get('early_stopping_patience', 10)
            
            # Train the model
            if self.val_loader is not None:
                train_history = trainer.train(
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    num_epochs=num_epochs,
                    save_best=True,
                    model_save_path=str(model_save_path),
                    early_stopping_patience=early_stopping_patience
                )
            else:
                print("No validation data available, using training data for validation")
                train_history = trainer.train(
                    train_loader=self.train_loader,
                    val_loader=self.train_loader,  # Use train as val for demonstration
                    num_epochs=num_epochs,
                    save_best=True,
                    model_save_path=str(model_save_path),
                    early_stopping_patience=early_stopping_patience
                )
            
            # Plot training history
            history_plot_path = self.output_dir / 'plots' / f'{encoder_name}_training_history.png'
            trainer.plot_training_history(save_path=str(history_plot_path))
            
            # Evaluate on validation set
            if self.val_loader is not None:
                print(f"\nEvaluating {encoder_name} on validation set...")
                val_loss, val_acc, detailed_metrics = trainer.validate_epoch(self.val_loader)
                
                # Plot confusion matrix
                cm_path = self.output_dir / 'plots' / f'{encoder_name}_confusion_matrix.png'
                self._plot_confusion_matrix(
                    detailed_metrics['confusion_matrix'], 
                    encoder_name, 
                    str(cm_path)
                )
            
            # Generate saliency maps for sample images
            self._generate_sample_saliency_maps(model, encoder_name)
            
            # Store results
            self.results[encoder_name] = {
                'model': model,
                'trainer': trainer,
                'train_history': train_history,
                'model_path': str(model_save_path),
                'final_val_acc': val_acc if self.val_loader else train_history['train_acc'][-1],
                'config': encoder_config
            }
            
            print(f"✅ {encoder_name} training completed successfully!")
            if self.val_loader:
                print(f"   Final validation accuracy: {val_acc:.4f}")
            else:
                print(f"   Final training accuracy: {train_history['train_acc'][-1]:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error training {encoder_name}: {e}")
            if self.config.get('debug_mode', False):
                import traceback
                print(traceback.format_exc())
            return False
    
    def _plot_confusion_matrix(self, cm, encoder_name, save_path):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        # Get phase names
        phase_names = [self.phase_mapping.get(i, f'Phase_{i}') for i in range(len(cm))]
        
        # Create heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=phase_names, 
            yticklabels=phase_names
        )
        
        plt.title(f'Confusion Matrix: {encoder_name}')
        plt.ylabel('True Phase')
        plt.xlabel('Predicted Phase')
        
        # Calculate and add accuracy
        accuracy = np.trace(cm) / np.sum(cm)
        plt.text(
            len(cm)/2, len(cm) + 0.1, 
            f'Accuracy: {accuracy:.3f}', 
            ha='center', 
            fontsize=14,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7)
        )
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to: {save_path}")
    
    def _generate_sample_saliency_maps(self, model, encoder_name, num_samples=3):
        """Generate saliency maps for sample images"""
        print(f"Generating sample saliency maps for {encoder_name}...")
        
        model.eval()
        
        # Get sample data
        data_iter = iter(self.val_loader if self.val_loader else self.train_loader)
        
        sample_count = 0
        for batch_idx, batch_data in enumerate(data_iter):
            if sample_count >= num_samples:
                break
            
            # Get data
            if isinstance(batch_data, dict):
                images = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)
            else:
                images, labels = batch_data
                images = images.to(self.device)
                labels = labels.to(self.device)
            
            # Process each image in the batch
            for i in range(min(images.shape[0], num_samples - sample_count)):
                if sample_count >= num_samples:
                    break
                
                # Get single image
                single_image = images[i:i+1]
                true_label = labels[i].item() if labels.dim() > 0 else labels.item()
                
                try:
                    # Generate GradCAM saliency map
                    saliency_results = model.generate_3d_saliency_visualization(
                        single_image,
                        save_path=str(
                            self.output_dir / 'saliency_maps' / 
                            f'{encoder_name}_gradcam_sample_{sample_count+1}.png'
                        ),
                        method='gradcam'
                    )
                    
                    # Generate attention-based saliency map if available
                    if model.mlp.use_attention:
                        try:
                            attention_results = model.generate_3d_saliency_visualization(
                                single_image,
                                save_path=str(
                                    self.output_dir / 'saliency_maps' / 
                                    f'{encoder_name}_attention_sample_{sample_count+1}.png'
                                ),
                                method='attention'
                            )
                        except Exception as e:
                            print(f"Attention saliency failed for sample {sample_count+1}: {e}")
                    
                    # Create comparison plot
                    self._create_saliency_comparison_plot(
                        saliency_results, 
                        true_label, 
                        encoder_name, 
                        sample_count + 1
                    )
                    
                    sample_count += 1
                    
                except Exception as e:
                    print(f"Failed to generate saliency map for sample {sample_count+1}: {e}")
                    sample_count += 1
                    continue
        
        print(f"Generated {sample_count} saliency map samples for {encoder_name}")
    
    def _create_saliency_comparison_plot(self, saliency_results, true_label, encoder_name, sample_idx):
        """Create detailed saliency comparison plot"""
        original_volume = saliency_results['original_volume']
        saliency_map = saliency_results['saliency_map']
        predicted_class = saliency_results['predicted_class']
        class_probability = saliency_results['class_probability']
        
        # Get phase names
        true_phase = self.phase_mapping.get(true_label, f'Phase_{true_label}')
        pred_phase = self.phase_mapping.get(predicted_class, f'Phase_{predicted_class}')
        
        # Select key slices for visualization
        depth = original_volume.shape[0]
        slice_indices = [
            depth // 4,           # First quarter
            depth // 2,           # Middle
            3 * depth // 4        # Third quarter
        ]
        
        # Create figure
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        for i, slice_idx in enumerate(slice_indices):
            # Original slice
            axes[i, 0].imshow(original_volume[slice_idx], cmap='gray')
            axes[i, 0].set_title(f'Original Slice {slice_idx}')
            axes[i, 0].axis('off')
            
            # Saliency map
            axes[i, 1].imshow(saliency_map[slice_idx], cmap='hot')
            axes[i, 1].set_title(f'Saliency Map {slice_idx}')
            axes[i, 1].axis('off')
            
            # Overlay
            axes[i, 2].imshow(original_volume[slice_idx], cmap='gray', alpha=0.7)
            axes[i, 2].imshow(saliency_map[slice_idx], cmap='hot', alpha=0.4)
            axes[i, 2].set_title(f'Overlay {slice_idx}')
            axes[i, 2].axis('off')
        
        # Add overall information
        title = f'{encoder_name} - Sample {sample_idx}\n'
        title += f'True: {true_phase} | Predicted: {pred_phase} ({class_probability:.3f})'
        
        if predicted_class == true_label:
            title += ' ✓'
        else:
            title += ' ✗'
        
        plt.suptitle(title, fontsize=14, y=0.95)
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / 'saliency_maps' / f'{encoder_name}_detailed_sample_{sample_idx}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_experiment(self):
        """Run the complete experiment"""
        print("Starting Contrast Phase Classification Experiment with MLP")
        print("=" * 80)
        
        # Setup data
        self.setup_data()
        
        # Get encoder configurations
        encoder_configs = self.create_encoder_configs()
        mlp_config = self.create_mlp_config()
        
        print(f"\nTraining {len(encoder_configs)} models:")
        for name in encoder_configs.keys():
            print(f"  - {name}")
        
        # Train each model
        successful_models = 0
        for encoder_name, encoder_config in encoder_configs.items():
            success = self.train_single_model(encoder_name, encoder_config, mlp_config)
            if success:
                successful_models += 1
        
        # Create comparison plots
        if successful_models > 1:
            self._create_model_comparison_plots()
        
        # Save experiment summary
        self._save_experiment_summary()
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT COMPLETED")
        print(f"{'='*80}")
        print(f"Successful models: {successful_models}/{len(encoder_configs)}")
        print(f"Results saved to: {self.output_dir}")
        
        if successful_models > 0:
            print(f"\nBest performing model:")
            best_model = max(
                self.results.items(), 
                key=lambda x: x[1]['final_val_acc']
            )
            print(f"  {best_model[0]}: {best_model[1]['final_val_acc']:.4f}")
        
        return self.results
    
    def _create_model_comparison_plots(self):
        """Create comparison plots across models"""
        print("Creating model comparison plots...")
        
        # Extract metrics
        model_names = list(self.results.keys())
        final_accs = [self.results[name]['final_val_acc'] for name in model_names]
        
        # Get training histories
        train_histories = {}
        for name in model_names:
            train_histories[name] = self.results[name]['train_history']
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Final accuracy comparison
        axes[0, 0].bar(model_names, final_accs, alpha=0.7)
        axes[0, 0].set_title('Final Validation Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for i, acc in enumerate(final_accs):
            axes[0, 0].text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
        
        # Training curves
        axes[0, 1].set_title('Training Loss Curves')
        axes[1, 0].set_title('Validation Accuracy Curves')
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (name, history) in enumerate(train_histories.items()):
            epochs = range(1, len(history['train_loss']) + 1)
            color = colors[i % len(colors)]
            
            # Training loss
            axes[0, 1].plot(epochs, history['train_loss'], 
                           color=color, label=name, alpha=0.7)
            
            # Validation accuracy
            axes[1, 0].plot(epochs, history['val_acc'], 
                           color=color, label=name, alpha=0.7)
        
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Model summary table
        axes[1, 1].axis('off')
        table_data = []
        for name in model_names:
            model = self.results[name]['model']
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            table_data.append([
                name,
                f"{total_params:,}",
                f"{trainable_params:,}",
                f"{final_accs[model_names.index(name)]:.3f}"
            ])
        
        table = axes[1, 1].table(
            cellText=table_data,
            colLabels=['Model', 'Total Params', 'Trainable Params', 'Final Acc'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('Model Summary')
        
        plt.suptitle('Model Comparison Results', fontsize=16)
        plt.tight_layout()
        
        # Save comparison plot
        comparison_path = self.output_dir / 'plots' / 'model_comparison.png'
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Model comparison plot saved to: {comparison_path}")
    
    def _save_experiment_summary(self):
        """Save comprehensive experiment summary"""
        # Create summary data
        summary = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'data_info': self.data_info,
                'device': str(self.device)
            },
            'results': {}
        }
        
        # Add results for each model
        for name, result in self.results.items():
            # Extract serializable data
            train_history = result['train_history']
            model_info = {
                'encoder_name': result['model'].encoder_name,
                'total_parameters': sum(p.numel() for p in result['model'].parameters()),
                'trainable_parameters': sum(p.numel() for p in result['model'].parameters() if p.requires_grad),
                'final_train_acc': float(train_history['train_acc'][-1]),
                'final_val_acc': float(result['final_val_acc']),
                'best_val_acc': float(max(train_history['val_acc'])),
                'total_epochs': len(train_history['train_acc']),
                'model_path': result['model_path'],
                'config': result['config']
            }
            
            summary['results'][name] = model_info
        
        # Save JSON summary
        json_path = self.output_dir / 'experiment_summary.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save text summary
        txt_path = self.output_dir / 'experiment_summary.txt'
        with open(txt_path, 'w') as f:
            f.write("CONTRAST PHASE CLASSIFICATION EXPERIMENT SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Timestamp: {summary['experiment_info']['timestamp']}\n")
            f.write(f"Device: {summary['experiment_info']['device']}\n")
            f.write(f"Training samples: {self.data_info['train_samples']}\n")
            f.write(f"Validation samples: {self.data_info['val_samples']}\n\n")
            
            f.write("MODEL RESULTS:\n")
            f.write("-" * 30 + "\n")
            
            for name, info in summary['results'].items():
                f.write(f"\n{name}:\n")
                f.write(f"  Total Parameters: {info['total_parameters']:,}\n")
                f.write(f"  Trainable Parameters: {info['trainable_parameters']:,}\n")
                f.write(f"  Final Training Accuracy: {info['final_train_acc']:.4f}\n")
                f.write(f"  Final Validation Accuracy: {info['final_val_acc']:.4f}\n")
                f.write(f"  Best Validation Accuracy: {info['best_val_acc']:.4f}\n")
                f.write(f"  Total Epochs: {info['total_epochs']}\n")
                f.write(f"  Model Path: {info['model_path']}\n")
        
        print(f"Experiment summary saved to: {json_path}")
        print(f"Text summary saved to: {txt_path}")


def load_trained_model_for_inference(model_path, encoder_type, encoder_config, 
                                    mlp_config=None, n_classes=5, device='cuda'):
    """
    Load a trained model for inference and saliency map generation
    
    Args:
        model_path: Path to saved model checkpoint
        encoder_type: Type of encoder used
        encoder_config: Encoder configuration
        mlp_config: MLP configuration (optional)
        n_classes: Number of classes
        device: Device to load model on
    
    Returns:
        Loaded model ready for inference
    """
    # Create model architecture
    model, _ = create_model_and_trainer(
        encoder_type=encoder_type,
        encoder_config=encoder_config,
        mlp_config=mlp_config,
        n_classes=n_classes,
        device=device
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from: {model_path}")
    print(f"Model validation accuracy: {checkpoint.get('val_acc', 'Unknown'):.4f}")
    
    return model


def create_default_config():
    """Create default configuration for the experiment"""
    return {
        # Data configuration
        'data_path': 'data',
        'labels_csv': None,  # Will default to data_path/labels.csv
        'spatial_size': [128, 128, 128],
        'batch_size': 4,
        'validation_split': 0.2,
        'use_augmentation': True,
        'max_samples_debug': None,
        
        # Model configuration
        'n_classes': 5,
        'latent_dim': 256,
        'max_slices': 32,
        'freeze_encoder': False,
        
        # Encoder selection
        'use_medvit': True,
        'use_timm_vit': True,
        'use_dinov3': True,
        
        # MedViT specific
        'medvit_size': 'small',
        'medvit_pretrained_path': 'pretrained_medvit_small.pth',
        'aggregation_method': 'lstm',
        
        # Timm ViT specific
        'timm_model_name': 'vit_small_patch16_224',
        'timm_pretrained': True,
        
        # DINO v3 specific
        'dinov3_size': 'small',
        'dinov3_pretrained': True,
        
        # MLP configuration
        'mlp_hidden_dims': [512, 256, 128],
        'mlp_dropout': 0.3,
        'mlp_use_attention': True,
        'mlp_attention_heads': 8,
        
        # Training configuration
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'early_stopping_patience': 10,
        
        # System configuration
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'output_dir': 'contrast_phase_mlp_results',
        'debug_mode': False
    }


def main():
    """Main function for running contrast phase classification experiments"""
    parser = argparse.ArgumentParser(
        description="Train contrast phase classification models with MLP and generate saliency maps"
    )
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='data', 
                       help='Path to training data directory')
    parser.add_argument('--labels_csv', type=str, default=None, 
                       help='Path to labels CSV file')
    parser.add_argument('--spatial_size', type=int, nargs=3, default=[128, 128, 128], 
                       help='Input volume size [depth, height, width]')
    parser.add_argument('--batch_size', type=int, default=4, 
                       help='Batch size for training')
    parser.add_argument('--validation_split', type=float, default=0.2, 
                       help='Fraction of data for validation')
    
    # Model arguments
    parser.add_argument('--n_classes', type=int, default=5, 
                       help='Number of contrast phases')
    parser.add_argument('--latent_dim', type=int, default=256, 
                       help='Latent dimension for encoders')
    parser.add_argument('--freeze_encoder', action='store_true', 
                       help='Freeze encoder weights during training')
    
    # Encoder selection
    parser.add_argument('--use_medvit', action='store_true', default=True, 
                       help='Use MedViT encoder')
    parser.add_argument('--use_timm_vit', action='store_true', default=True, 
                       help='Use Timm ViT encoder')
    parser.add_argument('--use_dinov3', action='store_true', default=True, 
                       help='Use DINO v3 encoder')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=50, 
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                       help='Weight decay')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--output_dir', type=str, default='contrast_phase_mlp_results', 
                       help='Output directory for results')
    parser.add_argument('--debug_mode', action='store_true', 
                       help='Enable debug mode')
    parser.add_argument('--max_samples_debug', type=int, default=None, 
                       help='Limit samples for debugging')
    
    # Inference mode
    parser.add_argument('--inference_mode', action='store_true', 
                       help='Run in inference mode to generate saliency maps')
    parser.add_argument('--model_path', type=str, default=None, 
                       help='Path to trained model for inference')
    parser.add_argument('--test_data_path', type=str, default=None, 
                       help='Path to test data for inference')
    
    args = parser.parse_args()
    
    # Create configuration from arguments
    config = create_default_config()
    
    # Update config with command line arguments
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None:
            config[arg_name] = arg_value
    
    # Handle device selection
    if config['device'] == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    if args.inference_mode:
        # Inference mode - generate saliency maps for existing model
        if not args.model_path:
            print("Error: --model_path required for inference mode")
            return
        
        print("Running in inference mode...")
        # This would require additional implementation for loading and testing
        print("Inference mode implementation needed - see load_trained_model_for_inference function")
        
    else:
        # Training mode
        print("Running in training mode...")
        
        # Create and run experiment
        experiment = ContrastPhaseMLPExperiment(config)
        results = experiment.run_experiment()
        
        print("\nExperiment completed successfully!")
        print(f"Results available in: {experiment.output_dir}")


if __name__ == "__main__":
    main()