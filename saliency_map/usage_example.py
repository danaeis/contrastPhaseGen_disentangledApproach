import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Import the MLP classifier modules
from contrast_phase_mlp_classifier import (
    ContrastPhaseMLPClassifier, 
    ContrastPhaseTrainer, 
    create_model_and_trainer
)
from train_contrast_phase_mlp import (
    ContrastPhaseMLPExperiment,
    create_default_config,
    load_trained_model_for_inference
)

# Setup imports from parent directory
import sys
import os
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import your existing modules (adjust imports as needed)
try:
    from models import TimmViTEncoder
    from dino_encoder import DinoV3Encoder
    from medViT_encoder import create_medvit_encoder
    from data import prepare_dataset_from_folders, prepare_data
except ImportError as e:
    print(f"Import warning: {e}")
    print("Please ensure all required modules are available")


def quick_training_example():
    """
    Quick example of training a single model with the MLP classifier
    """
    print("=" * 60)
    print("QUICK TRAINING EXAMPLE")
    print("=" * 60)
    
    # Configuration for a quick test
    config = {
        'data_path': 'data',  # Your data path
        'spatial_size': [64, 64, 64],  # Smaller size for quick testing
        'batch_size': 2,
        'n_classes': 5,
        'latent_dim': 128,  # Smaller latent dim for quick testing
        'num_epochs': 5,    # Few epochs for quick test
        'learning_rate': 1e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'output_dir': 'quick_test_results',
        'max_samples_debug': 20,  # Limit samples for quick test
        'use_medvit': False,      # Disable slower models for quick test
        'use_dinov3': False,
        'use_timm_vit': True      # Only use one model for quick test
    }
    
    # Create experiment
    experiment = ContrastPhaseMLPExperiment(config)
    
    try:
        # Run the experiment
        results = experiment.run_experiment()
        
        print("\n✅ Quick training example completed!")
        print(f"Results saved to: {experiment.output_dir}")
        
        return results
        
    except Exception as e:
        print(f"❌ Quick training example failed: {e}")
        print("This is expected if you don't have the data setup")
        return None


def manual_model_creation_example():
    """
    Example of manually creating and using a model
    """
    print("=" * 60)
    print("MANUAL MODEL CREATION EXAMPLE")
    print("=" * 60)
    
    # Create a simple dummy encoder for demonstration
    class DummyEncoder(nn.Module):
        def __init__(self, latent_dim=256):
            super().__init__()
            self.latent_dim = latent_dim
            self.conv = nn.Conv3d(1, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool3d((4, 4, 4))
            self.fc = nn.Linear(32 * 4 * 4 * 4, latent_dim)
            
        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    # Create encoder and model
    print("Creating dummy encoder and MLP classifier...")
    encoder = DummyEncoder(latent_dim=256)
    
    # MLP configuration
    mlp_config = {
        'hidden_dims': [256, 128, 64],
        'dropout_rate': 0.3,
        'use_attention': True,
        'attention_heads': 4
    }
    
    # Create the complete model
    model = ContrastPhaseMLPClassifier(
        encoder=encoder,
        encoder_name="DummyEncoder",
        mlp_config=mlp_config,
        n_classes=5,
        freeze_encoder=False,
        enable_gradcam=True
    )
    
    print(f"Model created: {model.encoder_name}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with dummy data
    print("\nTesting with dummy data...")
    dummy_input = torch.randn(2, 1, 32, 64, 64)  # (batch, channels, depth, height, width)
    
    # Forward pass
    with torch.no_grad():
        logits = model(dummy_input)
        predictions, probabilities = model.predict(dummy_input)
        
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output logits shape: {logits.shape}")
        print(f"Predictions: {predictions}")
        print(f"Probabilities shape: {probabilities.shape}")
    
    # Test saliency map generation
    print("\nTesting saliency map generation...")
    try:
        # Generate GradCAM saliency map
        saliency_map, predicted_class, class_prob = model.generate_gradcam(dummy_input[:1])
        
        print(f"Saliency map shape: {saliency_map.shape}")
        print(f"Predicted class: {predicted_class}")
        print(f"Class probability: {class_prob:.3f}")
        
        # Visualize a slice of the saliency map
        plt.figure(figsize=(12, 4))
        
        # Original slice
        plt.subplot(1, 3, 1)
        plt.imshow(dummy_input[0, 0, 16].cpu().numpy(), cmap='gray')
        plt.title('Original Slice')
        plt.axis('off')
        
        # Saliency map slice
        plt.subplot(1, 3, 2)
        plt.imshow(saliency_map[16], cmap='hot')
        plt.title('Saliency Map')
        plt.axis('off')
        
        # Overlay
        plt.subplot(1, 3, 3)
        plt.imshow(dummy_input[0, 0, 16].cpu().numpy(), cmap='gray', alpha=0.7)
        plt.imshow(saliency_map[16], cmap='hot', alpha=0.3)
        plt.title('Overlay')
        plt.axis('off')
        
        plt.suptitle('Dummy Saliency Map Example')
        plt.tight_layout()
        
        # Save if possible
        try:
            save_path = experiment.output_dir + '/dummy_saliency_example.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saliency map example saved to: {save_path}")
        except:
            print("Could not save figure, but visualization was successful")
        
        plt.show()
        
        print("✅ Saliency map generation successful!")
        
    except Exception as e:
        print(f"❌ Saliency map generation failed: {e}")
    
    # Test training setup
    print("\nTesting training setup...")
    try:
        trainer = ContrastPhaseTrainer(
            model=model,
            device='cpu',  # Use CPU for testing
            learning_rate=1e-3
        )
        
        print("✅ Trainer created successfully!")
        print(f"Optimizer: {type(trainer.optimizer).__name__}")
        print(f"Loss function: {type(trainer.criterion).__name__}")
        
    except Exception as e:
        print(f"❌ Trainer creation failed: {e}")
    
    print("\n✅ Manual model creation example completed!")


def saliency_map_comparison_example():
    """
    Example of comparing different saliency map methods
    """
    print("=" * 60)
    print("SALIENCY MAP COMPARISON EXAMPLE")
    print("=" * 60)
    
    # Create a simple model for testing
    class SimpleEncoder(nn.Module):
        def __init__(self, latent_dim=128):
            super().__init__()
            self.latent_dim = latent_dim
            self.features = nn.Sequential(
                nn.Conv3d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv3d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((2, 2, 2)),
                nn.Flatten(),
                nn.Linear(32 * 2 * 2 * 2, latent_dim)
            )
            
        def forward(self, x):
            return self.features(x)
    
    # Create model with attention
    encoder = SimpleEncoder(latent_dim=128)
    model = ContrastPhaseMLPClassifier(
        encoder=encoder,
        encoder_name="SimpleEncoder",
        mlp_config={
            'hidden_dims': [128, 64],
            'use_attention': True,
            'attention_heads': 4
        },
        n_classes=5,
        enable_gradcam=True
    )
    
    # Create test data with some structure
    print("Creating structured test data...")
    test_volume = torch.zeros(1, 1, 32, 32, 32)
    
    # Add some structure to make saliency maps more interesting
    # Central bright region
    test_volume[0, 0, 12:20, 12:20, 12:20] = 1.0
    # Some noise
    test_volume += torch.randn_like(test_volume) * 0.1
    # Border artifacts
    test_volume[0, 0, :2, :, :] = 0.5
    test_volume[0, 0, -2:, :, :] = 0.5
    
    print(f"Test volume shape: {test_volume.shape}")
    print(f"Test volume range: [{test_volume.min():.3f}, {test_volume.max():.3f}]")
    
    # Generate different types of saliency maps
    try:
        print("\nGenerating GradCAM saliency map...")
        gradcam_map, pred_class, prob = model.generate_gradcam(test_volume)
        
        print("\nGenerating attention-based saliency map...")
        attention_map, pred_class_att, prob_att = model.generate_attention_map(test_volume)
        
        print(f"GradCAM - Predicted: {pred_class}, Prob: {prob:.3f}")
        print(f"Attention - Predicted: {pred_class_att}, Prob: {prob_att:.3f}")
        
        # Visualize comparison
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Select middle slices for visualization
        slice_idx = test_volume.shape[2] // 2
        
        # Original
        axes[0, 0].imshow(test_volume[0, 0, slice_idx].cpu().numpy(), cmap='gray')
        axes[0, 0].set_title('Original (Axial)')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(test_volume[0, 0, :, slice_idx].cpu().numpy(), cmap='gray')
        axes[1, 0].set_title('Original (Coronal)')
        axes[1, 0].axis('off')
        
        # GradCAM
        axes[0, 1].imshow(gradcam_map[slice_idx], cmap='hot')
        axes[0, 1].set_title('GradCAM (Axial)')
        axes[0, 1].axis('off')
        
        axes[1, 1].imshow(gradcam_map[:, slice_idx], cmap='hot')
        axes[1, 1].set_title('GradCAM (Coronal)')
        axes[1, 1].axis('off')
        
        # Attention (create spatial map from feature attention)
        attention_spatial = np.ones_like(gradcam_map) * np.mean(attention_map)
        
        axes[0, 2].imshow(attention_spatial[slice_idx], cmap='viridis')
        axes[0, 2].set_title('Attention (Axial)')
        axes[0, 2].axis('off')
        
        axes[1, 2].imshow(attention_spatial[:, slice_idx], cmap='viridis')
        axes[1, 2].set_title('Attention (Coronal)')
        axes[1, 2].axis('off')
        
        # Overlays
        axes[0, 3].imshow(test_volume[0, 0, slice_idx].detach().cpu().numpy(), cmap='gray', alpha=0.7)
        axes[0, 3].imshow(gradcam_map[slice_idx], cmap='hot', alpha=0.3)
        axes[0, 3].set_title('GradCAM Overlay (Axial)')
        axes[0, 3].axis('off')
        
        axes[1, 3].imshow(test_volume[0, 0, :, slice_idx].detach().cpu().numpy(), cmap='gray', alpha=0.7)
        axes[1, 3].imshow(gradcam_map[:, slice_idx], cmap='hot', alpha=0.3)
        axes[1, 3].set_title('GradCAM Overlay (Coronal)')
        axes[1, 3].axis('off')
        
        plt.suptitle('Saliency Map Comparison Example', fontsize=16)
        plt.tight_layout()
        
        # Save if possible
        try:
            save_path = 'saliency_comparison_example.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saliency comparison saved to: {save_path}")
        except:
            print("Could not save figure, but visualization was successful")
        
        plt.show()
        
        print("✅ Saliency map comparison completed!")
        
    except Exception as e:
        print(f"❌ Saliency map comparison failed: {e}")
        import traceback
        print(traceback.format_exc())


def configuration_examples():
    """
    Show different configuration examples
    """
    print("=" * 60)
    print("CONFIGURATION EXAMPLES")
    print("=" * 60)
    
    # Example 1: Quick testing configuration
    print("1. Quick Testing Configuration:")
    quick_config = {
        'data_path': 'data',
        'spatial_size': [64, 64, 64],
        'batch_size': 2,
        'n_classes': 5,
        'latent_dim': 128,
        'num_epochs': 5,
        'max_samples_debug': 20,
        'use_medvit': False,
        'use_dinov3': False,
        'use_timm_vit': True,
        'mlp_hidden_dims': [128, 64],
        'freeze_encoder': True  # For quick testing
    }
    print("   - Small spatial size, few epochs")
    print("   - Single encoder (TimmViT)")
    print("   - Frozen encoder for faster training")
    print("   - Limited data samples")
    
    # Example 2: Full training configuration
    print("\n2. Full Training Configuration:")
    full_config = {
        'data_path': 'data',
        'spatial_size': [128, 128, 128],
        'batch_size': 8,
        'n_classes': 5,
        'latent_dim': 512,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'use_medvit': True,
        'use_dinov3': True,
        'use_timm_vit': True,
        'mlp_hidden_dims': [1024, 512, 256],
        'mlp_use_attention': True,
        'freeze_encoder': False,
        'early_stopping_patience': 15
    }
    print("   - Full spatial size")
    print("   - All three encoders")
    print("   - End-to-end training")
    print("   - Large MLP with attention")
    
    # Example 3: High-performance configuration
    print("\n3. High-Performance Configuration:")
    high_perf_config = {
        'data_path': 'data',
        'spatial_size': [256, 256, 256],
        'batch_size': 4,  # Smaller batch due to larger images
        'n_classes': 5,
        'latent_dim': 1024,
        'num_epochs': 200,
        'learning_rate': 5e-5,  # Lower LR for stability
        'weight_decay': 1e-5,
        'medvit_size': 'large',
        'dinov3_size': 'large',
        'timm_model_name': 'vit_large_patch16_224',
        'mlp_hidden_dims': [2048, 1024, 512, 256],
        'mlp_dropout': 0.4,
        'mlp_attention_heads': 16,
        'max_slices': 64,  # More slices for better 3D representation
        'freeze_encoder': False
    }
    print("   - Large spatial size for high resolution")
    print("   - Large encoder models")
    print("   - Deep MLP with more attention heads")
    print("   - More slices for better 3D representation")
    
    # Example 4: Saliency-focused configuration
    print("\n4. Saliency-Focused Configuration:")
    saliency_config = {
        'data_path': 'data',
        'spatial_size': [128, 128, 128],
        'batch_size': 1,  # Single sample for detailed saliency analysis
        'n_classes': 5,
        'latent_dim': 256,
        'num_epochs': 50,
        'use_medvit': True,  # MedViT often gives good saliency maps
        'use_dinov3': False,
        'use_timm_vit': False,
        'mlp_use_attention': True,  # Attention helps with interpretability
        'mlp_attention_heads': 8,
        'freeze_encoder': False,  # Trainable for better saliency
        'aggregation_method': 'attention',  # Use attention for slice aggregation
    }
    print("   - Focus on single encoder (MedViT)")
    print("   - Attention-based aggregation")
    print("   - Batch size 1 for detailed saliency analysis")
    print("   - Trainable encoder for better gradient flow")
    
    print("\n✅ Configuration examples completed!")


def main():
    """
    Main function demonstrating different usage examples
    """
    print("CONTRAST PHASE MLP CLASSIFIER - USAGE EXAMPLES")
    print("=" * 80)
    
    examples = [
        ("Manual Model Creation", manual_model_creation_example),
        ("Saliency Map Comparison", saliency_map_comparison_example),
        ("Configuration Examples", configuration_examples),
        ("Quick Training Example", quick_training_example),  # This one might fail without data
    ]
    
    for name, example_func in examples:
        print(f"\n\nRunning: {name}")
        try:
            example_func()
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            print("This might be expected if data is not available")
    
    print("\n" + "=" * 80)
    print("USAGE EXAMPLES COMPLETED")
    print("=" * 80)
    
    print("\nTo use this module in your project:")
    print("1. Install requirements: torch, torchvision, numpy, matplotlib, seaborn, sklearn")
    print("2. Ensure your data is in the correct format")
    print("3. Use train_contrast_phase_mlp.py for full experiments")
    print("4. Use the configuration examples as starting points")
    print("5. Generate saliency maps with the trained models")
    
    print("\nKey advantages over LDA approach:")
    print("✅ End-to-end trainable")
    print("✅ Better gradient flow for saliency maps")
    print("✅ Attention mechanisms for interpretability")
    print("✅ More flexible architecture")
    print("✅ Better performance on complex data")


if __name__ == "__main__":
    main()