# main.py - Optimized CT Contrast Phase Generation Pipeline

import torch
import argparse
import os
from models import Simple3DCNNEncoder, TimmViTEncoder, ResNet3DEncoder, LightweightHybridEncoder
from models import Generator, Discriminator, PhaseDetector, StableLightweightHybridEncoder
from data import prepare_data, prepare_dataset_from_folders
from training import train_contrast_phase_generation
from training_dann_style import train_dann_style_contrast_generation
from inference import benchmark_inference, generate_contrast_phase, save_volume
from medViT_encoder import create_medvit_encoder

# Try to import optional encoders
try:
    from monai_totalseg_encoder import create_monai_totalsegmentator_encoder
    MONAI_TOTALSEG_AVAILABLE = True
except ImportError:
    MONAI_TOTALSEG_AVAILABLE = False

def create_encoder(args):
    """Unified encoder creation with optimized config handling"""
    encoder_configs = {
        'simple_cnn': {
            'class': Simple3DCNNEncoder,
            'params': {'in_channels': 1, 'latent_dim': args.latent_dim, 'img_size': tuple(args.spatial_size)}
        },
        'timm_vit': {
            'class': TimmViTEncoder,
            'params': {'latent_dim': args.latent_dim, 'model_name': getattr(args, 'timm_model_name', 'vit_small_patch16_224'),
                      'pretrained': getattr(args, 'timm_pretrained', False), 'max_slices': getattr(args, 'max_slices', 32)}
        },
        'resnet3d': {
            'class': ResNet3DEncoder,
            'params': {'latent_dim': args.latent_dim, 'in_channels': 1}
        },
        'hybrid': {
            'class': StableLightweightHybridEncoder,
            'params': {'latent_dim': args.latent_dim, 'in_channels': 1}
        },
        'medvit': {
            'factory': create_medvit_encoder,
            'params': {
                'model_size': getattr(args, 'medvit_size', 'small'),
                'pretrained_path': getattr(args, 'medvit_pretrained_path', None) if hasattr(args, 'medvit_pretrained_path') and os.path.exists(getattr(args, 'medvit_pretrained_path', '')) else None,
                'latent_dim': args.latent_dim,
                'aggregation_method': getattr(args, 'aggregation_method', 'lstm'),
                'slice_sampling': 'uniform',
                'max_slices': getattr(args, 'max_slices', 32)
            }
        }
    }
    
    if args.encoder == 'monai_totalseg' and MONAI_TOTALSEG_AVAILABLE:
        encoder_configs['monai_totalseg'] = {
            'factory': create_monai_totalsegmentator_encoder,
            'params': {
                'latent_dim': args.latent_dim,
                'device': args.device,
                'roi_size': getattr(args, 'totalseg_roi_size', (32, 32, 32)),
                'use_enhanced_features': getattr(args, 'totalseg_enhanced', False),
                'use_pretrained': True
            }
        }
    
    if args.encoder not in encoder_configs:
        print(f"❌ Unknown encoder: {args.encoder}, falling back to simple_cnn")
        args.encoder = 'simple_cnn'
    
    config = encoder_configs[args.encoder]
    
    try:
        if 'factory' in config:
            encoder = config['factory'](config['params'])
        else:
            encoder = config['class'](**config['params'])
        
        print(f"✅ Created {args.encoder} encoder")
        return encoder, {'type': args.encoder, 'config': config['params']}
        
    except Exception as e:
        print(f"❌ Error creating {args.encoder} encoder: {e}")
        print("🔄 Falling back to simple_cnn encoder")
        fallback = Simple3DCNNEncoder(in_channels=1, latent_dim=args.latent_dim, img_size=tuple(args.spatial_size))
        return fallback, {'type': 'simple_cnn', 'config': {'latent_dim': args.latent_dim}}

def create_models(args):
    """Create all models efficiently"""
    encoder, encoder_config = create_encoder(args)
    img_size = tuple(args.spatial_size)
    
    generator = Generator(
        latent_dim=args.latent_dim,
        phase_dim=32,
        output_shape=(*img_size, 1)
    )
    
    discriminator = Discriminator(input_shape=(*img_size, 1))
    phase_detector = PhaseDetector(latent_dim=args.latent_dim, num_phases=4)
    
    return encoder, generator, discriminator, phase_detector, encoder_config

def train_models(args, train_loader, val_loader, models, checkpoint_dir, encoder_config):
    """Unified training function"""
    encoder, generator, discriminator, phase_detector = models
    
    # Select training strategy
    if args.training_strategy == "dann":
        print("🎯 Using DANN-style simultaneous training")
        return train_dann_style_contrast_generation(
            train_loader, encoder, generator, discriminator, phase_detector,
            num_epochs=args.epochs, device=args.device, checkpoint_dir=checkpoint_dir,
            use_mixed_precision=args.mixed_precision, validation_loader=val_loader,
            encoder_config=encoder_config, encoder_type=args.encoder
        )
    else:
        print("🎯 Using sequential training")
        return train_contrast_phase_generation(
            train_loader, encoder, generator, discriminator, phase_detector,
            num_epochs=args.epochs, device=args.device, checkpoint_dir=checkpoint_dir,
            use_mixed_precision=args.mixed_precision, validation_loader=val_loader,
            encoder_config=encoder_config, encoder_type=args.encoder
        )

def load_checkpoint_and_models(args):
    """Load checkpoint and recreate models for inference"""
    print(f"📄 Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Recreate encoder from checkpoint config
    if 'encoder_config' in checkpoint and checkpoint['encoder_config']:
        encoder_info = checkpoint['encoder_config']
        encoder_type = encoder_info['type']
        
        # Temporarily set args.encoder to the checkpoint type
        original_encoder = args.encoder
        args.encoder = encoder_type
        encoder, _ = create_encoder(args)
        args.encoder = original_encoder  # Restore
    else:
        # Fallback encoder
        img_size = tuple(args.spatial_size)
        encoder = Simple3DCNNEncoder(in_channels=1, latent_dim=args.latent_dim, img_size=img_size)
    
    # Create other models
    img_size = tuple(args.spatial_size)
    generator = Generator(latent_dim=args.latent_dim, phase_dim=32, output_shape=(*img_size, 1))
    
    # Load weights
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    # Move to device and set to eval
    encoder.to(args.device).eval()
    generator.to(args.device).eval()
    
    return encoder, generator

def run_inference(args):
    """Run inference with optimized loading"""
    required_args = ['checkpoint', 'input_volume', 'input_phase', 'target_phase', 'output_path']
    missing_args = [arg for arg in required_args if getattr(args, arg, None) is None]
    
    if missing_args:
        print(f"❌ Missing required arguments for inference: {missing_args}")
        return False
    
    try:
        # Load models
        encoder, generator = load_checkpoint_and_models(args)
        
        # Load and preprocess input
        print(f"📄 Loading input volume from {args.input_volume}")
        
        # Try MONAI first, fallback to nibabel
        try:
            from monai.transforms import LoadImage, EnsureChannelFirst, ScaleIntensityRange, Resize
            from monai.transforms import Compose as MonaiCompose
            
            transform = MonaiCompose([
                LoadImage(),
                EnsureChannelFirst(),
                ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0),
                Resize(spatial_size=args.spatial_size)
            ])
            
            input_volume = transform(args.input_volume).unsqueeze(0)
            
        except ImportError:
            # Fallback to nibabel
            import nibabel as nib
            
            nii = nib.load(args.input_volume)
            input_volume = torch.from_numpy(nii.get_fdata()).float()
            
            if input_volume.dim() == 3:
                input_volume = input_volume.unsqueeze(0).unsqueeze(0)
            
            input_volume = torch.clamp((input_volume + 1000) / 2000, 0, 1)
            
            if input_volume.shape[2:] != tuple(args.spatial_size):
                input_volume = torch.nn.functional.interpolate(
                    input_volume, size=args.spatial_size, mode='trilinear', align_corners=False
                )
        
        # Generate
        phase_names = ['arterial', 'venous', 'delayed', 'non-contrast']
        print(f"🎭 Generating {phase_names[args.target_phase]} from {phase_names[args.input_phase]}")
        
        generated_volume = generate_contrast_phase(
            input_volume, args.input_phase, args.target_phase, encoder, generator,
            device=args.device, use_mixed_precision=args.mixed_precision
        )
        
        # Save
        print(f"💾 Saving to {args.output_path}")
        save_volume(generated_volume, args.output_path)
        print("✅ Inference completed!")
        return True
        
    except Exception as e:
        print(f"❌ Inference error: {e}")
        return False

def run_benchmark(args):
    """Run benchmark with optimized loading"""
    if not args.checkpoint:
        print("❌ --checkpoint required for benchmark mode")
        return False
    
    try:
        encoder, generator = load_checkpoint_and_models(args)
        
        print("🚀 Running inference benchmark...")
        avg_time = benchmark_inference(encoder, generator, device=args.device)
        print(f"✅ Benchmark complete! Average: {avg_time*1000:.2f} ms ({1/avg_time:.2f} volumes/sec)")
        return True
        
    except Exception as e:
        print(f"❌ Benchmark error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Optimized CT Contrast Phase Generation")
    
    # Core arguments
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference", "benchmark"])
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Training arguments
    parser.add_argument("--training_strategy", type=str, default="sequential", choices=["sequential", "dann"])
    parser.add_argument("--phase_embedding", type=str, default="simple", choices=["simple", "sinusoidal"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--mixed_precision", action="store_true")
    
    # Model arguments  
    parser.add_argument("--encoder", type=str, default="simple_cnn", 
                       choices=["simple_cnn", "timm_vit", "resnet3d", "medvit", "hybrid", "monai_totalseg"])
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[128, 128, 128])
    
    # Encoder-specific arguments (simplified)
    parser.add_argument("--medvit_size", type=str, default="small", choices=["tiny", "small", "base"])
    parser.add_argument("--medvit_pretrained_path", type=str, default="pretrained_medvit_small.pth")
    parser.add_argument("--aggregation_method", type=str, default="lstm")
    parser.add_argument("--max_slices", type=int, default=32)
    parser.add_argument("--timm_model_name", type=str, default="vit_small_patch16_224")
    parser.add_argument("--timm_pretrained", action="store_true")
    parser.add_argument("--totalseg_roi_size", type=int, nargs=3, default=[32, 32, 32])
    parser.add_argument("--totalseg_enhanced", action="store_true")
    
    # Inference arguments
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--input_volume", type=str)
    parser.add_argument("--input_phase", type=int, choices=[0, 1, 2, 3])
    parser.add_argument("--target_phase", type=int, choices=[0, 1, 2, 3])
    parser.add_argument("--output_path", type=str)
    
    # Data preparation
    parser.add_argument("--apply_registration", action="store_true")
    parser.add_argument("--skip_prep", action="store_true")
    
    args = parser.parse_args()
    
    print(f"🚀 Optimized CT Contrast Phase Generation Pipeline")
    print(f"📊 Mode: {args.mode} | Device: {args.device} | Strategy: {getattr(args, 'training_strategy', 'N/A')}")
    
    if args.mode == "train":
        # Create checkpoint directory
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        # Prepare dataset
        print("📄 Preparing dataset...")
        labels_csv = os.path.join(args.data_path, "labels.csv")
        
        if not os.path.exists(labels_csv):
            print(f"❌ Labels CSV not found: {labels_csv}")
            return
        
        train_data_dicts, val_data_dicts = prepare_dataset_from_folders(
            args.data_path, labels_csv, validation_split=0.2,
            apply_registration=args.apply_registration, skip_prep=args.skip_prep
        )
        
        print(f"✅ Dataset: {len(train_data_dicts)} train, {len(val_data_dicts)} val samples")
        
        # Create data loaders
        img_size = tuple(args.spatial_size)
        train_loader = prepare_data(train_data_dicts, batch_size=args.batch_size, spatial_size=img_size)
        val_loader = prepare_data(val_data_dicts, batch_size=args.batch_size, augmentation=False, spatial_size=img_size)
        
        # Create models
        print("🏗️ Creating models...")
        models = create_models(args)
        encoder, generator, discriminator, phase_detector, encoder_config = models
        
        # Train models
        metrics = train_models(args, train_loader, val_loader, models[:4], args.checkpoint_dir, encoder_config)
        
        print("🎉 Training completed!")
        if args.training_strategy == "dann":
            print(f"📊 Final reconstruction loss: {metrics['reconstruction_losses'][-1]:.6f}")
            if metrics['confusion_accuracies'][-1] > 0.4:
                print("🎯 DANN Success: Learned phase-invariant features!")
        
    elif args.mode == "inference":
        success = run_inference(args)
        if not success:
            return 1
    
    elif args.mode == "benchmark":
        success = run_benchmark(args)
        if not success:
            return 1
    
    return 0

if __name__ == "__main__":
    exit(main())