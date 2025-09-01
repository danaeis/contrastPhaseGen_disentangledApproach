import torch
import argparse
import os
from models import Simple3DCNNEncoder, TimmViTEncoder, ResNet3DEncoder, LightweightHybridEncoder, Generator, Discriminator, PhaseDetector
from data import prepare_data, prepare_dataset_from_folders
from training import train_contrast_phase_generation
from inference import benchmark_inference, generate_contrast_phase, save_volume
from arc_medViT_encoder import create_medvit_encoder

def debug_generator_dimensions(encoder, args):
    """Debug function to identify dimension mismatches"""
    
    print("=" * 50)
    print("DEBUGGING GENERATOR DIMENSIONS")
    print("=" * 50)
    
    # Test encoder output
    dummy_input = torch.randn(1, 1, *args.spatial_size)
    
    with torch.no_grad():
        encoder_output = encoder(dummy_input)
        print(f"Encoder output shape: {encoder_output.shape}")
        print(f"Encoder latent_dim: {encoder_output.shape[-1]}")
    
    # Test phase embedding
    from utils import get_phase_embedding
    phase_emb = get_phase_embedding(0, dim=32, device='cpu')
    print(f"Phase embedding shape: {phase_emb.shape}")
    
    # Combined input to generator
    combined_input = torch.cat([encoder_output, phase_emb.unsqueeze(0)], dim=1)
    print(f"Combined input shape: {combined_input.shape}")
    print(f"Expected generator input: latent_dim + phase_dim = ? + 32")
    
    return encoder_output.shape[-1]

def main():
    parser = argparse.ArgumentParser(description="CT Contrast Phase Generation Pipeline")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference", "benchmark"], 
                        help="Mode: train, inference, or benchmark")
    parser.add_argument("--data_path", type=str, default="data", help="Path to data directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for saving checkpoints")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    
    # Encoder/model options
    parser.add_argument("--encoder", type=str, default="medvit", choices=["simple_cnn", "hybrid", "resnet3d", "timm_vit", "medvit"], help="Encoder backbone")
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[128,128,128], help="Input volume size D H W after resize")
    parser.add_argument("--vit_hidden", type=int, default=384, help="ViT hidden size")
    parser.add_argument("--patch_size", type=int, default=16, help="ViT patch size")
    parser.add_argument("--latent_dim", type=int, default=256, help="Latent dimension size")  # Fixed: Added missing latent_dim
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use for training/inference")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training/inference")
    
    # Inference arguments
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint for inference/benchmark")
    parser.add_argument("--input_volume", type=str, help="Path to input volume for inference")
    parser.add_argument("--input_phase", type=int, choices=[0, 1, 2, 3], help="Input phase (0: arterial, 1: venous, 2: delayed, 3: non-contrast)")
    parser.add_argument("--target_phase", type=int, choices=[0, 1, 2, 3], help="Target phase (0: arterial, 1: venous, 2: delayed, 3: non-contrast)")
    parser.add_argument("--output_path", type=str, help="Path to save generated volume")
    
    # Data preparation arguments
    parser.add_argument("--apply_registration", action="store_true", help="Apply registration to align volumes before training")
    parser.add_argument("--skip_prep", action="store_true", help="Skip dataset preparation and registration if cached data exists")

    # MedViT specific arguments
    parser.add_argument('--medvit_size', type=str, default='small', choices=['tiny', 'small', 'base'], help='MedViT model size')
    parser.add_argument('--medvit_pretrained_path', type=str, default='pretrained_medvit_small.pth', help='Path to pretrained MedViT weights')
    parser.add_argument('--aggregation_method', type=str, default='lstm', choices=['lstm', 'attention', 'mean', 'max'], help='Method to aggregate slice features')
    parser.add_argument('--slice_sampling', type=str, default='uniform', choices=['uniform', 'all'], help='How to sample slices from 3D volume')
    parser.add_argument('--max_slices', type=int, default=32, help='Maximum number of slices to process')
    parser.add_argument('--medvit_lr', type=float, default=1e-5, help='Learning rate for pretrained MedViT parameters')
    parser.add_argument('--use_scheduler', action='store_true', help='Use learning rate scheduler')
    
    # Timm ViT specific arguments
    parser.add_argument('--timm_model_name', type=str, default='vit_small_patch16_224', help='Timm ViT model name')
    parser.add_argument('--timm_pretrained', action='store_true', help='Use pretrained weights for Timm model')
    
    args = parser.parse_args()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    if args.mode == "train":
        # Prepare data
        print("Preparing data...")
        
        # Path to labels.csv
        labels_csv = os.path.join(args.data_path, "labels.csv")
        
        if not os.path.exists(labels_csv):
            print(f"Error: labels.csv not found at {labels_csv}")
            return
        
        # Prepare dataset from folders
        train_data_dicts, val_data_dicts = prepare_dataset_from_folders(
            args.data_path,
            labels_csv,
            validation_split=0.2,
            apply_registration=args.apply_registration,
            skip_prep=args.skip_prep
        )
        
        print(f"Training with {len(train_data_dicts)} samples, validating with {len(val_data_dicts)} samples")
        if args.apply_registration:
            print("Registration applied to align volumes")
        
        # Fixed: Correct spatial_size usage
        img_size = tuple(args.spatial_size)
        
        # Create data loaders
        train_loader = prepare_data(train_data_dicts, batch_size=args.batch_size, spatial_size=img_size)
        val_loader = prepare_data(val_data_dicts, batch_size=args.batch_size, augmentation=False, spatial_size=img_size)
        
        
        # Initialize encoder based on choice
        print(f"Initializing {args.encoder} encoder...")
        
        encoder_config_for_save = None
        
        if args.encoder == "medvit":
            # MedViT configuration
            medvit_config = {
                'model_size': args.medvit_size,
                'pretrained_path': args.medvit_pretrained_path,
                'latent_dim': args.latent_dim,
                'aggregation_method': args.aggregation_method,
                'slice_sampling': args.slice_sampling
            }
            
            # Check if pretrained weights exist
            if os.path.exists(args.medvit_pretrained_path):
                print(f"Using pretrained MedViT weights from {args.medvit_pretrained_path}")
            else:
                print(f"Warning: Pretrained weights not found at {args.medvit_pretrained_path}")
                print("Training MedViT from scratch...")
                medvit_config['pretrained_path'] = None
            
            encoder = create_medvit_encoder(medvit_config)
            encoder_config_for_save = {'type': 'medvit', 'config': medvit_config}
            
        elif args.encoder == "simple_cnn":
            # Simple 3D CNN configuration
            encoder = Simple3DCNNEncoder(
                in_channels=1,
                latent_dim=args.latent_dim,
                img_size=img_size
            )
            encoder_config_for_save = {
                'type': 'simple_cnn', 
                'config': {
                    'in_channels': 1,
                    'latent_dim': args.latent_dim,
                    'img_size': img_size
                }
            }
            
        elif args.encoder == "timm_vit":
            # timm ViT configuration
            try:
                encoder = TimmViTEncoder(
                    latent_dim=args.latent_dim,
                    model_name=args.timm_model_name,
                    pretrained=args.timm_pretrained,
                    max_slices=128,  # Good balance for large volumes
                    slice_sampling='adaptive' 
                )                
                encoder_config_for_save = {
                    'type': 'timm_vit',
                    'config': {
                        'latent_dim': args.latent_dim,
                        'model_name': args.timm_model_name,
                        'pretrained': args.timm_pretrained
                    }
                }
            except ImportError:
                print("Error: timm library not found. Install with: pip install timm")
                print("Falling back to simple_cnn encoder...")
                encoder = Simple3DCNNEncoder(in_channels=1, latent_dim=args.latent_dim, img_size=img_size)
                encoder_config_for_save = {'type': 'simple_cnn', 'config': {'latent_dim': args.latent_dim}}
                
        elif args.encoder == "resnet3d":
            # ResNet3D configuration
            encoder = ResNet3DEncoder(
                latent_dim=args.latent_dim,
                in_channels=1
            )
            encoder_config_for_save = {
                'type': 'resnet3d',
                'config': {
                    'latent_dim': args.latent_dim,
                    'in_channels': 1
                }
            }
            
        elif args.encoder == "hybrid":
            # Hybrid CNN-Transformer configuration
            encoder = LightweightHybridEncoder(
                latent_dim=args.latent_dim,
                in_channels=1
            )
            encoder_config_for_save = {
                'type': 'hybrid',
                'config': {
                    'latent_dim': args.latent_dim,
                    'in_channels': 1
                }
            }
            
        else:
            raise ValueError(f"Unknown encoder type: {args.encoder}")
            # print("Initializing standard ViT encoder...")
            # encoder = Encoder(
            #     img_size=img_size, 
            #     vit_hidden_size=args.vit_hidden, 
            #     latent_dim=args.latent_dim,  # Fixed: Use args.latent_dim instead of hardcoded 256
            #     patch_size=args.patch_size
            # )

        # debug_generator_dimensions(encoder, args)

        # Initialize other models with consistent latent dimensions
        print("Initializing Generator, Discriminator, and PhaseDetector...")
        generator = Generator(
            latent_dim=args.latent_dim, 
            phase_dim=32, 
            output_shape=(*img_size, 1)  # Fixed: Use img_size instead of hardcoded values
        )
        discriminator = Discriminator(input_shape=(*img_size, 1))  # Fixed: Use img_size
        phase_detector = PhaseDetector(latent_dim=args.latent_dim, num_phases=4)
        
        # # Prepare encoder config for saving (for MedViT compatibility)
        # encoder_config_for_save = None
        # if args.encoder == "medvit":
        #     encoder_config_for_save = medvit_config
        
        # Train
        print("Starting training...")
        metrics = train_contrast_phase_generation(
            train_loader,
            encoder,
            generator,
            discriminator,
            phase_detector,
            num_epochs=args.epochs,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            use_mixed_precision=args.mixed_precision,
            validation_loader=val_loader,
            encoder_config=encoder_config_for_save  # Pass encoder config for saving
        )
        
        print("Training complete!")
        
    elif args.mode == "inference":
        # Validate required arguments
        required_args = [
            ("checkpoint", args.checkpoint),
            ("input_volume", args.input_volume),
            ("input_phase", args.input_phase),
            ("target_phase", args.target_phase),
            ("output_path", args.output_path)
        ]
        
        for arg_name, arg_value in required_args:
            if arg_value is None:
                print(f"Error: --{arg_name} must be specified for inference mode")
                return
        
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        
        # Determine encoder type and create encoder
        # Determine encoder type and create encoder
        if 'encoder_config' in checkpoint and checkpoint['encoder_config'] is not None:
            encoder_info = checkpoint['encoder_config']
            encoder_type = encoder_info['type']
            encoder_config = encoder_info['config']
            
            print(f"Loading {encoder_type} encoder from checkpoint...")
            
            if encoder_type == 'medvit':
                encoder = create_medvit_encoder(encoder_config)
            elif encoder_type == 'simple_cnn':
                encoder = Simple3DCNNEncoder(**encoder_config)
            elif encoder_type == 'timm_vit':
                encoder = TimmViTEncoder(**encoder_config)
            elif encoder_type == 'resnet3d':
                encoder = ResNet3DEncoder(**encoder_config)
            elif encoder_type == 'hybrid':
                encoder = LightweightHybridEncoder(**encoder_config)
            else:
                raise ValueError(f"Unknown encoder type in checkpoint: {encoder_type}")
                
        else:
            # Fallback to simple CNN if no encoder config
            print("No encoder config found in checkpoint, using simple CNN...")
            img_size = tuple(args.spatial_size)
            encoder = Simple3DCNNEncoder(
                in_channels=1,
                latent_dim=args.latent_dim,
                img_size=img_size
            )
      
        
        # Initialize generator
        img_size = tuple(args.spatial_size)
        generator = Generator(
            latent_dim=args.latent_dim + 32,
            phase_dim=32, 
            output_shape=(*img_size, 1)
        )
        
        # Load model weights
        try:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            generator.load_state_dict(checkpoint['generator_state_dict'])
            print("Loaded encoder and generator from checkpoint")
        except KeyError:
            print("Error: Checkpoint format not recognized")
            return
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return
        
        # Move models to device
        encoder.to(args.device)
        generator.to(args.device)
        encoder.eval()
        generator.eval()
        
        # Load and preprocess input volume
        print(f"Loading input volume from {args.input_volume}")
        try:
            # Try MONAI first, then fallback to alternatives
            try:
                from monai.transforms import LoadImage, EnsureChannelFirst, ScaleIntensityRange, Resize
                import monai.transforms
                
                loader = LoadImage()
                input_volume = loader(args.input_volume)
                
                transform = monai.transforms.Compose([
                    EnsureChannelFirst(),
                    ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0),
                    Resize(spatial_size=args.spatial_size)
                ])
                input_volume = transform(input_volume).unsqueeze(0)
                
            except ImportError:
                # Fallback to nibabel + torch
                import nibabel as nib
                
                nii = nib.load(args.input_volume)
                input_volume = torch.from_numpy(nii.get_fdata()).float()
                
                # Add channel dimension and batch dimension
                if input_volume.dim() == 3:
                    input_volume = input_volume.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
                
                # Normalize
                input_volume = torch.clamp((input_volume + 1000) / 2000, 0, 1)
                
                # Resize if needed
                if input_volume.shape[2:] != tuple(args.spatial_size):
                    input_volume = torch.nn.functional.interpolate(
                        input_volume, size=args.spatial_size, mode='trilinear', align_corners=False
                    )
                
        except Exception as e:
            print(f"Error loading input volume: {e}")
            return
        
        # Generate contrast phase
        phase_names = ['arterial', 'venous', 'delayed', 'non-contrast']
        print(f"Generating {phase_names[args.target_phase]} phase from {phase_names[args.input_phase]} input")
        
        try:
            generated_volume = generate_contrast_phase(
                input_volume,
                args.input_phase,
                args.target_phase,
                encoder,
                generator,
                device=args.device,
                use_mixed_precision=args.mixed_precision
            )
            
            # Save output
            print(f"Saving generated volume to {args.output_path}")
            save_volume(generated_volume, args.output_path)
            print("Inference complete!")
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return
        
    elif args.mode == "benchmark":
        # Similar to inference but for benchmarking
        if not args.checkpoint:
            print("Error: --checkpoint must be specified for benchmark mode")
            return
        
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        
        # Load encoder (similar to inference)
        if 'encoder_config' in checkpoint and checkpoint['encoder_config'] is not None:
            encoder_info = checkpoint['encoder_config']
            encoder_type = encoder_info['type']
            encoder_config = encoder_info['config']
            
            if encoder_type == 'medvit':
                encoder = create_medvit_encoder(encoder_config)
            elif encoder_type == 'simple_cnn':
                encoder = Simple3DCNNEncoder(**encoder_config)
            elif encoder_type == 'timm_vit':
                encoder = TimmViTEncoder(**encoder_config)
            elif encoder_type == 'resnet3d':
                encoder = ResNet3DEncoder(**encoder_config)
            elif encoder_type == 'hybrid':
                encoder = LightweightHybridEncoder(**encoder_config)
        else:
            # Fallback
            img_size = tuple(args.spatial_size)
            encoder = Simple3DCNNEncoder(in_channels=1, latent_dim=args.latent_dim, img_size=img_size)
        
        # Initialize and load generator
        img_size = tuple(args.spatial_size)
        generator = Generator(
            latent_dim=args.latent_dim + 32,
            phase_dim=32, 
            output_shape=(*img_size, 1)
        )
        
        try:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            generator.load_state_dict(checkpoint['generator_state_dict'])
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return
        
        # Move models to device
        encoder.to(args.device)
        generator.to(args.device)
        
        # Run benchmark
        print("Running inference benchmark...")
        try:
            avg_time = benchmark_inference(encoder, generator, device=args.device)
            print(f"Benchmark complete! Average inference time: {avg_time*1000:.2f} ms ({1/avg_time:.2f} volumes/second)")
        except Exception as e:
            print(f"Error during benchmark: {e}")
            return

if __name__ == "__main__":
    main()