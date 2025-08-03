import torch
import argparse
import os
from models import Encoder, Generator, Discriminator, PhaseDetector
from data import prepare_data
from training import train_contrast_phase_generation
from inference import benchmark_inference, generate_contrast_phase, save_volume

def main():
    parser = argparse.ArgumentParser(description="CT Contrast Phase Generation Pipeline")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference", "benchmark"], 
                        help="Mode: train, inference, or benchmark")
    parser.add_argument("--data_path", type=str, default="data", help="Path to data directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for saving checkpoints")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use for training/inference")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training/inference")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint for inference/benchmark")
    parser.add_argument("--input_volume", type=str, help="Path to input volume for inference")
    parser.add_argument("--input_phase", type=int, choices=[0, 1, 2, 3], help="Input phase (0: arterial, 1: venous, 2: delayed, 3: non-contrast)")
    parser.add_argument("--target_phase", type=int, choices=[0, 1, 2, 3], help="Target phase (0: arterial, 1: venous, 2: delayed, 3: non-contrast)")
    parser.add_argument("--output_path", type=str, help="Path to save generated volume")
    
    args = parser.parse_args()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize models
    encoder = Encoder(input_shape=(128, 128, 128, 1), latent_dim=256)
    generator = Generator(latent_dim=256, phase_dim=32, output_shape=(128, 128, 128, 1))
    discriminator = Discriminator(input_shape=(128, 128, 128, 1))
    phase_detector = PhaseDetector(latent_dim=256, num_phases=4)
    
    # Replace the data preparation section in the train mode (around line 38)
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
            validation_split=0.2
        )
        
        # Create data loaders
        train_loader = prepare_data(train_data_dicts, batch_size=args.batch_size)
        val_loader = prepare_data(val_data_dicts, batch_size=args.batch_size, augmentation=False)
        
        # Try to load pre-trained MedViT weights if available
        medvit_path = "pretrained_medvit.pth"
        if os.path.exists(medvit_path):
            print(f"Loading pre-trained MedViT weights from {medvit_path}")
            encoder.vit.load_state_dict(torch.load(medvit_path))
        else:
            print("No pre-trained MedViT weights found. Training from scratch.")
        
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
            validation_loader=val_loader  # Pass validation loader
        )
        
        print("Training complete!")
        
    elif args.mode == "inference":
        # Load checkpoint
        if not args.checkpoint:
            print("Error: --checkpoint must be specified for inference mode")
            return
        if not args.input_volume:
            print("Error: --input_volume must be specified for inference mode")
            return
        if args.input_phase is None:
            print("Error: --input_phase must be specified for inference mode")
            return
        if args.target_phase is None:
            print("Error: --target_phase must be specified for inference mode")
            return
        if not args.output_path:
            print("Error: --output_path must be specified for inference mode")
            return
        
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        
        # Load model weights
        if 'encoder_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            generator.load_state_dict(checkpoint['generator_state_dict'])
        else:
            # Assume it's just the encoder weights
            encoder.load_state_dict(checkpoint)
            generator_path = args.checkpoint.replace('encoder', 'generator')
            if os.path.exists(generator_path):
                generator.load_state_dict(torch.load(generator_path, map_location=args.device))
            else:
                print(f"Warning: Generator weights not found at {generator_path}")
        
        # Load input volume
        from monai.transforms import LoadImage, EnsureChannelFirst, ScaleIntensityRange, Resize
        import nibabel as nib
        
        print(f"Loading input volume from {args.input_volume}")
        transform = LoadImage()
        input_volume = transform(args.input_volume)
        
        # Preprocess
        transform = monai.transforms.Compose([
            EnsureChannelFirst(),
            ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0),
            Resize(spatial_size=(128, 128, 128))
        ])
        input_volume = transform(input_volume).unsqueeze(0)  # Add batch dimension
        
        # Generate contrast phase
        print(f"Generating {['arterial', 'venous', 'delayed', 'non-contrast'][args.target_phase]} phase from {['arterial', 'venous', 'delayed', 'non-contrast'][args.input_phase]} input")
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
        
    elif args.mode == "benchmark":
        # Load checkpoint
        if not args.checkpoint:
            print("Error: --checkpoint must be specified for benchmark mode")
            return
        
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        
        # Load model weights
        if 'encoder_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            generator.load_state_dict(checkpoint['generator_state_dict'])
        else:
            # Assume it's just the encoder weights
            encoder.load_state_dict(checkpoint)
            generator_path = args.checkpoint.replace('encoder', 'generator')
            if os.path.exists(generator_path):
                generator.load_state_dict(torch.load(generator_path, map_location=args.device))
            else:
                print(f"Warning: Generator weights not found at {generator_path}")
        
        # Run benchmark
        print("Running inference benchmark...")
        avg_time = benchmark_inference(encoder, generator, device=args.device)
        
        print(f"Benchmark complete! Average inference time: {avg_time*1000:.2f} ms ({1/avg_time:.2f} volumes/second)")

if __name__ == "__main__":
    main()