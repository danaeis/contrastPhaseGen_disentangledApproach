import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from utils import get_phase_embedding, save_volume
import time

def optimize_models_for_inference(encoder, generator, device="cuda"):
    """Optimize models for inference using torch.jit.trace."""
    # Set models to evaluation mode
    encoder.eval()
    generator.eval()
    
    # Create example inputs
    dummy_input = torch.randn(1, 1, 128, 128, 128, device=device)
    dummy_z = encoder(dummy_input)
    dummy_phase_emb = get_phase_embedding(0, dim=32).unsqueeze(0).to(device)
    
    # Trace models
    traced_encoder = torch.jit.trace(encoder, dummy_input)
    traced_generator = torch.jit.trace(generator, (dummy_z, dummy_phase_emb))
    
    return traced_encoder, traced_generator

def generate_contrast_phase(input_volume, input_phase, target_phase, encoder, generator, device="cuda", use_mixed_precision=True):
    """Generate a contrast phase from input volume.
    
    Args:
        input_volume: Input CT volume tensor (1, 1, 128, 128, 128)
        input_phase: Integer representing input phase (0-3)
        target_phase: Integer representing target phase (0-3)
        encoder: Encoder model
        generator: Generator model
        device: Device to run inference on
        use_mixed_precision: Whether to use mixed precision inference
    
    Returns:
        generated_volume: Generated CT volume in target phase
    """
    # Move input to device
    input_volume = input_volume.to(device)
    
    # Set models to evaluation mode
    encoder.eval()
    generator.eval()
    
    # Inference
    with torch.no_grad(), autocast(enabled=use_mixed_precision):
        # Encode input volume
        z = encoder(input_volume)
        
        # Get phase embedding for target phase
        phase_emb = get_phase_embedding(target_phase, dim=32).unsqueeze(0).to(device)
        
        # Generate target volume
        generated_volume = generator(z, phase_emb)
    
    return generated_volume

def batch_inference(input_volumes, input_phases, target_phases, encoder, generator, batch_size=4, device="cuda"):
    """Run inference on multiple volumes in batches.
    
    Args:
        input_volumes: List of input volume tensors
        input_phases: List of input phase integers
        target_phases: List of target phase integers
        encoder: Encoder model
        generator: Generator model
        batch_size: Batch size for inference
        device: Device to run inference on
    
    Returns:
        generated_volumes: List of generated volumes
    """
    # Optimize models for inference
    traced_encoder, traced_generator = optimize_models_for_inference(encoder, generator, device)
    
    # Set models to evaluation mode
    traced_encoder.eval()
    traced_generator.eval()
    
    # Prepare output list
    generated_volumes = []
    
    # Process in batches
    for i in range(0, len(input_volumes), batch_size):
        batch_inputs = input_volumes[i:i+batch_size]
        batch_target_phases = target_phases[i:i+batch_size]
        
        # Stack inputs into a batch
        batch_input_tensor = torch.stack(batch_inputs).to(device)
        
        # Inference
        with torch.no_grad(), autocast():
            # Encode batch
            batch_z = traced_encoder(batch_input_tensor)
            
            # Process each item in batch with its target phase
            for j, (z, target_phase) in enumerate(zip(batch_z, batch_target_phases)):
                # Get phase embedding
                phase_emb = get_phase_embedding(target_phase, dim=32).unsqueeze(0).to(device)
                
                # Generate volume
                z_unsqueezed = z.unsqueeze(0)  # Add batch dimension
                generated_volume = traced_generator(z_unsqueezed, phase_emb)
                
                # Add to results
                generated_volumes.append(generated_volume.squeeze(0))
    
    return generated_volumes

def benchmark_inference(encoder, generator, device="cuda", num_runs=100):
    """Benchmark inference speed."""
    # Optimize models
    traced_encoder, traced_generator = optimize_models_for_inference(encoder, generator, device)
    
    # Create dummy input
    dummy_input = torch.randn(1, 1, 128, 128, 128, device=device)
    dummy_phase = 1  # Venous phase
    
    # Warmup
    for _ in range(10):
        with torch.no_grad(), autocast():
            z = traced_encoder(dummy_input)
            phase_emb = get_phase_embedding(dummy_phase, dim=32).unsqueeze(0).to(device)
            _ = traced_generator(z, phase_emb)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad(), autocast():
            z = traced_encoder(dummy_input)
            phase_emb = get_phase_embedding(dummy_phase, dim=32).unsqueeze(0).to(device)
            _ = traced_generator(z, phase_emb)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    print(f"Average inference time: {avg_time*1000:.2f} ms ({1/avg_time:.2f} volumes/second)")
    
    return avg_time