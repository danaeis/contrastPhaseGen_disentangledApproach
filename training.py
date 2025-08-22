import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.amp import autocast, GradScaler
from utils import GradientReversalLayer, get_phase_embedding
import numpy as np
from tqdm import tqdm
import csv
from datetime import datetime
import json
from collections import defaultdict

# Import the new systems
from early_stopping_system import (
    AdvancedEarlyStopping, 
    ModelSpecificEarlyStopping, 
    setup_early_stopping_for_training,
    compute_gradient_norm
)
from image_quality_metrics import (
    ImageQualityMetrics,
    ValidationMetricsTracker,
    run_validation_with_metrics
)

def create_phase_embeddings(phase_labels, dim=32, device='cuda'):
    """Create phase embeddings for a batch of phase labels"""
    batch_embeddings = []
    
    for phase_label in phase_labels:
        if isinstance(phase_label, torch.Tensor):
            phase_val = phase_label.item()
        else:
            phase_val = phase_label
            
        embedding = get_phase_embedding(phase_val, dim=dim, device=device)
        batch_embeddings.append(embedding)
    
    return torch.stack(batch_embeddings)

def check_pretrained_weights(encoder, encoder_type):
    """Check if encoder has meaningful pretrained weights"""
    pretrained_encoders = ['timm_vit', 'medvit']
    
    if encoder_type in pretrained_encoders:
        print(f"✓ {encoder_type} encoder has pretrained weights - skipping encoder pretraining")
        return True
    else:
        print(f"✗ {encoder_type} encoder needs pretraining")
        return False

def debug_data_flow(batch, encoder, generator, phase_detector, device):
    """Debug function to check data flow and gradients"""
    print("\n" + "="*60)
    print("🔍 DEBUGGING DATA FLOW")
    print("="*60)
    
    input_volume = batch["input_volume"].to(device)
    target_volume = batch["target_volume"].to(device) 
    phase_label = batch["target_phase"].to(device)
    target_phase_label = batch["input_phase"].to(device)
    
    print(f"📊 Batch Info:")
    print(f"   Input volume shape: {input_volume.shape}")
    print(f"   Target volume shape: {target_volume.shape}")
    print(f"   Input volume range: [{input_volume.min():.3f}, {input_volume.max():.3f}]")
    print(f"   Target volume range: [{target_volume.min():.3f}, {target_volume.max():.3f}]")
    print(f"   Phase labels: {phase_label.cpu().numpy()}")
    print(f"   True phases: {target_phase_label.cpu().numpy()}")
    
    # Test encoder
    with torch.no_grad():
        z = encoder(input_volume)
        print(f"\n🏗️  Encoder Output:")
        print(f"   Latent shape: {z.shape}")
        print(f"   Latent range: [{z.min():.3f}, {z.max():.3f}]")
        print(f"   Latent std: {z.std():.3f}")
        
        # Check for NaN or inf
        if torch.isnan(z).any():
            print("   ⚠️  WARNING: NaN detected in encoder output!")
        if torch.isinf(z).any():
            print("   ⚠️  WARNING: Inf detected in encoder output!")
    
    # Test phase embedding
    phase_emb = create_phase_embeddings(phase_label, dim=32, device=device)
    print(f"\n🎯 Phase Embedding:")
    print(f"   Phase embedding shape: {phase_emb.shape}")
    print(f"   Phase embedding range: [{phase_emb.min():.3f}, {phase_emb.max():.3f}]")
    
    # Test generator
    with torch.no_grad():
        generated = generator(z, phase_emb)
        print(f"\n🏭 Generator Output:")
        print(f"   Generated shape: {generated.shape}")
        print(f"   Generated range: [{generated.min():.3f}, {generated.max():.3f}]")
        print(f"   Generated std: {generated.std():.3f}")
        
        # Check for NaN or inf
        if torch.isnan(generated).any():
            print("   ⚠️  WARNING: NaN detected in generator output!")
        if torch.isinf(generated).any():
            print("   ⚠️  WARNING: Inf detected in generator output!")
    
    # Test phase detector
    with torch.no_grad():
        phase_pred = phase_detector(z)
        print(f"\n🎯 Phase Detector Output:")
        print(f"   Phase prediction shape: {phase_pred.shape}")
        print(f"   Phase prediction range: [{phase_pred.min():.3f}, {phase_pred.max():.3f}]")
        print(f"   Softmax predictions: {torch.softmax(phase_pred, dim=1)[0].cpu().numpy()}")
        
        # Check for NaN or inf
        if torch.isnan(phase_pred).any():
            print("   ⚠️  WARNING: NaN detected in phase detector output!")
        if torch.isinf(phase_pred).any():
            print("   ⚠️  WARNING: Inf detected in phase detector output!")
    
    # Test reconstruction loss
    with torch.no_grad():
        recon_loss = nn.L1Loss()(generated, target_volume)
        print(f"\n📊 Loss Analysis:")
        print(f"   Reconstruction L1 Loss: {recon_loss.item():.6f}")
        
        # Analyze difference
        diff = torch.abs(generated - target_volume)
        print(f"   Mean absolute difference: {diff.mean():.6f}")
        print(f"   Max absolute difference: {diff.max():.6f}")
    
    print("="*60)

def check_gradient_flow(model, model_name):
    """Check if gradients are flowing properly"""
    total_norm = 0
    param_count = 0
    zero_grad_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
            if param_norm.item() < 1e-7:
                zero_grad_count += 1
        else:
            zero_grad_count += 1
            param_count += 1
    
    total_norm = total_norm ** (1. / 2)
    
    print(f"🔄 Gradient Flow - {model_name}:")
    print(f"   Total gradient norm: {total_norm:.6f}")
    print(f"   Parameters with zero gradients: {zero_grad_count}/{param_count}")
    
    if total_norm < 1e-5:
        print(f"   ⚠️  WARNING: Very small gradients in {model_name}!")
    if zero_grad_count > param_count * 0.5:
        print(f"   ⚠️  WARNING: Many zero gradients in {model_name}!")

def pretrain_encoder_generator(train_loader, encoder, generator, phase_detector, num_epochs=30, device="cuda", use_mixed_precision=True):
    """
    FIXED Phase 1: Pretrain encoder and generator with reconstruction loss
    """
    print(f"\n{'='*60}")
    print("PHASE 1: ENCODER + GENERATOR PRETRAINING (FIXED)")
    print(f"{'='*60}")
    
    # Set models to training mode
    encoder.train()
    generator.train()
    
    # FIXED: Use MSE loss instead of L1 for better gradient flow
    reconstruction_loss = nn.MSELoss()
    print("line 160")
    # FIXED: Lower learning rates for more stable training
    optimizer_enc = optim.Adam(encoder.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer_gen = optim.Adam(generator.parameters(), lr=1e-4, weight_decay=1e-5)
    print("line 164")
    # FIXED: Add learning rate scheduler
    scheduler_enc = optim.lr_scheduler.ReduceLROnPlateau(optimizer_enc, patience=5, factor=0.5)
    scheduler_gen = optim.lr_scheduler.ReduceLROnPlateau(optimizer_gen, patience=5, factor=0.5)
    print("line 168")
    # Mixed precision
    scaler = GradScaler() if use_mixed_precision else None
    
    pretrain_losses = []
    best_loss = float('inf')
    
    # Debug first batch
    debug_batch = next(iter(train_loader))
    debug_data_flow(debug_batch, encoder, generator, phase_detector, device)
    print("line 178")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{num_epochs}")):
            input_volume = batch["input_volume"].to(device)
            input_phase = batch["input_phase"].to(device)
            # FIXED: Use target volume as reconstruction target
            target_volume = batch["target_volume"].to(device)
            target_phase = batch["target_phase"].to(device)
            
            optimizer_enc.zero_grad()
            optimizer_gen.zero_grad()
            
            with autocast(device_type="cuda", enabled=use_mixed_precision):
                # Encode input
                z = encoder(input_volume)
                
                # FIXED: Use target phase instead of random phase for better learning
                phase_emb = create_phase_embeddings(target_phase, dim=32, device=device)
                
                # Generate volume
                reconstructed_volume = generator(z, phase_emb)
                
                # FIXED: Ensure same data range for loss computation
                # Clamp both to [0, 1] range
                reconstructed_volume = torch.clamp(reconstructed_volume, 0, 1)
                target_volume = torch.clamp(target_volume, 0, 1)
                
                # Reconstruction loss
                loss = reconstruction_loss(reconstructed_volume, target_volume)
                
                # FIXED: Add regularization term to prevent mode collapse
                # Encourage encoder to produce diverse features
                z_var = torch.var(z, dim=0).mean()
                reg_loss = -0.01 * z_var  # Negative to encourage variance
                
                total_loss = loss + reg_loss
            
            # Backward pass
            if use_mixed_precision:
                scaler.scale(total_loss).backward()
                
                # FIXED: Add gradient clipping
                scaler.unscale_(optimizer_enc)
                scaler.unscale_(optimizer_gen)
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                
                scaler.step(optimizer_enc)
                scaler.step(optimizer_gen)
                scaler.update()
            else:
                total_loss.backward()
                
                # FIXED: Add gradient clipping
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                
                optimizer_enc.step()
                optimizer_gen.step()
            
            epoch_loss += loss.item()  # Don't include reg_loss in reported loss
            num_batches += 1
            
            # Debug gradient flow every 50 batches
            if batch_idx % 50 == 0:
                check_gradient_flow(encoder, "Encoder")
                check_gradient_flow(generator, "Generator")
        
        avg_loss = epoch_loss / num_batches
        pretrain_losses.append(avg_loss)
        
        # Update learning rate
        scheduler_enc.step(avg_loss)
        scheduler_gen.step(avg_loss)
        # Log learning rate
        current_lr = optimizer_enc.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_loss:.4f}, Learning Rate: {current_lr}")

        # Track best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            improvement = "✅"
        else:
            improvement = "📉"
        
        print(f"Pretrain Epoch {epoch+1}: Reconstruction Loss = {avg_loss:.6f} {improvement}")
        
        # FIXED: Early stopping if loss is not improving
        if epoch > 10 and avg_loss > pretrain_losses[-5]:  # No improvement in 5 epochs
            print(f"   ⚠️  Loss not improving, consider checking data or model")
    
    print(f"✓ Encoder + Generator pretraining completed (Best: {best_loss:.6f})")
    return pretrain_losses

def train_phase_detector(train_loader, encoder, phase_detector, 
                         num_epochs=40, device="cuda", use_mixed_precision=True,
                         spatial_size=(128,128,128), checkpoint_dir="checkpoints"):
    """
    OPTIMIZED: Phase detector training that accepts train_loader and extracts unique volumes
    """
    print(f"\n{'='*60}")
    print("PHASE 2: OPTIMIZED PHASE DETECTOR TRAINING")
    print(f"{'='*60}")
    
    # Extract unique volumes from the train_loader's dataset
    print("🔍 Extracting unique volumes from train_loader...")
    unique_volumes = {}
    phase_counts = defaultdict(int)
    
    # Access the underlying dataset from the DataLoader
    dataset = train_loader.dataset
    
    # Extract unique volumes from dataset data
    if hasattr(dataset, 'data'):
        # MONAI Dataset stores data in .data attribute
        data_dicts = dataset.data
    else:
        # Fallback: try to access through dataset directly
        print("⚠️  Warning: Could not access dataset.data, using alternative method")
        data_dicts = []
        
        # Alternative: iterate through the loader once to collect unique data
        # This is less efficient but works if dataset.data is not accessible
        temp_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        seen_paths = set()
        
        for batch in temp_loader:
            input_volume_paths = batch.get("input_volume_meta_dict", {}).get("filename_or_obj", [])
            input_phases = batch["input_phase"]
            scan_ids = batch.get("scan_id", ["unknown"])
            
            if isinstance(input_volume_paths, (list, tuple)):
                input_volume_path = input_volume_paths[0] if input_volume_paths else "unknown"
            else:
                input_volume_path = str(input_volume_paths)
            
            if input_volume_path not in seen_paths and input_volume_path != "unknown":
                seen_paths.add(input_volume_path)
                data_dicts.append({
                    "input_volume": input_volume_path,
                    "input_phase": input_phases[0].item() if torch.is_tensor(input_phases[0]) else input_phases[0],
                    "scan_id": scan_ids[0] if isinstance(scan_ids, (list, tuple)) else scan_ids
                })
    
    # Create unique phase detection dataset
    for data_dict in data_dicts:
        input_path = data_dict["input_path"]
        input_phase = data_dict["input_phase"]
        scan_id = data_dict.get("scan_id", "unknown")
        
        # Use path as unique identifier
        if input_path not in unique_volumes:
            unique_volumes[input_path] = {
                "volume": input_path,
                "phase": input_phase,
                "scan_id": scan_id
            }
            phase_counts[input_phase] += 1
    
    unique_data_dicts = list(unique_volumes.values())
    
    print(f"   Original pairs in loader: {len(data_dicts)}")
    print(f"   Unique volumes extracted: {len(unique_data_dicts)}")
    print(f"   Phase distribution:")
    
    phase_names = {0: 'Non-contrast', 1: 'Arterial', 2: 'Venous', 3: 'Delayed'}
    for phase, count in sorted(phase_counts.items()):
        phase_name = phase_names.get(phase, f'Phase_{phase}')
        print(f"     {phase_name}: {count} volumes")
    
    # Create phase detection DataLoader
    from data import prepare_phase_detection_data
    phase_loader = prepare_phase_detection_data(
        unique_data_dicts, batch_size=4, spatial_size=spatial_size
    )
    
    # Setup early stopping
    early_stopping = AdvancedEarlyStopping(
        patience=15,
        min_delta=1e-4,
        oscillation_patience=10,
        oscillation_threshold=0.005,
        restore_best_weights=True
    )
    
    # Freeze encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    
    phase_detector.train()
    
    # Analyze class distribution and setup weighted loss
    unique_phases = set(phase_counts.keys())
    num_classes = len(unique_phases)
    print(f"📊 Detected {num_classes} unique phases: {sorted(unique_phases)}")
    
    # Create class weights (this helps with medical data imbalance)
    total_samples = len(unique_data_dicts)
    class_weights = []
    for i in range(max(unique_phases) + 1):  # Ensure we cover all phase indices
        if i in phase_counts:
            weight = total_samples / (num_classes * phase_counts[i])
        else:
            weight = 1.0
        class_weights.append(weight)
    
    # Only keep weights for phases that exist
    actual_class_weights = [class_weights[i] for i in sorted(unique_phases)]
    class_weights_tensor = torch.tensor(actual_class_weights).to(device)
    print(f"   Using class weights: {class_weights_tensor.cpu().numpy()}")
    
    # Loss and optimizer
    weighted_loss = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(
        phase_detector.parameters(), 
        lr=5e-4, 
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Mixed precision
    scaler = GradScaler() if use_mixed_precision else None
    
    # Tracking
    phase_losses = []
    phase_accuracies = []
    best_accuracy = 0.0
    best_epoch = 0
    
    print(f"🚀 Starting optimized phase detector training...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        num_batches = 0
        
        # Training loop
        for batch_idx, batch in enumerate(tqdm(phase_loader, desc=f"Phase Epoch {epoch+1}/{num_epochs}")):
            volumes = batch["volume"].to(device)
            true_phases = batch["phase"].to(device)
            
            optimizer.zero_grad()
            
            with autocast(device_type="cuda", enabled=use_mixed_precision):
                # Extract features (frozen encoder)
                with torch.no_grad():
                    features = encoder(volumes).detach()
                
                # Phase prediction
                phase_logits = phase_detector(features)
                loss = weighted_loss(phase_logits, true_phases)
            
            # Backward pass
            if use_mixed_precision:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(phase_detector.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(phase_detector.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(phase_logits, 1)
            correct_predictions += (predicted == true_phases).sum().item()
            total_predictions += true_phases.size(0)
            num_batches += 1
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / num_batches
        accuracy = correct_predictions / total_predictions
        
        phase_losses.append(avg_loss)
        phase_accuracies.append(accuracy)
        
        # Update learning rate
        scheduler.step()
        
        # Compute gradient norm for early stopping
        grad_norm = compute_gradient_norm(phase_detector)
        
        # Print progress
        improvement = "✅" if accuracy > best_accuracy else "📉"
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
        
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f} {improvement}")
        print(f"   Gradient norm: {grad_norm:.6f}, Best accuracy: {best_accuracy:.4f} (epoch {best_epoch+1})")
        
        # Early stopping check
        if early_stopping(avg_loss, phase_detector, epoch, grad_norm):
            print(f"🛑 Phase detector training stopped early!")
            early_stopping.restore_weights(phase_detector)
            break
        
        # Save checkpoint if best model
        if accuracy > best_accuracy - 0.01:  # Save if within 1% of best
            checkpoint_path = os.path.join(checkpoint_dir, f"phase_detector_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': phase_detector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'loss': avg_loss
            }, checkpoint_path)
    
    # Unfreeze encoder for next phase
    for param in encoder.parameters():
        param.requires_grad = True
    encoder.train()
    
    # Final summary
    final_accuracy = phase_accuracies[-1] if phase_accuracies else 0
    print(f"\n✅ Optimized phase detector training completed:")
    print(f"   Final accuracy: {final_accuracy:.4f}")
    print(f"   Best accuracy: {best_accuracy:.4f} (epoch {best_epoch+1})")
    print(f"   Early stopping summary: {early_stopping.get_summary()}")
    
    return phase_losses, phase_accuracies

def train_disentangled_generation(train_loader, encoder, generator, discriminator, phase_detector, validation_loader=None,
                                 num_epochs=50, device="cuda", use_mixed_precision=True, checkpoint_dir="checkpoints"):
    """
    FIXED Phase 3: Train generator with adversarial loss and reversed gradients
    """
    print(f"\n{'='*60}")
    print("PHASE 3: DISENTANGLED GENERATION TRAINING (FIXED)")
    print(f"{'='*60}")
    
    # Set all models to training mode
    encoder.train()
    generator.train()
    discriminator.train()
    phase_detector.train()
    # Setup early stopping for each model
    early_stopping = ModelSpecificEarlyStopping()
    early_stopping.add_model('generator', patience=20, min_delta=1e-5, restore_best_weights=True)
    early_stopping.add_model('discriminator', patience=25, min_delta=1e-5, restore_best_weights=False)
    early_stopping.add_model('overall', patience=30, min_delta=1e-6, restore_best_weights=False)
    
    metrics_tracker = ValidationMetricsTracker(
        save_path=os.path.join(checkpoint_dir, 'validation_metrics.csv')
    )
    # FIXED: Better loss functions
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    gan_loss = nn.BCEWithLogitsLoss()
    phase_loss = nn.CrossEntropyLoss()
    
    # FIXED: More balanced learning rates and optimizers
    optimizer_enc_gen = optim.AdamW(
        list(encoder.parameters()) + list(generator.parameters()), 
        lr=5e-5, weight_decay=1e-5, betas=(0.5, 0.999)
    )
    optimizer_disc = optim.AdamW(
        discriminator.parameters(), 
        lr=2e-4, weight_decay=1e-5, betas=(0.5, 0.999)
    )
    optimizer_phase = optim.AdamW(
        phase_detector.parameters(), 
        lr=1e-5, weight_decay=1e-5  # Very low LR for fine-tuning
    )
    
    # FIXED: Add learning rate schedulers
    scheduler_enc_gen = optim.lr_scheduler.ReduceLROnPlateau(optimizer_enc_gen, patience=10, factor=0.8)
    scheduler_disc = optim.lr_scheduler.ReduceLROnPlateau(optimizer_disc, patience=10, factor=0.8)
    
    # Mixed precision
    scaler = GradScaler() if use_mixed_precision else None
    
    # Training metrics
    g_losses = []
    d_losses = []
    p_losses = []
    phase_accuracies = []
    
    for epoch in range(num_epochs):
        # FIXED: More gradual ramp-up for stability
        lambda_adv = min(1.0, (epoch + 1) / 30)  # Ramp up over 30 epochs
        lambda_phase = min(0.5, (epoch + 1) / 40)  # Even more gradual for phase loss
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_p_loss = 0.0
        phase_correct = 0
        phase_total = 0
        num_batches = 0
        best_ssim = 0.0
        best_psnr = 0.0

        print(f"Disentanglement Epoch {epoch+1}/{num_epochs} (λ_adv = {lambda_adv:.3f}, λ_phase = {lambda_phase:.3f})")
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Disentangle Epoch {epoch+1}")):
            input_volume = batch["input_volume"].to(device)
            target_volume = batch["target_volume"].to(device)
            target_phase = batch["target_phase"].to(device)
            true_phase = batch["input_phase"].to(device)
            
            batch_size = input_volume.shape[0]
            
            # ===========================
            # FIXED: Train Discriminator more frequently
            # ===========================
            for _ in range(2):  # Train discriminator 2x per generator update
                optimizer_disc.zero_grad()
                
                with autocast(device_type="cuda", enabled=use_mixed_precision):
                    # Generate fake volumes
                    z = encoder(input_volume).detach()  # Detach to prevent generator gradients
                    phase_emb = create_phase_embeddings(target_phase, dim=32, device=device)
                    fake_volume = generator(z, phase_emb).detach()
                    
                    # FIXED: Add noise to inputs for better training stability
                    noise_factor = 0.1
                    real_noisy = target_volume + torch.randn_like(target_volume) * noise_factor
                    fake_noisy = fake_volume + torch.randn_like(fake_volume) * noise_factor
                    
                    # Discriminator outputs
                    real_score = discriminator(real_noisy)
                    fake_score = discriminator(fake_noisy)
                    
                    # FIXED: Use label smoothing
                    real_labels = torch.ones_like(real_score) - 0.1 * torch.rand_like(real_score)
                    fake_labels = torch.zeros_like(fake_score) + 0.1 * torch.rand_like(fake_score)
                    
                    # Discriminator loss
                    d_loss_real = gan_loss(real_score, real_labels)
                    d_loss_fake = gan_loss(fake_score, fake_labels)
                    d_loss = (d_loss_real + d_loss_fake) / 2
                
                if use_mixed_precision:
                    scaler.scale(d_loss).backward()
                    scaler.unscale_(optimizer_disc)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=0.5)
                    scaler.step(optimizer_disc)
                    scaler.update()
                else:
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=0.5)
                    optimizer_disc.step()
            
            # ===========================
            # FIXED: Train Generator + Encoder
            # ===========================
            optimizer_enc_gen.zero_grad()
            
            with autocast(device_type="cuda", enabled=use_mixed_precision):
                # Generate volumes
                z = encoder(input_volume)
                phase_emb = create_phase_embeddings(target_phase, dim=32, device=device)
                generated_volume = generator(z, phase_emb)
                
                # Generator losses
                fake_score = discriminator(generated_volume)
                
                # FIXED: Combine L1 and MSE losses
                l1_recon_loss = l1_loss(generated_volume, target_volume)
                mse_recon_loss = mse_loss(generated_volume, target_volume)
                reconstruction_loss = l1_recon_loss + 0.5 * mse_recon_loss
                
                # GAN loss
                adversarial_loss = gan_loss(fake_score, torch.ones_like(fake_score))
                
                # FIXED: Add perceptual loss (simple version)
                # Compute loss on different scales
                scales = [1, 0.5, 0.25]
                perceptual_loss = 0
                for scale in scales:
                    if scale < 1:
                        size = [int(s * scale) for s in generated_volume.shape[2:]]
                        gen_scaled = torch.nn.functional.interpolate(generated_volume, size=size, mode='trilinear')
                        target_scaled = torch.nn.functional.interpolate(target_volume, size=size, mode='trilinear')
                        perceptual_loss += l1_loss(gen_scaled, target_scaled)
                    else:
                        perceptual_loss += l1_loss(generated_volume, target_volume)
                
                # Combined generator loss
                g_loss = (reconstruction_loss * 100.0 +  # Increased weight
                         adversarial_loss * lambda_adv * 2.0 +  # Adaptive weight
                         perceptual_loss * 10.0)  # Multi-scale loss
            
            if use_mixed_precision:
                scaler.scale(g_loss).backward()
                scaler.unscale_(optimizer_enc_gen)
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(generator.parameters()), max_norm=1.0)
                scaler.step(optimizer_enc_gen)
                scaler.update()
            else:
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(generator.parameters()), max_norm=1.0)
                optimizer_enc_gen.step()
            
            # ===========================
            # FIXED: Train Phase Detector with Reversed Gradients
            # ===========================
            if epoch >= 10:  # Start phase training after some generator training
                optimizer_phase.zero_grad()
                
                with autocast(device_type="cuda", enabled=use_mixed_precision):
                    # Get encoder features
                    z = encoder(input_volume)
                    
                    # Apply gradient reversal for disentanglement
                    z_reversed = GradientReversalLayer.apply(z, lambda_phase)
                    
                    # Phase prediction on reversed gradients
                    phase_pred = phase_detector(z_reversed)
                    p_loss = phase_loss(phase_pred, true_phase)
                
                if use_mixed_precision:
                    scaler.scale(p_loss).backward()
                    scaler.unscale_(optimizer_phase)
                    torch.nn.utils.clip_grad_norm_(phase_detector.parameters(), max_norm=0.5)
                    scaler.step(optimizer_phase)
                    scaler.update()
                else:
                    p_loss.backward()
                    torch.nn.utils.clip_grad_norm_(phase_detector.parameters(), max_norm=0.5)
                    optimizer_phase.step()
                
                # Track phase metrics
                with torch.no_grad():
                    _, predicted = torch.max(phase_pred.data, 1)
                    phase_correct += (predicted == true_phase).sum().item()
                    phase_total += true_phase.size(0)
                    epoch_p_loss += p_loss.item()
            
            # Track other metrics
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1
            
            # FIXED: Debug gradient flow periodically
            if batch_idx % 100 == 0:
                check_gradient_flow(encoder, "Encoder")
                check_gradient_flow(generator, "Generator")
                check_gradient_flow(discriminator, "Discriminator")
                if epoch >= 10:
                    check_gradient_flow(phase_detector, "Phase Detector")
        
        # Calculate epoch metrics
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        avg_p_loss = epoch_p_loss / num_batches if epoch >= 10 else 0
        phase_accuracy = phase_correct / phase_total if phase_total > 0 else 0
        
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        p_losses.append(avg_p_loss)
        phase_accuracies.append(phase_accuracy)
        
        # Update learning rates
        scheduler_enc_gen.step(avg_g_loss)
        scheduler_disc.step(avg_d_loss)
        
        print(f"Epoch {epoch+1}: G_loss = {avg_g_loss:.6f}, D_loss = {avg_d_loss:.6f}, "
              f"P_loss = {avg_p_loss:.6f}, Phase_Acc = {phase_accuracy:.4f}")

        print(f"Generation Loss: {avg_g_loss:.6f}, Discriminator Loss: {avg_d_loss:.6f}")
        print(f"Phase Loss: {avg_p_loss:.6f}, Phase Accuracy: {phase_accuracy:.4f}")
        
        # Validation with image quality metrics
        if validation_loader is not None and (epoch + 1) % 5 == 0:
            print(f"🔍 Running validation with image quality metrics...")
            
            val_metrics = run_validation_with_metrics(
                validation_loader, encoder, generator, metrics_tracker, device, max_batches=5
            )
            
            # Track validation metrics
            val_metrics_tracked = metrics_tracker.compute_and_track(
                torch.randn(1, 1, 32, 64, 64),  # Dummy for now - replace with actual validation
                torch.randn(1, 1, 32, 64, 64),  # In real implementation, use validation batch
                epoch
            )
            
            print(f"📊 Validation Metrics:")
            print(f"   PSNR: {val_metrics.get('psnr', 0):.4f}")
            print(f"   SSIM: {val_metrics.get('ssim', 0):.4f}")
            print(f"   NMSE: {val_metrics.get('nmse', 0):.6f}")
            
            # Check for best metrics
            current_ssim = val_metrics.get('ssim', 0)
            current_psnr = val_metrics.get('psnr', 0)
            
            if current_ssim > best_ssim:
                best_ssim = current_ssim
                print(f"   🏆 New best SSIM: {best_ssim:.4f}")
                
                # Save best SSIM model
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'generator_state_dict': generator.state_dict(),
                    'ssim': best_ssim,
                    'metrics': val_metrics
                }, os.path.join(checkpoint_dir, 'best_ssim_model.pth'))
            
            if current_psnr > best_psnr:
                best_psnr = current_psnr
                print(f"   🏆 New best PSNR: {best_psnr:.4f}")
                
                # Save best PSNR model
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'generator_state_dict': generator.state_dict(),
                    'psnr': best_psnr,
                    'metrics': val_metrics
                }, os.path.join(checkpoint_dir, 'best_psnr_model.pth'))
        
        # Early stopping checks
        gen_grad_norm = compute_gradient_norm(generator)
        disc_grad_norm = compute_gradient_norm(discriminator)
        
        # Check generator early stopping
        if early_stopping.check_model('generator', avg_g_loss, generator, epoch, gen_grad_norm):
            print(f"🛑 Generator training stopped early!")
            early_stopping.restore_best_weights('generator', generator)
        
        # Check discriminator early stopping
        if early_stopping.check_model('discriminator', avg_d_loss, discriminator, epoch, disc_grad_norm):
            print(f"🛑 Discriminator training stopped early!")
        
        # Check overall early stopping
        overall_loss = (avg_g_loss + avg_d_loss) / 2
        if early_stopping.check_model('overall', overall_loss, None, epoch):
            print(f"🛑 Overall training stopped early!")
            break
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"generation_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'phase_detector_state_dict': phase_detector.state_dict(),
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss,
                'metrics': {
                    'g_losses': g_losses,
                    'd_losses': d_losses,
                    'p_losses': p_losses,
                    'phase_accuracies': phase_accuracies
                }
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
    # Final summary
    print(f"\n✅ Enhanced disentangled generation training completed:")
    print(f"   Final Generator Loss: {g_losses[-1]:.6f}")
    print(f"   Final Discriminator Loss: {d_losses[-1]:.6f}")
    print(f"   Best SSIM: {best_ssim:.4f}")
    print(f"   Best PSNR: {best_psnr:.4f}")
    
    # Print early stopping summary
    early_stopping_summary = early_stopping.get_summary()
    print(f"\n🛑 Early Stopping Summary:")
    for model_name, summary in early_stopping_summary.items():
        if summary['stopped']:
            print(f"   {model_name}: Stopped - {summary['stop_reason']}")
        else:
            print(f"   {model_name}: Completed normally")
    
    # Print final validation metrics summary
    if validation_loader is not None:
        metrics_tracker.print_summary()
    
    return g_losses, d_losses, p_losses, phase_accuracies

    



def train_contrast_phase_generation(
    train_loader,
    encoder,
    generator,
    discriminator,
    phase_detector,
    num_epochs=150,
    device="cuda",
    checkpoint_dir="checkpoints",
    use_mixed_precision=True,
    validation_loader=None,
    encoder_config=None,
    encoder_type="simple_cnn"
):
    """
    Main training function with optimized sequential training phases
    """
    print(f"🚀 Starting optimized sequential training pipeline")
    print(f"📊 Total epochs: {num_epochs}")
    print(f"🔧 Device: {device}")
    print(f"⚡ Mixed precision: {use_mixed_precision}")
    print(f"🏗️  Encoder type: {encoder_type}")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Move all models to device
    encoder.to(device)
    generator.to(device)
    discriminator.to(device)
    phase_detector.to(device)
    
    # Initialize comprehensive metrics
    metrics = {
        "pretrain_losses": [],
        "phase_losses": [],
        "phase_accuracies": [],
        "g_losses": [],
        "d_losses": [],
        "p_losses": [],
        "final_phase_accuracies": [],
        "val_metrics": {},
        "early_stopping_summary": {},
        "best_models": {}
    }
    
    
    # Determine training phases based on encoder type
    has_pretrained = check_pretrained_weights(encoder, encoder_type)
    
    # Phase allocation
    if has_pretrained:
        # Skip encoder pretraining for pretrained models
        phase_detector_epochs = 40
        disentanglement_epochs = num_epochs - phase_detector_epochs
    else:
        # Include encoder pretraining
        encoder_pretrain_epochs = 30
        phase_detector_epochs = 40
        disentanglement_epochs = num_epochs - encoder_pretrain_epochs - phase_detector_epochs
    
    print(f"\n📋 Training phases:")
    if not has_pretrained:
        print(f"   Phase 1: Encoder + Generator pretraining ({encoder_pretrain_epochs} epochs)")
    print(f"   Phase {'2' if not has_pretrained else '1'}: Phase detector training ({phase_detector_epochs} epochs)")
    print(f"   Phase {'3' if not has_pretrained else '2'}: Disentangled generation ({disentanglement_epochs} epochs)")
    
    current_epoch = 0
    
    # ===========================
    # PHASE 1: Encoder + Generator Pretraining (if needed)
    # ===========================
    if not has_pretrained:
        print(f"\n🔄 Starting Phase 1...")
        pretrain_losses = pretrain_encoder_generator(
            train_loader, encoder, generator, phase_detector,
            num_epochs=encoder_pretrain_epochs,
            device=device,
            use_mixed_precision=use_mixed_precision
        )
        metrics["pretrain_losses"] = pretrain_losses
        current_epoch += encoder_pretrain_epochs
        
        # Save checkpoint after pretraining
        torch.save({
            'phase': 'encoder_pretrain_complete',
            'epoch': current_epoch,
            'encoder_state_dict': encoder.state_dict(),
            'generator_state_dict': generator.state_dict(),
            'metrics': metrics,
            'encoder_config': encoder_config
        }, os.path.join(checkpoint_dir, f"checkpoint_phase1_complete.pth"))
    
    # ===========================
    # PHASE 2: Phase Detector Training
    # ===========================
    print(f"\n🔄 Starting Phase {'2' if not has_pretrained else '1'}...")
    phase_losses, phase_accuracies = train_phase_detector(
        train_loader, encoder, phase_detector,
        num_epochs=phase_detector_epochs,
        device=device,
        use_mixed_precision=use_mixed_precision,
        spatial_size=(128, 128, 128),
        checkpoint_dir=checkpoint_dir
    )
    
    metrics["phase_losses"] = phase_losses
    metrics["phase_accuracies"] = phase_accuracies
    current_epoch += phase_detector_epochs
    
    # Save checkpoint after phase detector training
    torch.save({
        'phase': 'phase_detector_complete',
        'epoch': current_epoch,
        'encoder_state_dict': encoder.state_dict(),
        'generator_state_dict': generator.state_dict(),
        'phase_detector_state_dict': phase_detector.state_dict(),
        'metrics': metrics,
        'encoder_config': encoder_config
    }, os.path.join(checkpoint_dir, f"checkpoint_phase2_complete.pth"))
    
    # ===========================
    # PHASE 3: Disentangled Generation Training
    # ===========================
    print(f"\n🔄 Starting Phase {'3' if not has_pretrained else '2'}...")
    g_losses, d_losses, p_losses, final_phase_accuracies = train_disentangled_generation(
        train_loader, encoder, generator, discriminator, phase_detector,
        validation_loader=validation_loader,
        num_epochs=disentanglement_epochs,
        device=device,
        use_mixed_precision=use_mixed_precision,
        checkpoint_dir=checkpoint_dir
    )

    metrics["g_losses"] = g_losses
    metrics["d_losses"] = d_losses
    metrics["p_losses"].extend(p_losses)  # Extend because we may have previous p_losses
    metrics["final_phase_accuracies"] = final_phase_accuracies
    
    # ===========================
    # Validation (if provided)
    # ===========================
    # Final validation with comprehensive metrics
    if validation_loader is not None:
        print(f"\n🔍 Running final comprehensive validation...")
        
        # Setup final metrics tracker
        final_metrics_tracker = ValidationMetricsTracker(
            save_path=os.path.join(checkpoint_dir, 'final_validation_metrics.csv')
        )
        
        final_val_metrics = run_validation_with_metrics(
            validation_loader, encoder, generator, final_metrics_tracker, device, max_batches=20
        )
        
        metrics["val_metrics"] = final_val_metrics
        
        print(f"📊 Final Validation Results:")
        print(f"   PSNR: {final_val_metrics.get('psnr', 0):.4f}")
        print(f"   SSIM: {final_val_metrics.get('ssim', 0):.4f}")
        print(f"   MS-SSIM: {final_val_metrics.get('ms_ssim', 0):.4f}")
        print(f"   NMSE: {final_val_metrics.get('nmse', 0):.6f}")
        print(f"   NCC: {final_val_metrics.get('ncc', 0):.4f}")
        print(f"   Mutual Information: {final_val_metrics.get('mi', 0):.4f}")
        
        # Save final metrics summary
        final_metrics_tracker.print_summary()
 
    # if validation_loader is not None:
    #     print(f"\n🔍 Running final validation...")
    #     val_metrics = run_validation(
    #         validation_loader, encoder, generator, discriminator, phase_detector,
    #         device=device, use_mixed_precision=use_mixed_precision
    #     )
    #     metrics["val_metrics"] = val_metrics
        
    #     print(f"✅ Validation Results:")
    #     print(f"   Reconstruction Loss: {val_metrics['reconstruction_loss']:.6f}")
    #     print(f"   GAN Loss: {val_metrics['gan_loss']:.6f}")
    #     print(f"   Phase Accuracy: {val_metrics['phase_accuracy']:.4f}")
    
    # ===========================
    # Save Final Checkpoint
    # ===========================
    final_checkpoint = {
        'phase': 'enhanced_training_complete',
        'total_epochs': num_epochs,
        'encoder_state_dict': encoder.state_dict(),
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'phase_detector_state_dict': phase_detector.state_dict(),
        'metrics': metrics,
        'encoder_config': encoder_config,
        'encoder_type': encoder_type,
        'training_enhancements': {
            'early_stopping': True,
            'image_quality_metrics': True,
            'advanced_validation': True
        }
    }
    
    torch.save(final_checkpoint, os.path.join(checkpoint_dir, "final_checkpoint.pth"))
    
    # Save metrics to CSV
    save_metrics_to_csv(metrics, checkpoint_dir)
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📁 Checkpoints saved to: {checkpoint_dir}")
    
    return metrics

def run_validation(validation_loader, encoder, generator, discriminator, phase_detector, 
                  device="cuda", use_mixed_precision=True):
    """Run validation and return metrics"""
    encoder.eval()
    generator.eval()
    discriminator.eval()
    phase_detector.eval()
    
    l1_loss = nn.L1Loss()
    gan_loss = nn.BCEWithLogitsLoss()
    phase_loss = nn.CrossEntropyLoss()
    
    val_reconstruction_loss = 0.0
    val_gan_loss = 0.0
    val_phase_loss = 0.0
    phase_correct = 0
    phase_total = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(validation_loader, desc="Validation"):
            input_volume = batch["input_volume"].to(device)
            target_volume = batch["target_volume"].to(device)
            target_phase = batch["target_phase"].to(device)
            true_phase = batch["input_phase"].to(device)
            
            with autocast(device_type="cuda", enabled=use_mixed_precision):
                # Forward pass
                z = encoder(input_volume)
                phase_emb = create_phase_embeddings(target_phase, dim=32, device=device)
                generated_volume = generator(z, phase_emb)
                
                # Phase prediction
                phase_pred = phase_detector(z)
                
                # Calculate losses
                recon_loss = l1_loss(generated_volume, target_volume)
                
                real_score = discriminator(target_volume)
                fake_score = discriminator(generated_volume)
                g_loss = gan_loss(fake_score, torch.ones_like(fake_score))
                
                p_loss = phase_loss(phase_pred, true_phase)
            
            val_reconstruction_loss += recon_loss.item()
            val_gan_loss += g_loss.item()
            val_phase_loss += p_loss.item()
            
            # Phase accuracy
            _, predicted = torch.max(phase_pred.data, 1)
            phase_correct += (predicted == true_phase).sum().item()
            phase_total += true_phase.size(0)
            num_batches += 1
    
    return {
        'reconstruction_loss': val_reconstruction_loss / num_batches,
        'gan_loss': val_gan_loss / num_batches,
        'phase_loss': val_phase_loss / num_batches,
        'phase_accuracy': phase_correct / phase_total if phase_total > 0 else 0
    }

def save_metrics_to_csv(metrics, checkpoint_dir):

    """Save enhanced training metrics with additional columns"""
    csv_path = os.path.join(checkpoint_dir, 'enhanced_training_metrics.csv')
    
    # Create comprehensive metrics CSV
    max_len = max([
        len(metrics.get("phase_losses", [])),
        len(metrics.get("g_losses", [])),
        len(metrics.get("d_losses", [])),
        len(metrics.get("p_losses", []))
    ])
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'epoch', 'phase', 'phase_detector_loss', 'phase_accuracy', 
            'generator_loss', 'discriminator_loss', 'final_phase_loss', 
            'final_phase_accuracy', 'early_stopped', 'validation_psnr',
            'validation_ssim', 'validation_nmse'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write phase detector metrics
        phase_losses = metrics.get("phase_losses", [])
        phase_accs = metrics.get("phase_accuracies", [])
        for i in range(len(phase_losses)):
            writer.writerow({
                'epoch': i + 1,
                'phase': 'enhanced_phase_detector',
                'phase_detector_loss': phase_losses[i],
                'phase_accuracy': phase_accs[i] if i < len(phase_accs) else 0,
                'early_stopped': False  # Would need to track this from early stopping
            })
        
        # Write generation metrics
        g_losses = metrics.get("g_losses", [])
        d_losses = metrics.get("d_losses", [])
        final_p_losses = metrics.get("p_losses", [])[len(phase_losses):]
        final_p_accs = metrics.get("final_phase_accuracies", [])
        
        for i in range(len(g_losses)):
            base_epoch = len(phase_losses)
            writer.writerow({
                'epoch': base_epoch + i + 1,
                'phase': 'enhanced_disentangled_generation',
                'generator_loss': g_losses[i] if i < len(g_losses) else 0,
                'discriminator_loss': d_losses[i] if i < len(d_losses) else 0,
                'final_phase_loss': final_p_losses[i] if i < len(final_p_losses) else 0,
                'final_phase_accuracy': final_p_accs[i] if i < len(final_p_accs) else 0,
                'early_stopped': False,  # Would need to track this
                # Validation metrics would be added when available
                'validation_psnr': metrics.get('val_metrics', {}).get('psnr', 0) if (i+1) % 5 == 0 else '',
                'validation_ssim': metrics.get('val_metrics', {}).get('ssim', 0) if (i+1) % 5 == 0 else '',
                'validation_nmse': metrics.get('val_metrics', {}).get('nmse', 0) if (i+1) % 5 == 0 else ''
            })
    
    print(f"📊 Enhanced metrics saved to: {csv_path}")
    
    # Also save a summary JSON
    summary_path = os.path.join(checkpoint_dir, 'training_summary.json')
    summary = {
        'training_completed': datetime.now().isoformat(),
        'total_epochs': len(metrics.get("phase_losses", [])) + len(metrics.get("g_losses", [])),
        'phase_detector_epochs': len(metrics.get("phase_losses", [])),
        'generation_epochs': len(metrics.get("g_losses", [])),
        'best_phase_accuracy': max(metrics.get("phase_accuracies", [0])),
        'final_generator_loss': metrics.get("g_losses", [0])[-1],
        'final_validation_metrics': metrics.get("val_metrics", {}),
        'enhancements_used': {
            'early_stopping': True,
            'image_quality_metrics': True,
            'class_weighted_loss': True,
            'gradient_clipping': True,
            'mixed_precision': True
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Training summary saved to: {summary_path}")


    # """Save training metrics to CSV file"""
    # csv_path = os.path.join(checkpoint_dir, 'training_metrics.csv')
    
    # # Determine maximum length for alignment
    # max_len = max([
    #     len(metrics.get("pretrain_losses", [])),
    #     len(metrics.get("phase_losses", [])),
    #     len(metrics.get("g_losses", [])),
    #     len(metrics.get("d_losses", [])),
    #     len(metrics.get("p_losses", []))
    # ])
    
    # with open(csv_path, 'w', newline='') as csvfile:
    #     fieldnames = ['epoch', 'phase', 'pretrain_loss', 'phase_detector_loss', 
    #                  'phase_accuracy', 'generator_loss', 'discriminator_loss', 
    #                  'final_phase_loss', 'final_phase_accuracy']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
        
    #     # Write pretraining metrics
    #     for i, loss in enumerate(metrics.get("pretrain_losses", [])):
    #         writer.writerow({
    #             'epoch': i + 1,
    #             'phase': 'encoder_pretraining',
    #             'pretrain_loss': loss
    #         })
        
    #     # Write phase detector metrics
    #     phase_losses = metrics.get("phase_losses", [])
    #     phase_accs = metrics.get("phase_accuracies", [])
    #     for i in range(len(phase_losses)):
    #         writer.writerow({
    #             'epoch': len(metrics.get("pretrain_losses", [])) + i + 1,
    #             'phase': 'phase_detector',
    #             'phase_detector_loss': phase_losses[i],
    #             'phase_accuracy': phase_accs[i] if i < len(phase_accs) else 0
    #         })
        
    #     # Write disentanglement metrics
    #     g_losses = metrics.get("g_losses", [])
    #     d_losses = metrics.get("d_losses", [])
    #     final_p_losses = metrics.get("p_losses", [])[len(phase_losses):]  # Skip phase detector losses
    #     final_p_accs = metrics.get("final_phase_accuracies", [])
        
    #     for i in range(len(g_losses)):
    #         base_epoch = len(metrics.get("pretrain_losses", [])) + len(phase_losses)
    #         writer.writerow({
    #             'epoch': base_epoch + i + 1,
    #             'phase': 'disentangled_generation',
    #             'generator_loss': g_losses[i] if i < len(g_losses) else 0,
    #             'discriminator_loss': d_losses[i] if i < len(d_losses) else 0,
    #             'final_phase_loss': final_p_losses[i] if i < len(final_p_losses) else 0,
    #             'final_phase_accuracy': final_p_accs[i] if i < len(final_p_accs) else 0
    #         })
    
    # print(f"📊 Metrics saved to: {csv_path}")
