import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.amp import autocast, GradScaler
from utils import GradientReversalLayer, get_phase_embedding
import numpy as np
from tqdm import tqdm
import csv

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

def train_phase_detector(train_loader, encoder, phase_detector, num_epochs=40, device="cuda", use_mixed_precision=True):
    """
    FIXED Phase 2: Train phase detector with frozen encoder
    """
    print(f"\n{'='*60}")
    print("PHASE 2: PHASE DETECTOR TRAINING (FIXED)")
    print(f"{'='*60}")
    
    # FIXED: Properly freeze encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    
    phase_detector.train()
    
    # Phase classification loss
    phase_loss = nn.CrossEntropyLoss()
    
    # FIXED: Use different optimizer and learning rate
    optimizer_phase = optim.AdamW(phase_detector.parameters(), lr=5e-4, weight_decay=1e-4)
    
    # FIXED: Add learning rate scheduler
    scheduler_phase = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase, T_max=num_epochs)
    
    # Mixed precision
    scaler = GradScaler() if use_mixed_precision else None
    
    phase_losses = []
    phase_accuracies = []
    
    # FIXED: Analyze class distribution
    print("📊 Analyzing phase distribution in dataset...")
    class_count = 3
    phase_counts = {i: 0 for i in range(class_count)}
    total_samples = 0
    
    with torch.no_grad():
        for batch in train_loader:
            target_phase_labels = batch["input_phase"]
            for phase in target_phase_labels:
                phase_counts[phase.item()] += 1
                total_samples += 1
    
    print(f"   Phase distribution:")
    for phase, count in phase_counts.items():
        percentage = (count / total_samples) * 100
        print(f"   Phase {phase}: {count} samples ({percentage:.1f}%)")
    
    # FIXED: Create class weights for imbalanced data
    class_weights = []
    for i in range(class_count):
        weight = total_samples / (class_count * phase_counts[i]) if phase_counts[i] > 0 else 1.0
        class_weights.append(weight)
    
    class_weights = torch.tensor(class_weights).to(device)
    weighted_phase_loss = nn.CrossEntropyLoss(weight=class_weights)
    print(f"   Using class weights: {class_weights.cpu().numpy()}")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        phase_correct = 0
        phase_total = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Phase Epoch {epoch+1}/{num_epochs}")):
            input_volume = batch["input_volume"].to(device)
            target_phase_label = batch["input_phase"].to(device)
            optimizer_phase.zero_grad()
            
            with autocast(device_type="cuda", enabled=use_mixed_precision):
                # Get encoder features (frozen)
                with torch.no_grad():
                    z = encoder(input_volume)
                    # FIXED: Detach to ensure no gradients flow to encoder
                    z = z.detach()
                
                # Predict phase
                phase_pred = phase_detector(z)
                loss = weighted_phase_loss(phase_pred, target_phase_label)
            
            # Backward pass
            if use_mixed_precision:
                scaler.scale(loss).backward()
                
                # FIXED: Add gradient clipping
                scaler.unscale_(optimizer_phase)
                torch.nn.utils.clip_grad_norm_(phase_detector.parameters(), max_norm=1.0)
                
                scaler.step(optimizer_phase)
                scaler.update()
            else:
                loss.backward()
                
                # FIXED: Add gradient clipping
                torch.nn.utils.clip_grad_norm_(phase_detector.parameters(), max_norm=1.0)
                
                optimizer_phase.step()
            
            # Track metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(phase_pred.data, 1)
            phase_correct += (predicted == target_phase_label).sum().item()
            phase_total += target_phase_label.size(0)
            num_batches += 1
            
            # Debug gradient flow
            if batch_idx % 50 == 0:
                check_gradient_flow(phase_detector, "Phase Detector")
        
        avg_loss = epoch_loss / num_batches
        accuracy = phase_correct / phase_total if phase_total > 0 else 0
        
        phase_losses.append(avg_loss)
        phase_accuracies.append(accuracy)
        
        # Update learning rate
        scheduler_phase.step()
        
        # Track improvement
        if epoch == 0 or accuracy > max(phase_accuracies[:-1]):
            improvement = "✅"
        else:
            improvement = "📉"
        
        print(f"Phase Epoch {epoch+1}: Loss = {avg_loss:.6f}, Accuracy = {accuracy:.4f} {improvement}")
        
        # FIXED: Print per-class accuracy every 10 epochs
        if (epoch + 1) % 10 == 0:
            print("   Per-class accuracy analysis:")
            class_correct = {0: 0, 1: 0, 2: 0, 3: 0}
            class_total = {0: 0, 1: 0, 2: 0, 3: 0}
            
            with torch.no_grad():
                for batch in train_loader:
                    input_volume = batch["input_volume"].to(device)
                    target_phase_label = batch["input_phase"].to(device)
                    
                    z = encoder(input_volume).detach()
                    phase_pred = phase_detector(z)
                    _, predicted = torch.max(phase_pred.data, 1)
                    
                    for i in range(len(target_phase_label)):
                        true_label = target_phase_label[i].item()
                        pred_label = predicted[i].item()
                        class_total[true_label] += 1
                        if true_label == pred_label:
                            class_correct[true_label] += 1
            
            phase_names = ['Non-contrast', 'Arterial', 'Venous'] # and , 'Delayed'
            for i in range(len(phase_names)):
                if class_total[i] > 0:
                    acc = class_correct[i] / class_total[i]
                    print(f"     {phase_names[i]}: {acc:.3f} ({class_correct[i]}/{class_total[i]})")
    
    # Unfreeze encoder for next phase
    for param in encoder.parameters():
        param.requires_grad = True
    encoder.train()
    
    print(f"✓ Phase detector training completed (Best accuracy: {max(phase_accuracies):.4f})")
    return phase_losses, phase_accuracies

def train_disentangled_generation(train_loader, encoder, generator, discriminator, phase_detector, 
                                 num_epochs=50, device="cuda", use_mixed_precision=True):
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
        "val_metrics": {}
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
        use_mixed_precision=use_mixed_precision
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
        num_epochs=disentanglement_epochs,
        device=device,
        use_mixed_precision=use_mixed_precision
    )
    metrics["g_losses"] = g_losses
    metrics["d_losses"] = d_losses
    metrics["p_losses"].extend(p_losses)  # Extend because we may have previous p_losses
    metrics["final_phase_accuracies"] = final_phase_accuracies
    
    # ===========================
    # Validation (if provided)
    # ===========================
    if validation_loader is not None:
        print(f"\n🔍 Running final validation...")
        val_metrics = run_validation(
            validation_loader, encoder, generator, discriminator, phase_detector,
            device=device, use_mixed_precision=use_mixed_precision
        )
        metrics["val_metrics"] = val_metrics
        
        print(f"✅ Validation Results:")
        print(f"   Reconstruction Loss: {val_metrics['reconstruction_loss']:.6f}")
        print(f"   GAN Loss: {val_metrics['gan_loss']:.6f}")
        print(f"   Phase Accuracy: {val_metrics['phase_accuracy']:.4f}")
    
    # ===========================
    # Save Final Checkpoint
    # ===========================
    final_checkpoint = {
        'phase': 'training_complete',
        'total_epochs': num_epochs,
        'encoder_state_dict': encoder.state_dict(),
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'phase_detector_state_dict': phase_detector.state_dict(),
        'metrics': metrics,
        'encoder_config': encoder_config,
        'encoder_type': encoder_type
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
    """Save training metrics to CSV file"""
    csv_path = os.path.join(checkpoint_dir, 'training_metrics.csv')
    
    # Determine maximum length for alignment
    max_len = max([
        len(metrics.get("pretrain_losses", [])),
        len(metrics.get("phase_losses", [])),
        len(metrics.get("g_losses", [])),
        len(metrics.get("d_losses", [])),
        len(metrics.get("p_losses", []))
    ])
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'phase', 'pretrain_loss', 'phase_detector_loss', 
                     'phase_accuracy', 'generator_loss', 'discriminator_loss', 
                     'final_phase_loss', 'final_phase_accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write pretraining metrics
        for i, loss in enumerate(metrics.get("pretrain_losses", [])):
            writer.writerow({
                'epoch': i + 1,
                'phase': 'encoder_pretraining',
                'pretrain_loss': loss
            })
        
        # Write phase detector metrics
        phase_losses = metrics.get("phase_losses", [])
        phase_accs = metrics.get("phase_accuracies", [])
        for i in range(len(phase_losses)):
            writer.writerow({
                'epoch': len(metrics.get("pretrain_losses", [])) + i + 1,
                'phase': 'phase_detector',
                'phase_detector_loss': phase_losses[i],
                'phase_accuracy': phase_accs[i] if i < len(phase_accs) else 0
            })
        
        # Write disentanglement metrics
        g_losses = metrics.get("g_losses", [])
        d_losses = metrics.get("d_losses", [])
        final_p_losses = metrics.get("p_losses", [])[len(phase_losses):]  # Skip phase detector losses
        final_p_accs = metrics.get("final_phase_accuracies", [])
        
        for i in range(len(g_losses)):
            base_epoch = len(metrics.get("pretrain_losses", [])) + len(phase_losses)
            writer.writerow({
                'epoch': base_epoch + i + 1,
                'phase': 'disentangled_generation',
                'generator_loss': g_losses[i] if i < len(g_losses) else 0,
                'discriminator_loss': d_losses[i] if i < len(d_losses) else 0,
                'final_phase_loss': final_p_losses[i] if i < len(final_p_losses) else 0,
                'final_phase_accuracy': final_p_accs[i] if i < len(final_p_accs) else 0
            })
    
    print(f"📊 Metrics saved to: {csv_path}")
