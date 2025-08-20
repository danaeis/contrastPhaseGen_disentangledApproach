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

def pretrain_encoder_generator(train_loader, encoder, generator, num_epochs=20, device="cuda", use_mixed_precision=True):
    """
    Phase 1: Pretrain encoder and generator with reconstruction loss
    This creates meaningful embeddings before phase detection training
    """
    print(f"\n{'='*60}")
    print("PHASE 1: ENCODER + GENERATOR PRETRAINING")
    print(f"{'='*60}")
    
    # Set models to training mode
    encoder.train()
    generator.train()
    
    # Reconstruction loss
    reconstruction_loss = nn.L1Loss()
    
    # Optimizers with higher learning rate for pretraining
    optimizer_enc = optim.Adam(encoder.parameters(), lr=2e-4)
    optimizer_gen = optim.Adam(generator.parameters(), lr=2e-4)
    
    # Mixed precision
    scaler = GradScaler() if use_mixed_precision else None
    
    pretrain_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{num_epochs}"):
            input_volume = batch["input_volume"].to(device)
            # For pretraining, we use input volume as target (autoencoder-like)
            
            optimizer_enc.zero_grad()
            optimizer_gen.zero_grad()
            
            with autocast(device_type="cuda", enabled=use_mixed_precision):
                # Encode input
                z = encoder(input_volume)
                
                # Generate random phase embedding for diversity
                batch_size = input_volume.shape[0]
                random_phases = torch.randint(0, 4, (batch_size,)).to(device)
                phase_emb = create_phase_embeddings(random_phases, dim=32, device=device)
                
                # Generate volume
                reconstructed_volume = generator(z, phase_emb)
                
                # Reconstruction loss
                loss = reconstruction_loss(reconstructed_volume, input_volume)
            
            # Backward pass
            if use_mixed_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer_enc)
                scaler.step(optimizer_gen)
                scaler.update()
            else:
                loss.backward()
                optimizer_enc.step()
                optimizer_gen.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        pretrain_losses.append(avg_loss)
        print(f"Pretrain Epoch {epoch+1}: Reconstruction Loss = {avg_loss:.6f}")
    
    print(f"✓ Encoder + Generator pretraining completed")
    return pretrain_losses

def train_phase_detector(train_loader, encoder, phase_detector, num_epochs=30, device="cuda", use_mixed_precision=True):
    """
    Phase 2: Train phase detector with frozen encoder
    This learns to classify contrast phases from encoder embeddings
    """
    print(f"\n{'='*60}")
    print("PHASE 2: PHASE DETECTOR TRAINING")
    print(f"{'='*60}")
    
    # Freeze encoder, train phase detector
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    
    phase_detector.train()
    
    # Phase classification loss
    phase_loss = nn.CrossEntropyLoss()
    
    # Optimizer with higher learning rate for phase detector
    optimizer_phase = optim.Adam(phase_detector.parameters(), lr=1e-3)
    
    # Mixed precision
    scaler = GradScaler() if use_mixed_precision else None
    
    phase_losses = []
    phase_accuracies = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        phase_correct = 0
        phase_total = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Phase Epoch {epoch+1}/{num_epochs}"):
            input_volume = batch["input_volume"].to(device)
            true_phase_label = batch["input_phase"].to(device)
            
            optimizer_phase.zero_grad()
            
            with autocast(device_type="cuda", enabled=use_mixed_precision):
                # Get encoder features (frozen)
                with torch.no_grad():
                    z = encoder(input_volume)
                
                # Predict phase
                phase_pred = phase_detector(z)
                loss = phase_loss(phase_pred, true_phase_label)
            
            # Backward pass
            if use_mixed_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer_phase)
                scaler.update()
            else:
                loss.backward()
                optimizer_phase.step()
            
            # Track metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(phase_pred.data, 1)
            phase_correct += (predicted == true_phase_label).sum().item()
            phase_total += true_phase_label.size(0)
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        accuracy = phase_correct / phase_total if phase_total > 0 else 0
        
        phase_losses.append(avg_loss)
        phase_accuracies.append(accuracy)
        
        print(f"Phase Epoch {epoch+1}: Loss = {avg_loss:.6f}, Accuracy = {accuracy:.4f}")
    
    # Unfreeze encoder for next phase
    for param in encoder.parameters():
        param.requires_grad = True
    
    print(f"✓ Phase detector training completed")
    return phase_losses, phase_accuracies

def train_disentangled_generation(train_loader, encoder, generator, discriminator, phase_detector, 
                                 num_epochs=50, device="cuda", use_mixed_precision=True):
    """
    Phase 3: Train generator with adversarial loss and reversed gradients from phase detector
    This achieves disentangled representations
    """
    print(f"\n{'='*60}")
    print("PHASE 3: DISENTANGLED GENERATION TRAINING")
    print(f"{'='*60}")
    
    # Set all models to training mode
    encoder.train()
    generator.train()
    discriminator.train()
    phase_detector.train()
    
    # Losses
    l1_loss = nn.L1Loss()
    gan_loss = nn.BCEWithLogitsLoss()
    phase_loss = nn.CrossEntropyLoss()
    
    # Optimizers with balanced learning rates
    optimizer_enc_gen = optim.Adam(list(encoder.parameters()) + list(generator.parameters()), lr=1e-4)
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=1e-4)
    optimizer_phase = optim.Adam(phase_detector.parameters(), lr=5e-5)  # Lower LR for fine-tuning
    
    # Mixed precision
    scaler = GradScaler() if use_mixed_precision else None
    
    # Training metrics
    g_losses = []
    d_losses = []
    p_losses = []
    phase_accuracies = []
    
    for epoch in range(num_epochs):
        # Gradual increase of adversarial weight
        lambda_adv = min(1.0, (epoch + 1) / 20)  # Ramp up over 20 epochs
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_p_loss = 0.0
        phase_correct = 0
        phase_total = 0
        num_batches = 0
        
        print(f"Disentanglement Epoch {epoch+1}/{num_epochs} (λ_adv = {lambda_adv:.3f})")
        
        for batch in tqdm(train_loader, desc=f"Disentangle Epoch {epoch+1}"):
            input_volume = batch["input_volume"].to(device)
            target_volume = batch["target_volume"].to(device)
            target_phase = batch["target_phase"].to(device)
            true_phase = batch["input_phase"].to(device)
            
            batch_size = input_volume.shape[0]
            
            # ===========================
            # Train Discriminator
            # ===========================
            optimizer_disc.zero_grad()
            
            with autocast(device_type="cuda", enabled=use_mixed_precision):
                # Generate fake volumes
                z = encoder(input_volume)
                phase_emb = create_phase_embeddings(target_phase, dim=32, device=device)
                fake_volume = generator(z, phase_emb).detach()
                
                # Discriminator outputs
                real_score = discriminator(target_volume)
                fake_score = discriminator(fake_volume)
                
                # Discriminator loss
                d_loss_real = gan_loss(real_score, torch.ones_like(real_score))
                d_loss_fake = gan_loss(fake_score, torch.zeros_like(fake_score))
                d_loss = (d_loss_real + d_loss_fake) / 2
            
            if use_mixed_precision:
                scaler.scale(d_loss).backward()
                scaler.step(optimizer_disc)
                scaler.update()
            else:
                d_loss.backward()
                optimizer_disc.step()
            
            # ===========================
            # Train Generator + Encoder
            # ===========================
            optimizer_enc_gen.zero_grad()
            
            with autocast(device_type="cuda", enabled=use_mixed_precision):
                # Generate volumes
                z = encoder(input_volume)
                phase_emb = create_phase_embeddings(target_phase, dim=32, device=device)
                generated_volume = generator(z, phase_emb)
                
                # Generator losses
                fake_score = discriminator(generated_volume)
                
                # L1 reconstruction loss
                reconstruction_loss = l1_loss(generated_volume, target_volume)
                
                # GAN loss
                adversarial_loss = gan_loss(fake_score, torch.ones_like(fake_score))
                
                # Combined generator loss
                g_loss = reconstruction_loss * 10.0 + adversarial_loss * lambda_adv
            
            if use_mixed_precision:
                scaler.scale(g_loss).backward()
                scaler.step(optimizer_enc_gen)
                scaler.update()
            else:
                g_loss.backward()
                optimizer_enc_gen.step()
            
            # ===========================
            # Train Phase Detector with Reversed Gradients
            # ===========================
            optimizer_phase.zero_grad()
            
            with autocast(device_type="cuda", enabled=use_mixed_precision):
                # Get encoder features
                z = encoder(input_volume)
                
                # Apply gradient reversal for disentanglement
                z_reversed = GradientReversalLayer.apply(z, lambda_adv)
                
                # Phase prediction on reversed gradients
                phase_pred = phase_detector(z_reversed)
                p_loss = phase_loss(phase_pred, true_phase)
            
            if use_mixed_precision:
                scaler.scale(p_loss).backward()
                scaler.step(optimizer_phase)
                scaler.update()
            else:
                p_loss.backward()
                optimizer_phase.step()
            
            # Track metrics
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_p_loss += p_loss.item()
            
            _, predicted = torch.max(phase_pred.data, 1)
            phase_correct += (predicted == true_phase).sum().item()
            phase_total += true_phase.size(0)
            num_batches += 1
        
        # Calculate epoch metrics
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        avg_p_loss = epoch_p_loss / num_batches
        phase_accuracy = phase_correct / phase_total if phase_total > 0 else 0
        
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        p_losses.append(avg_p_loss)
        phase_accuracies.append(phase_accuracy)
        
        print(f"Epoch {epoch+1}: G_loss = {avg_g_loss:.6f}, D_loss = {avg_d_loss:.6f}, "
              f"P_loss = {avg_p_loss:.6f}, Phase_Acc = {phase_accuracy:.4f}")
    
    print(f"✓ Disentangled generation training completed")
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
            train_loader, encoder, generator, 
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
