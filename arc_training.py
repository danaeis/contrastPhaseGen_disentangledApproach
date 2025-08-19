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
    """
    Create phase embeddings for a batch of phase labels
    
    Args:
        phase_labels (torch.Tensor): Batch of phase labels
        dim (int): Embedding dimension  
        device (str or torch.device): Target device
    
    Returns:
        torch.Tensor: Batch of phase embeddings (batch_size, dim)
    """
    batch_embeddings = []
    
    for phase_label in phase_labels:
        # Extract scalar value if tensor
        if isinstance(phase_label, torch.Tensor):
            phase_val = phase_label.item()
        else:
            phase_val = phase_label
            
        # Create embedding on correct device
        embedding = get_phase_embedding(phase_val, dim=dim, device=device)
        batch_embeddings.append(embedding)
    
    return torch.stack(batch_embeddings)

def train_contrast_phase_generation(
    data_loader,
    encoder,
    generator,
    discriminator,
    phase_detector,
    num_epochs=130,  # Updated total
    device="cuda",
    checkpoint_dir="checkpoints",
    use_mixed_precision=True,
    validation_loader=None,
    encoder_config=None
):
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Losses
    l1_loss = nn.L1Loss()
    gan_loss = nn.BCEWithLogitsLoss()
    phase_loss = nn.CrossEntropyLoss()

    # Optimizers with initial LRs
    optimizer_enc_gen = optim.Adam(list(encoder.parameters()) + list(generator.parameters()), lr=1e-4)
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=1e-4)
    optimizer_phase = optim.Adam(phase_detector.parameters(), lr=1e-3)  # Higher LR for phase 1

    # Move models to device
    encoder.to(device)
    generator.to(device)
    discriminator.to(device)
    phase_detector.to(device)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if use_mixed_precision else None
    
    # Training metrics
    metrics = {
        "g_losses": [],
        "d_losses": [],
        "p_losses": [],
        "phase_accuracy": [],
        "val_g_losses": [],
        "val_d_losses": [],
        "val_p_losses": [],
        "val_phase_accuracy": []
    }
    print(f"Starting training with {len(data_loader)} batches per epoch")
    print(f"Device: {device}, Mixed Precision: {use_mixed_precision}")
    # Training loop
    for epoch in range(num_epochs):
        if epoch < 30:  # Phase 1: Pre-train phase detector
            phase_name = "Phase 1: Phase Detector Pre-training"
            lambda_ = 0.0
            encoder.eval()
            generator.eval()
            discriminator.eval()
            phase_detector.train()
            for param_group in optimizer_phase.param_groups:
                param_group['lr'] = 1e-3
        elif epoch < 80:  # Phase 2: GAN Training (epochs 30-79)
            phase_name = "Phase 2: GAN Training"
            lambda_ = 0.0
            phase_detector.eval()
            sub_epoch = epoch % 10
            if sub_epoch < 5:
                # Train generator
                encoder.train()
                generator.train()
                discriminator.eval()
            else:
                # Train discriminator
                encoder.eval()
                generator.eval()
                discriminator.train()
            for param_group in optimizer_enc_gen.param_groups:
                param_group['lr'] = 1e-4
            for param_group in optimizer_disc.param_groups:
                param_group['lr'] = 1e-4
        else:  # Phase 3: Disentanglement (epochs 80-129)
            phase_name = "Phase 3: Disentanglement"
            progress = (epoch - 80) / 50.0
            lambda_ = progress * 1.0  # Gradual ramp-up
            encoder.train()
            generator.train()
            discriminator.train()
            phase_detector.train()
            for param_group in optimizer_phase.param_groups:
                param_group['lr'] = 1e-4

        print(f"Epoch {epoch+1}/{num_epochs}: {phase_name} (λ={lambda_:.2f})")

        # Initialize epoch metrics
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_p_loss = 0.0
        phase_correct = 0
        phase_total = 0

        # Training mode
        if epoch < 10:
            # Pre-train phase detector
            encoder.eval()
            generator.eval()
            discriminator.eval()
            phase_detector.train()
            print(f"Epoch {epoch+1}/{num_epochs}: Pre-training phase detector")
        elif epoch % 10 < 5:
            # Train generator and discriminator
            encoder.train()
            generator.train()
            discriminator.train()
            phase_detector.eval()
            print(f"Epoch {epoch+1}/{num_epochs}: Training generator and discriminator")
        else:
            # Train phase detector with reverse gradient
            encoder.train()
            generator.eval()
            discriminator.eval()
            phase_detector.train()
            print(f"Epoch {epoch+1}/{num_epochs}: Training phase detector with reverse gradient (λ={lambda_:.2f})")

        # Process batches with progress bar
        for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_volume = batch["input_volume"].to(device)  # (batch, 1, 128, 128, 128)
            target_volume = batch["target_volume"].to(device)  # (batch, 1, 128, 128, 128)
            phase_label = batch["target_phase"].to(device)  # (batch,) e.g., 0, 1, 2, 3
            true_phase_label = batch["input_phase"].to(device)  # For phase detector

            if epoch < 30:  # Phase 1
                # Pre-train phase detector
                optimizer_phase.zero_grad()
                
                with autocast(device_type="cuda", enabled=use_mixed_precision):
                    z = encoder(input_volume).detach()
                    phase_pred = phase_detector(z)
                    # print(f"Phase embedding shape: {phase_emb.shape}")
                    # print(f"Encoder output shape: {z.shape}")
                    # print(f"Generator input shape: {z.shape} + {phase_emb.shape}")
                    p_loss = phase_loss(phase_pred, true_phase_label)
                
                if use_mixed_precision:
                    scaler.scale(p_loss).backward()
                    scaler.step(optimizer_phase)
                    scaler.update()
                else:
                    p_loss.backward()
                    optimizer_phase.step()
                
                # Track metrics
                epoch_p_loss += p_loss.item()
                _, predicted = torch.max(phase_pred.data, 1)
                phase_correct += (predicted == true_phase_label).sum().item()
                phase_total += true_phase_label.size(0)
                
            elif epoch % 10 < 5:
                # Train generator and discriminator
                optimizer_enc_gen.zero_grad()
                optimizer_disc.zero_grad()
                
                with autocast(device_type="cuda", enabled=use_mixed_precision):
                    # Forward pass
                    z = encoder(input_volume)
                    # phase_emb = torch.stack([get_phase_embedding(p, dim=32).to(device) for p in phase_label])
                    phase_emb = create_phase_embeddings(phase_label, dim=32, device=device)
                    # print(f"Phase embedding shape: {phase_emb.shape}")
                    # print(f"Encoder output shape: {z.shape}")
                    # print(f"Generator input shape: {z.shape} + {phase_emb.shape}")
                    generated_volume = generator(z, phase_emb)
                    
                    # Discriminator outputs
                    real_score = discriminator(target_volume)
                    fake_score = discriminator(generated_volume.detach())  # Detach for discriminator update
                    
                    # Discriminator loss
                    d_loss = gan_loss(real_score, torch.ones_like(real_score)) + \
                             gan_loss(fake_score, torch.zeros_like(fake_score))
                
                # Update discriminator
                if use_mixed_precision:
                    scaler.scale(d_loss).backward()
                    scaler.step(optimizer_disc)
                    scaler.update()
                else:
                    d_loss.backward()
                    optimizer_disc.step()
                
                # Generator forward pass (need to recompute fake_score since we detached earlier)
                optimizer_enc_gen.zero_grad()
                
                with autocast(device_type="cuda", enabled=use_mixed_precision):
                    fake_score = discriminator(generated_volume)
                    
                    # Generator loss (L1 + GAN)
                    g_loss = l1_loss(generated_volume, target_volume) * 10.0 + \
                             gan_loss(fake_score, torch.ones_like(fake_score))
                
                # Update generator
                if use_mixed_precision:
                    scaler.scale(g_loss).backward()
                    scaler.step(optimizer_enc_gen)
                    scaler.update()
                else:
                    g_loss.backward()
                    optimizer_enc_gen.step()
                
                # Track metrics
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                
            else:
                # Train phase detector with reverse gradient (and possibly gen/disc in phase 4)
                optimizer_phase.zero_grad()
                if epoch >= 90:  # Also optimize enc_gen and disc in phase 4
                    optimizer_enc_gen.zero_grad()
                    optimizer_disc.zero_grad()
                
                with autocast(device_type="cuda", enabled=use_mixed_precision):
                    z = encoder(input_volume)
                    phase_emb = create_phase_embeddings(phase_label, dim=32, device=device)
                    if epoch >= 90:  # Generate for gen/disc training in phase 4
                        generated_volume = generator(z, phase_emb)
                        real_score = discriminator(target_volume)
                        fake_score = discriminator(generated_volume.detach())
                        d_loss = gan_loss(real_score, torch.ones_like(real_score)) + \
                                 gan_loss(fake_score, torch.zeros_like(fake_score))
                        fake_score = discriminator(generated_volume)  # For generator
                        g_loss = l1_loss(generated_volume, target_volume) * 10.0 + \
                                 gan_loss(fake_score, torch.ones_like(fake_score))
                    
                    z_reversed = GradientReversalLayer.apply(z, lambda_)
                    phase_pred = phase_detector(z_reversed)
                    p_loss = phase_loss(phase_pred, true_phase_label)
                
                if epoch >= 90:
                    # Update disc and enc_gen in phase 4
                    if use_mixed_precision:
                        scaler.scale(d_loss).backward()
                        scaler.step(optimizer_disc)
                        scaler.update()
                        scaler.scale(g_loss).backward()
                        scaler.step(optimizer_enc_gen)
                        scaler.update()
                    else:
                        d_loss.backward()
                        optimizer_disc.step()
                        g_loss.backward()
                        optimizer_enc_gen.step()
                    epoch_g_loss += g_loss.item()
                    epoch_d_loss += d_loss.item()
                    epoch_p_loss += p_loss.item()
                    _, predicted = torch.max(phase_pred.data, 1)
                    phase_correct += (predicted == true_phase_label).sum().item()
                    phase_total += true_phase_label.size(0)

        # Calculate epoch metrics (adjust for phases)
        num_batches = len(data_loader)
        if epoch < 60:  # Phases 1 and 2
            if epoch < 30:
                epoch_p_loss /= num_batches
                phase_accuracy = phase_correct / phase_total if phase_total > 0 else 0
                metrics["p_losses"].append(epoch_p_loss)
                metrics["phase_accuracy"].append(phase_accuracy)
                print(f"Phase Detector Loss: {epoch_p_loss:.4f}, Accuracy: {phase_accuracy:.4f}")
            else:
                epoch_g_loss /= num_batches
                epoch_d_loss /= num_batches
                metrics["g_losses"].append(epoch_g_loss)
                metrics["d_losses"].append(epoch_d_loss)
                print(f"Generator Loss: {epoch_g_loss:.4f}, Discriminator Loss: {epoch_d_loss:.4f}")
        else:  # Phases 3 and 4
            epoch_p_loss /= num_batches
            phase_accuracy = phase_correct / phase_total if phase_total > 0 else 0
            metrics["p_losses"].append(epoch_p_loss)
            metrics["phase_accuracy"].append(phase_accuracy)
            print(f"Phase Detector Loss: {epoch_p_loss:.4f}, Accuracy: {phase_accuracy:.4f}, Lambda: {lambda_:.4f}")
            if epoch >= 90:
                epoch_g_loss /= num_batches
                epoch_d_loss /= num_batches
                metrics["g_losses"].append(epoch_g_loss)
                metrics["d_losses"].append(epoch_d_loss)
                print(f"Generator Loss: {epoch_g_loss:.4f}, Discriminator Loss: {epoch_d_loss:.4f}")

        # Updated metrics calculation
        num_batches = len(data_loader)
        if epoch < 30:
            epoch_p_loss /= num_batches
            phase_accuracy = phase_correct / phase_total if phase_total > 0 else 0
            print(f"Phase Detector Loss: {epoch_p_loss:.4f}, Accuracy: {phase_accuracy:.4f}")
        elif epoch < 80:
            if sub_epoch < 5:
                epoch_g_loss /= num_batches
                print(f"Generator Loss: {epoch_g_loss:.4f}")
            else:
                epoch_d_loss /= num_batches
                print(f"Discriminator Loss: {epoch_d_loss:.4f}")
        else:
            epoch_p_loss /= num_batches
            phase_accuracy = phase_correct / phase_total if phase_total > 0 else 0
            print(f"Phase Detector Loss: {epoch_p_loss:.4f}, Accuracy: {phase_accuracy:.4f}, Lambda: {lambda_:.4f}")
            if epoch >= 90:
                epoch_g_loss /= num_batches
                epoch_d_loss /= num_batches
                metrics["g_losses"].append(epoch_g_loss)
                metrics["d_losses"].append(epoch_d_loss)
                print(f"Generator Loss: {epoch_g_loss:.4f}, Discriminator Loss: {epoch_d_loss:.4f}")

        # Save checkpoints every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'phase_detector_state_dict': phase_detector.state_dict(),
                'optimizer_enc_gen_state_dict': optimizer_enc_gen.state_dict(),
                'optimizer_disc_state_dict': optimizer_disc.state_dict(),
                'optimizer_phase_state_dict': optimizer_phase.state_dict(),
                'metrics': metrics,
                'encoder_config': encoder_config 
            }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))
            
            # Save individual model weights for easier loading
            torch.save(encoder.state_dict(), os.path.join(checkpoint_dir, f"encoder_epoch_{epoch+1}.pth"))
            torch.save(generator.state_dict(), os.path.join(checkpoint_dir, f"generator_epoch_{epoch+1}.pth"))
            torch.save(phase_detector.state_dict(), os.path.join(checkpoint_dir, f"phase_detector_epoch_{epoch+1}.pth"))
            
        # After training loop
        if validation_loader is not None:
            print("Running validation...")
            encoder.eval()
            generator.eval()
            discriminator.eval()
            phase_detector.eval()
            
            val_g_loss = 0.0
            val_d_loss = 0.0
            val_p_loss = 0.0
            val_phase_correct = 0
            val_phase_total = 0
            val_batches = 0
            
            with torch.no_grad():
                for val_batch in tqdm(validation_loader, desc="Validation"):
                    input_volume = val_batch["input_volume"].to(device)
                    target_volume = val_batch["target_volume"].to(device)
                    phase_label = val_batch["target_phase"].to(device)
                    true_phase_label = val_batch["input_phase"].to(device)
                    
                    # Forward pass
                    z = encoder(input_volume)
                    phase_emb = create_phase_embeddings(phase_label, dim=32, device=device)
                    generated_volume = generator(z, phase_emb)
                    phase_pred = phase_detector(z)
                    
                    # Calculate losses
                    val_g_loss += l1_loss(generated_volume, target_volume).item()
                    real_score = discriminator(target_volume)
                    fake_score = discriminator(generated_volume)
                    d_loss_real = gan_loss(real_score, torch.ones_like(real_score))
                    d_loss_fake = gan_loss(fake_score, torch.zeros_like(fake_score))
                    val_d_loss += ((d_loss_real + d_loss_fake) / 2).item()
                    val_p_loss += phase_loss(phase_pred, true_phase_label).item()
                    
                    # Phase accuracy
                    _, predicted = torch.max(phase_pred.data, 1)
                    val_phase_correct += (predicted == true_phase_label).sum().item()
                    val_phase_total += true_phase_label.size(0)
                    
                    val_batches += 1
            
            val_g_loss /= val_batches
            val_d_loss /= val_batches
            val_p_loss /= val_batches
            val_p_accuracy = val_phase_correct / val_phase_total if val_phase_total > 0 else 0
            
            print(f"Validation - G_loss: {val_g_loss:.4f}, D_loss: {val_d_loss:.4f}, P_loss: {val_p_loss:.4f}, Phase Accuracy: {val_p_accuracy:.4f}")
            
            # Add to metrics
            metrics['val_g_losses'].append(val_g_loss)
            metrics['val_d_losses'].append(val_d_loss)
            metrics['val_p_losses'].append(val_p_loss)
            metrics['val_phase_accuracy'].append(val_p_accuracy)
        
        # Save metrics to CSV
        csv_path = os.path.join(checkpoint_dir, 'metrics.csv')
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if epoch == 0:
                writer.writerow(['epoch'] + list(metrics.keys()))
            row = [epoch + 1] + [metrics[k][-1] if metrics[k] else 0 for k in metrics.keys()]
            writer.writerow(row)
    return metrics
