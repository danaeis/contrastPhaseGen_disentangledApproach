# training.py - Optimized Sequential Training

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import numpy as np
from utils import get_phase_embedding

def create_phase_embeddings(phase_labels, dim=32, device='cuda', embedding_type='simple'):
    """Unified phase embedding creation"""
    if isinstance(phase_labels, (list, np.ndarray)):
        phase_labels = torch.tensor(phase_labels, device=device)
    elif phase_labels.device != device:
        phase_labels = phase_labels.to(device)
    
    if embedding_type == 'simple':
        # One-hot encoding padded to required dimension
        one_hot = torch.nn.functional.one_hot(phase_labels, num_classes=4).float()
        if dim > 4:
            padding = torch.zeros(len(phase_labels), dim - 4, device=device)
            return torch.cat([one_hot, padding], dim=1)
        return one_hot[:, :dim]
    else:  # sinusoidal
        embeddings = []
        for phase in phase_labels:
            emb = get_phase_embedding(phase.item(), dim=dim, device=device)
            embeddings.append(emb)
        return torch.stack(embeddings)

def train_phase(train_loader, models, optimizers, phase_config, device, use_mixed_precision=True):
    """Generic phase training function"""
    losses = []
    scaler = GradScaler() if use_mixed_precision else None
    
    for epoch in range(phase_config['epochs']):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f"{phase_config['name']} Epoch {epoch+1}", leave=False):
            try:
                batch_loss = phase_config['train_step'](batch, models, optimizers, device, use_mixed_precision, scaler)
                epoch_loss += batch_loss
                num_batches += 1
            except Exception as e:
                print(f"⚠️ Batch error: {e}")
                continue
        
        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"  {phase_config['name']} Epoch {epoch+1}: Loss = {avg_loss:.6f}")
        
        # Update schedulers
        for scheduler in optimizers.get('schedulers', []):
            scheduler.step(avg_loss)
    
    return losses

def pretrain_encoder_generator_step(batch, models, optimizers, device, use_mixed_precision, scaler):
    """Encoder + Generator pretraining step"""
    encoder, generator = models['encoder'], models['generator']
    optimizer_enc, optimizer_gen = optimizers['encoder'], optimizers['generator']
    
    input_volume = batch["input_path"].to(device)
    target_volume = batch["target_path"].to(device)
    target_phase = batch["target_phase"].to(device)
    
    optimizer_enc.zero_grad()
    optimizer_gen.zero_grad()
    
    with autocast(device_type="cuda", enabled=use_mixed_precision):
        z = encoder(input_volume)
        phase_emb = create_phase_embeddings(target_phase, dim=32, device=device)
        generated = generator(z, phase_emb)
        loss = nn.L1Loss()(generated, target_volume)
    
    if use_mixed_precision:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer_enc)
        scaler.unscale_(optimizer_gen)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        scaler.step(optimizer_enc)
        scaler.step(optimizer_gen)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        optimizer_enc.step()
        optimizer_gen.step()
    
    return loss.item()

def train_phase_detector_step(batch, models, optimizers, device, use_mixed_precision, scaler):
    """Phase detector training step"""
    encoder, phase_detector = models['encoder'], models['phase_detector']
    optimizer_phase = optimizers['phase_detector']
    
    input_volume = batch["input_path"].to(device)
    input_phase = batch["input_phase"].to(device)
    
    optimizer_phase.zero_grad()
    
    with autocast(device_type="cuda", enabled=use_mixed_precision):
        with torch.no_grad():
            features = encoder(input_volume)
        phase_pred = phase_detector(features)
        loss = nn.CrossEntropyLoss()(phase_pred, input_phase)
    
    if use_mixed_precision:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer_phase)
        torch.nn.utils.clip_grad_norm_(phase_detector.parameters(), max_norm=1.0)
        scaler.step(optimizer_phase)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(phase_detector.parameters(), max_norm=1.0)
        optimizer_phase.step()
    
    return loss.item()

def train_disentangled_step(batch, models, optimizers, device, use_mixed_precision, scaler):
    """Disentangled generation training step"""
    encoder, generator, discriminator, phase_detector = (
        models['encoder'], models['generator'], models['discriminator'], models['phase_detector']
    )
    
    input_volume = batch["input_path"].to(device)
    target_volume = batch["target_path"].to(device)
    target_phase = batch["target_phase"].to(device)
    
    # Train Generator + Encoder
    optimizers['generator'].zero_grad()
    optimizers['encoder'].zero_grad()
    
    with autocast(device_type="cuda", enabled=use_mixed_precision):
        z = encoder(input_volume)
        phase_emb = create_phase_embeddings(target_phase, dim=32, device=device)
        generated = generator(z, phase_emb)
        
        # Losses
        recon_loss = nn.L1Loss()(generated, target_volume)
        fake_scores = discriminator(generated)
        adv_loss = nn.BCEWithLogitsLoss()(fake_scores, torch.ones_like(fake_scores))
        
        g_loss = recon_loss * 100.0 + adv_loss * 1.0
    
    if use_mixed_precision:
        scaler.scale(g_loss).backward()
        scaler.unscale_(optimizers['generator'])
        scaler.unscale_(optimizers['encoder'])
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        scaler.step(optimizers['generator'])
        scaler.step(optimizers['encoder'])
        scaler.update()
    else:
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        optimizers['generator'].step()
        optimizers['encoder'].step()
    
    # Train Discriminator
    optimizers['discriminator'].zero_grad()
    
    with autocast(device_type="cuda", enabled=use_mixed_precision):
        real_scores = discriminator(target_volume)
        fake_scores = discriminator(generated.detach())
        
        d_loss = (nn.BCEWithLogitsLoss()(real_scores, torch.ones_like(real_scores)) +
                  nn.BCEWithLogitsLoss()(fake_scores, torch.zeros_like(fake_scores))) / 2
    
    if use_mixed_precision:
        scaler.scale(d_loss).backward()
        scaler.unscale_(optimizers['discriminator'])
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        scaler.step(optimizers['discriminator'])
        scaler.update()
    else:
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        optimizers['discriminator'].step()
    
    return g_loss.item()

def train_contrast_phase_generation(train_loader, encoder, generator, discriminator, 
                                   phase_detector, num_epochs=80, device="cuda", 
                                   checkpoint_dir="checkpoints", use_mixed_precision=True,
                                   validation_loader=None, encoder_config=None, 
                                   encoder_type="simple_cnn"):
    """Optimized sequential training"""
    
    print(f"🚀 Starting optimized sequential training for {num_epochs} epochs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Move models to device
    models = {
        'encoder': encoder.to(device),
        'generator': generator.to(device), 
        'discriminator': discriminator.to(device),
        'phase_detector': phase_detector.to(device)
    }
    
    # Setup optimizers
    optimizers = {
        'encoder': optim.AdamW(encoder.parameters(), lr=1e-4),
        'generator': optim.AdamW(generator.parameters(), lr=2e-4),
        'discriminator': optim.AdamW(discriminator.parameters(), lr=1e-4),
        'phase_detector': optim.AdamW(phase_detector.parameters(), lr=5e-5)
    }
    
    # Phase configurations
    encoder_epochs = max(1, num_epochs // 10)
    phase_epochs = max(1, num_epochs // 5) 
    gen_epochs = num_epochs - encoder_epochs - phase_epochs
    
    phases = [
        {
            'name': 'Encoder+Generator Pretrain',
            'epochs': encoder_epochs,
            'models': {'encoder': encoder, 'generator': generator},
            'train_step': pretrain_encoder_generator_step
        },
        {
            'name': 'Phase Detector',
            'epochs': phase_epochs,
            'models': {'encoder': encoder, 'phase_detector': phase_detector},
            'train_step': train_phase_detector_step
        },
        {
            'name': 'Disentangled Generation', 
            'epochs': gen_epochs,
            'models': models,
            'train_step': train_disentangled_step
        }
    ]
    
    # Train each phase
    all_metrics = {'pretrain_losses': [], 'phase_losses': [], 'g_losses': []}
    
    for phase in phases:
        print(f"\n{'='*60}")
        print(f"PHASE: {phase['name']} ({phase['epochs']} epochs)")
        print(f"{'='*60}")
        
        losses = train_phase(train_loader, models, optimizers, phase, device, use_mixed_precision)
        
        # Store metrics
        if 'Pretrain' in phase['name']:
            all_metrics['pretrain_losses'] = losses
        elif 'Phase' in phase['name']:
            all_metrics['phase_losses'] = losses
        else:
            all_metrics['g_losses'] = losses
        
        # Save checkpoint after each phase
        torch.save({
            'phase': phase['name'],
            'encoder_state_dict': encoder.state_dict(),
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'phase_detector_state_dict': phase_detector.state_dict(),
            'metrics': all_metrics,
            'encoder_config': encoder_config
        }, os.path.join(checkpoint_dir, f"checkpoint_{phase['name'].lower().replace(' ', '_')}.pth"))
    
    # Final validation
    if validation_loader:
        val_metrics = quick_validation(validation_loader, models, device, use_mixed_precision)
        all_metrics['val_metrics'] = val_metrics
        print(f"🔍 Final Validation: {val_metrics}")
    
    # Save final checkpoint
    torch.save({
        'epoch': num_epochs,
        'encoder_state_dict': encoder.state_dict(),
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'phase_detector_state_dict': phase_detector.state_dict(),
        'metrics': all_metrics,
        'encoder_config': encoder_config,
        'encoder_type': encoder_type
    }, os.path.join(checkpoint_dir, "final_checkpoint.pth"))
    
    print(f"✅ Sequential training completed!")
    return all_metrics

def comprehensive_validation(val_loader, models, device, use_mixed_precision, checkpoint_dir):
    """Comprehensive validation with image quality metrics"""
    try:
        from image_quality_metrics import ImageQualityMetrics, ValidationMetricsTracker, compute_metrics_for_volume_pair
        use_quality_metrics = True
    except ImportError:
        print("⚠️ Image quality metrics not available - using basic validation")
        use_quality_metrics = False
    
    encoder, generator, phase_detector = models['encoder'], models['generator'], models['phase_detector']
    
    for model in models.values():
        model.eval()
    
    # Initialize metrics tracking
    total_recon_loss = 0
    total_phase_acc = 0
    num_batches = 0
    
    if use_quality_metrics:
        quality_metrics = ImageQualityMetrics(device=device)
        metrics_tracker = ValidationMetricsTracker(
            save_path=os.path.join(checkpoint_dir, 'validation_metrics.csv'),
            metrics_list=['psnr', 'ssim', 'ms_ssim', 'nmse', 'ncc', 'mi']
        )
        all_quality_metrics = []
    
    print("🔍 Running comprehensive validation...")
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            if num_batches >= 15:  # Limit validation batches for speed
                break
            
            try:
                input_vol = batch["input_path"].to(device)
                target_vol = batch["target_path"].to(device) 
                target_phase = batch["target_phase"].to(device)
                input_phase = batch["input_phase"].to(device)
                
                features = encoder(input_vol)
                phase_emb = create_phase_embeddings(target_phase, dim=32, device=device)
                generated = generator(features, phase_emb)
                phase_pred = phase_detector(features)
                
                # Ensure volumes are in [0, 1] range
                generated = torch.clamp(generated, 0, 1)
                target_vol = torch.clamp(target_vol, 0, 1)
                
                # Basic metrics
                recon_loss = nn.L1Loss()(generated, target_vol)
                _, pred = torch.max(phase_pred, 1)
                phase_acc = (pred == input_phase).float().mean()
                
                total_recon_loss += recon_loss.item()
                total_phase_acc += phase_acc.item()
                
                # Image quality metrics
                if use_quality_metrics:
                    batch_quality_metrics = quality_metrics.compute_all_metrics(generated, target_vol)
                    all_quality_metrics.append(batch_quality_metrics)
                
                num_batches += 1
                
            except Exception as e:
                print(f"⚠️ Validation batch error: {e}")
                continue
    
    # Calculate averages
    validation_results = {
        'reconstruction_loss': total_recon_loss / max(num_batches, 1),
        'phase_accuracy': total_phase_acc / max(num_batches, 1)
    }
    
    if use_quality_metrics and all_quality_metrics:
        # Average all quality metrics
        averaged_quality = {}
        for metric_name in ['psnr', 'ssim', 'ms_ssim', 'nmse', 'ncc', 'mi']:
            values = [m.get(metric_name, 0) for m in all_quality_metrics if metric_name in m]
            if values:
                averaged_quality[metric_name] = np.mean(values)
            else:
                averaged_quality[metric_name] = 0
        
        validation_results.update(averaged_quality)
        
        # Update metrics tracker
        metrics_tracker.update(1, averaged_quality)
        
        # Print comprehensive results
        print(f"📊 Validation Results:")
        print(f"   🎯 PSNR: {averaged_quality.get('psnr', 0):.4f} dB")
        print(f"   🔍 SSIM: {averaged_quality.get('ssim', 0):.4f}")
        print(f"   📈 MS-SSIM: {averaged_quality.get('ms_ssim', 0):.4f}")
        print(f"   📉 NMSE: {averaged_quality.get('nmse', 0):.6f}")
        print(f"   🤝 NCC: {averaged_quality.get('ncc', 0):.4f}")
        print(f"   🧠 MI: {averaged_quality.get('mi', 0):.4f}")
        
        # Clinical assessment
        psnr = averaged_quality.get('psnr', 0)
        ssim = averaged_quality.get('ssim', 0)
        
        if psnr >= 30 and ssim >= 0.8:
            clinical_status = "🟢 CLINICALLY ACCEPTABLE"
        elif psnr >= 25 and ssim >= 0.7:
            clinical_status = "🟡 SHOWS CLINICAL PROMISE"
        else:
            clinical_status = "🔴 NEEDS CLINICAL IMPROVEMENT"
        
        print(f"🏥 Clinical Assessment: {clinical_status}")
        validation_results['clinical_status'] = clinical_status
    
    for model in models.values():
        model.train()
    
    return validation_results

def quick_validation(val_loader, models, device, use_mixed_precision):
    """Quick validation function (fallback)"""
    encoder, generator, phase_detector = models['encoder'], models['generator'], models['phase_detector']
    
    for model in models.values():
        model.eval()
    
    total_recon_loss = 0
    total_phase_acc = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if num_batches >= 5:  # Quick validation
                break
            
            try:
                input_vol = batch["input_path"].to(device)
                target_vol = batch["target_path"].to(device) 
                target_phase = batch["target_phase"].to(device)
                input_phase = batch["input_phase"].to(device)
                
                features = encoder(input_vol)
                phase_emb = create_phase_embeddings(target_phase, dim=32, device=device)
                generated = generator(features, phase_emb)
                phase_pred = phase_detector(features)
                
                recon_loss = nn.L1Loss()(generated, target_vol)
                _, pred = torch.max(phase_pred, 1)
                phase_acc = (pred == input_phase).float().mean()
                
                total_recon_loss += recon_loss.item()
                total_phase_acc += phase_acc.item()
                num_batches += 1
                
            except Exception:
                continue
    
    for model in models.values():
        model.train()
    
    return {
        'reconstruction_loss': total_recon_loss / max(num_batches, 1),
        'phase_accuracy': total_phase_acc / max(num_batches, 1)
    }