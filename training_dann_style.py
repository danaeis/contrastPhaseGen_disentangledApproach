# training_dann_style.py - Optimized DANN-Style Training

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import numpy as np
from utils import GradientReversalLayer, get_phase_embedding

def create_simple_phase_embedding(phase_labels, dim=32, device='cuda'):
    """Simple one-hot phase embedding with padding"""
    if isinstance(phase_labels, (list, np.ndarray)):
        phase_labels = torch.tensor(phase_labels, device=device)
    elif phase_labels.device != device:
        phase_labels = phase_labels.to(device)
    
    one_hot = F.one_hot(phase_labels, num_classes=4).float()
    if dim > 4:
        padding = torch.zeros(len(phase_labels), dim - 4, device=device)
        return torch.cat([one_hot, padding], dim=1)
    return one_hot[:, :dim]

def get_dann_lambda(epoch, max_epochs, schedule='adaptive'):
    """DANN lambda scheduling"""
    p = epoch / max_epochs
    
    if schedule == 'adaptive':
        return min(1.0, p * 2)  # Linear growth, capped at 1.0
    elif schedule == 'exp':
        return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0  # Original DANN formula
    else:  # linear
        return p

class DANNTrainer:
    """Optimized DANN-style trainer"""
    
    def __init__(self, device="cuda", use_mixed_precision=True):
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.scaler = GradScaler() if use_mixed_precision else None
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss() 
        self.ce_loss = nn.CrossEntropyLoss()
    
    def setup_optimizers(self, encoder, generator, discriminator, phase_detector):
        """Setup DANN-style optimizers"""
        return {
            'encoder': optim.Adam(encoder.parameters(), lr=1e-4, betas=(0.5, 0.999)),
            'generator': optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999)),
            'discriminator': optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999)),
            'phase_detector': optim.Adam(phase_detector.parameters(), lr=5e-5, betas=(0.5, 0.999))
        }
    
    def dann_training_step(self, batch, models, optimizers, epoch, max_epochs):
        """Single DANN training step"""
        encoder, generator, discriminator, phase_detector = models
        
        # Prepare data
        input_vol = batch["input_path"].to(self.device)
        target_vol = batch["target_path"].to(self.device)
        input_phase = batch["input_phase"].to(self.device)
        target_phase = batch["target_phase"].to(self.device)
        
        # DANN lambda scheduling
        lambda_grl = get_dann_lambda(epoch, max_epochs, 'adaptive')
        
        losses = {}
        
        with autocast(device_type="cuda", enabled=self.use_mixed_precision):
            # 1. Feature extraction
            features = encoder(input_vol)
            
            # 2. Generation
            phase_emb = create_simple_phase_embedding(target_phase, dim=32, device=self.device)
            generated = generator(features, phase_emb)
            
            # 3. Discrimination
            real_scores = discriminator(target_vol)
            fake_scores = discriminator(generated)
            
            # 4. Phase prediction (normal and reversed)
            phase_pred = phase_detector(features)
            reversed_features = GradientReversalLayer.apply(features, lambda_grl)
            confusion_pred = phase_detector(reversed_features)
            
            # Compute individual losses
            recon_loss = self.l1_loss(generated, target_vol)
            d_real_loss = self.bce_loss(real_scores, torch.ones_like(real_scores))
            d_fake_loss = self.bce_loss(fake_scores, torch.zeros_like(fake_scores))
            d_loss = (d_real_loss + d_fake_loss) * 0.5
            g_adv_loss = self.bce_loss(fake_scores, torch.ones_like(fake_scores))
            phase_loss = self.ce_loss(phase_pred, input_phase)
            confusion_loss = self.ce_loss(confusion_pred, input_phase)
            
            # Combined losses for each model
            generator_loss = recon_loss * 100.0 + g_adv_loss * 0.1
            encoder_loss = recon_loss * 10.0 + confusion_loss * lambda_grl
        
        # Backward passes
        self._backward_step('generator', generator_loss, optimizers)
        self._backward_step('discriminator', d_loss, optimizers)
        self._backward_step('encoder', encoder_loss, optimizers)
        self._backward_step('phase_detector', phase_loss, optimizers)
        
        # Track metrics
        with torch.no_grad():
            _, phase_pred_labels = torch.max(phase_pred, 1)
            _, confusion_pred_labels = torch.max(confusion_pred, 1)
            
            losses = {
                'reconstruction': recon_loss.item(),
                'generator': generator_loss.item(),
                'discriminator': d_loss.item(),
                'phase': phase_loss.item(),
                'confusion': confusion_loss.item(),
                'phase_acc': (phase_pred_labels == input_phase).float().mean().item(),
                'confusion_acc': (confusion_pred_labels != input_phase).float().mean().item(),
                'lambda_grl': lambda_grl
            }
        
        return losses
    
    def _backward_step(self, model_name, loss, optimizers):
        """Efficient backward step"""
        optimizer = optimizers[model_name]
        optimizer.zero_grad()
        
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            # Get parameters for this optimizer
            params = [p for group in optimizer.param_groups for p in group['params']]
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            params = [p for group in optimizer.param_groups for p in group['params']]
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

def train_dann_style_contrast_generation(train_loader, encoder, generator, discriminator,
                                        phase_detector, num_epochs=100, device="cuda",
                                        checkpoint_dir="checkpoints", use_mixed_precision=True,
                                        validation_loader=None, encoder_config=None,
                                        encoder_type="simple_cnn"):
    """Optimized DANN-style training"""
    
    print(f"🚀 Starting optimized DANN-style training for {num_epochs} epochs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = DANNTrainer(device, use_mixed_precision)
    
    # Move models to device and set to training mode
    models = [encoder.to(device), generator.to(device), discriminator.to(device), phase_detector.to(device)]
    for model in models:
        model.train()
    
    # Setup optimizers
    optimizers = trainer.setup_optimizers(*models)
    
    # Setup schedulers
    schedulers = {
        name: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
        for name, opt in optimizers.items()
    }
    
    # Metrics tracking
    metrics = {
        "reconstruction_losses": [],
        "generator_losses": [],
        "discriminator_losses": [],
        "phase_classification_losses": [],
        "phase_confusion_losses": [],
        "phase_accuracies": [],
        "confusion_accuracies": []
    }
    
    # Track best metrics for model saving
    best_reconstruction_loss = float('inf')
    best_psnr = 0.0
    best_ssim = 0.0
    
    print(f"\n{'='*60}")
    print("DANN-STYLE SIMULTANEOUS TRAINING WITH IMAGE QUALITY METRICS")
    print(f"{'='*60}")
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # Training loop
        progress_bar = tqdm(train_loader, desc=f"DANN Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                step_losses = trainer.dann_training_step(batch, models, optimizers, epoch, num_epochs)
                epoch_losses.append(step_losses)
                
                # Update progress bar
                if batch_idx % 50 == 0:
                    progress_bar.set_postfix({
                        'recon': f"{step_losses['reconstruction']:.4f}",
                        'phase_acc': f"{step_losses['phase_acc']:.3f}",
                        'conf_acc': f"{step_losses['confusion_acc']:.3f}",
                        'λ': f"{step_losses['lambda_grl']:.3f}"
                    })
                    
            except Exception as e:
                print(f"⚠️ Batch {batch_idx} error: {e}")
                continue
        
        # Calculate epoch averages
        if epoch_losses:
            avg_losses = {
                key: np.mean([loss[key] for loss in epoch_losses])
                for key in epoch_losses[0].keys()
            }
            
            # Store metrics
            metrics["reconstruction_losses"].append(avg_losses['reconstruction'])
            metrics["generator_losses"].append(avg_losses['generator'])
            metrics["discriminator_losses"].append(avg_losses['discriminator'])
            metrics["phase_classification_losses"].append(avg_losses['phase'])
            metrics["phase_confusion_losses"].append(avg_losses['confusion'])
            metrics["phase_accuracies"].append(avg_losses['phase_acc'])
            metrics["confusion_accuracies"].append(avg_losses['confusion_acc'])
            
            # Print progress
            if epoch % 10 == 0:
                print(f"\nEpoch {epoch+1}: λ_grl = {avg_losses['lambda_grl']:.3f}")
                print(f"  Reconstruction: {avg_losses['reconstruction']:.6f}")
                print(f"  Phase Acc: {avg_losses['phase_acc']:.4f}")
                print(f"  Confusion Acc: {avg_losses['confusion_acc']:.4f}")
                
                # DANN success indicator
                if avg_losses['confusion_acc'] > 0.4:
                    print("  ✅ DANN: Encoder learning phase-invariant features!")
            
            # Save best reconstruction loss model
            if avg_losses['reconstruction'] < best_reconstruction_loss:
                best_reconstruction_loss = avg_losses['reconstruction']
                torch.save({
                    'epoch': epoch + 1,
                    'encoder_state_dict': encoder.state_dict(),
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'phase_detector_state_dict': phase_detector.state_dict(),
                    'reconstruction_loss': best_reconstruction_loss,
                    'metrics': avg_losses,
                    'encoder_config': encoder_config
                }, os.path.join(checkpoint_dir, 'dann_best_reconstruction.pth'))
        
        # Update schedulers
        for scheduler in schedulers.values():
            scheduler.step()
        
        # Comprehensive validation with image quality metrics
        if validation_loader and epoch % 15 == 0:
            val_results, quality_metrics_dict = comprehensive_dann_validation(
                validation_loader, models, device, use_mixed_precision, checkpoint_dir
            )
            
            # Track best PSNR and SSIM models
            if quality_metrics_dict:
                current_psnr = quality_metrics_dict.get('psnr', 0)
                current_ssim = quality_metrics_dict.get('ssim', 0)
                
                if current_psnr > best_psnr:
                    best_psnr = current_psnr
                    torch.save({
                        'epoch': epoch + 1,
                        'encoder_state_dict': encoder.state_dict(),
                        'generator_state_dict': generator.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(),
                        'phase_detector_state_dict': phase_detector.state_dict(),
                        'psnr': best_psnr,
                        'quality_metrics': quality_metrics_dict,
                        'encoder_config': encoder_config
                    }, os.path.join(checkpoint_dir, 'dann_best_psnr.pth'))
                    print(f"  🏆 New best PSNR: {best_psnr:.4f} dB")
                
                if current_ssim > best_ssim:
                    best_ssim = current_ssim
                    torch.save({
                        'epoch': epoch + 1,
                        'encoder_state_dict': encoder.state_dict(),
                        'generator_state_dict': generator.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(),
                        'phase_detector_state_dict': phase_detector.state_dict(),
                        'ssim': best_ssim,
                        'quality_metrics': quality_metrics_dict,
                        'encoder_config': encoder_config
                    }, os.path.join(checkpoint_dir, 'dann_best_ssim.pth'))
                    print(f"  🏆 New best SSIM: {best_ssim:.4f}")
        
        # Regular checkpoints
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'phase_detector_state_dict': phase_detector.state_dict(),
                'metrics': metrics,
                'encoder_config': encoder_config,
                'encoder_type': encoder_type
            }, os.path.join(checkpoint_dir, f"dann_checkpoint_epoch_{epoch+1}.pth"))
    
    # Final checkpoint
    torch.save({
        'epoch': num_epochs,
        'encoder_state_dict': encoder.state_dict(),
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'phase_detector_state_dict': phase_detector.state_dict(),
        'metrics': metrics,
        'encoder_config': encoder_config,
        'encoder_type': encoder_type,
        'training_type': 'dann_style'
    }, os.path.join(checkpoint_dir, "dann_final_checkpoint.pth"))
    
    print(f"\n✅ DANN-style training completed!")
    print(f"📊 Training Summary:")
    print(f"   Best Reconstruction Loss: {best_reconstruction_loss:.6f}")
    print(f"   Best PSNR: {best_psnr:.4f} dB")
    print(f"   Best SSIM: {best_ssim:.4f}")
    
    if metrics["confusion_accuracies"]:
        final_confusion_acc = metrics["confusion_accuracies"][-1]
        if final_confusion_acc > 0.4:
            print("🎯 DANN Success: Model learned phase-invariant features!")
        else:
            print("⚠️  DANN Warning: Low confusion rate - consider tuning λ_grl")
    
    # Final comprehensive validation
    if validation_loader:
        print(f"\n🔍 Final Comprehensive Validation:")
        final_val_results, final_quality_metrics = comprehensive_dann_validation(
            validation_loader, models, device, use_mixed_precision, checkpoint_dir
        )
        
        # Clinical readiness assessment
        if final_quality_metrics:
            final_psnr = final_quality_metrics.get('psnr', 0)
            final_ssim = final_quality_metrics.get('ssim', 0)
            
            print(f"\n🏥 Final Clinical Assessment:")
            if final_psnr >= 30 and final_ssim >= 0.8:
                print("   🟢 Model is READY for clinical evaluation")
            elif final_psnr >= 25 and final_ssim >= 0.7:
                print("   🟡 Model shows PROMISE for clinical application")
            else:
                print("   🔴 Model needs IMPROVEMENT before clinical use")
            
            print(f"   Final Image Quality Metrics:")
            print(f"     PSNR: {final_psnr:.4f} dB")
            print(f"     SSIM: {final_ssim:.4f}")
            print(f"     MS-SSIM: {final_quality_metrics.get('ms_ssim', 0):.4f}")
            print(f"     NMSE: {final_quality_metrics.get('nmse', 0):.6f}")
    
    return metrics

def quick_dann_validation(val_loader, models, device, use_mixed_precision):
    """Quick DANN validation"""
    encoder, generator, discriminator, phase_detector = models
    
    for model in models:
        model.eval()
    
    total_recon = 0
    total_phase_acc = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if num_batches >= 10:
                break
            
            try:
                input_vol = batch["input_path"].to(device)
                target_vol = batch["target_path"].to(device)
                input_phase = batch["input_phase"].to(device)
                target_phase = batch["target_phase"].to(device)
                
                features = encoder(input_vol)
                phase_emb = create_simple_phase_embedding(target_phase, dim=32, device=device)
                generated = generator(features, phase_emb)
                phase_pred = phase_detector(features)
                
                recon_loss = nn.L1Loss()(generated, target_vol)
                _, pred = torch.max(phase_pred, 1)
                phase_acc = (pred == input_phase).float().mean()
                
                total_recon += recon_loss.item()
                total_phase_acc += phase_acc.item()
                num_batches += 1
                
            except Exception:
                continue
    
    for model in models:
        model.train()
    
    return {
        'val_recon_loss': total_recon / max(num_batches, 1),
        'val_phase_acc': total_phase_acc / max(num_batches, 1)
    }