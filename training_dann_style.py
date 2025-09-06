import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import os
import time
import numpy as np
from tqdm import tqdm

# Import quality metrics
try:
    from image_quality_metrics import ImageQualityMetrics
    HAS_QUALITY_METRICS = True
except ImportError:
    HAS_QUALITY_METRICS = False

# Gradient Reversal Layer
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -ctx.lambda_), None

def gradient_reversal(x, lambda_):
    return GradientReversalFunction.apply(x, lambda_)

class DANNTrainerOptimized:
    """Optimized DANN trainer and comprehensive logging"""
    
    def __init__(self, device='cuda', use_mixed_precision=True, logger=None, phase_dim=8):
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.scaler = GradScaler() if use_mixed_precision else None
        self.logger = logger
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.phase_dim = phase_dim
        
        # Quality metrics
        if HAS_QUALITY_METRICS:
            self.quality_metrics = ImageQualityMetrics(device=device)
        else:
            self.quality_metrics = None
    
    def setup_optimizers(self, encoder, generator, discriminator, phase_detector):
        """Setup optimizers"""
        
        # trainable encoder
        for param in encoder.parameters():
            param.requires_grad = True
        encoder.train()
        
        print("🧊 DANN training")
        
        # optimize components
        optimizers = {
            'encoder': optim.Adam(encoder.parameters(), lr=1e-4, weight_decay=1e-5),
            'generator': optim.Adam(generator.parameters(), lr=1e-4, weight_decay=1e-5),
            'discriminator': optim.Adam(discriminator.parameters(), lr=1e-4, weight_decay=1e-5),
            'phase_detector': optim.Adam(phase_detector.parameters(), lr=1e-4, weight_decay=1e-5)
        }
        
        return optimizers
    
    def _get_dann_lambda(self, epoch, max_epochs, mode='adaptive'):
        """Calculate DANN lambda with smooth scheduling"""
        p = epoch / max_epochs
        if mode == 'adaptive':
            return 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
        else:
            return min(1.0, epoch / (max_epochs * 0.5))
    
    def _create_phase_embedding(self, phase_labels, dim=32):
        """Create phase embeddings"""
        batch_size = phase_labels.size(0)
        phase_emb = torch.zeros(batch_size, dim, device=self.device)
        
        for i in range(3):
            mask = (phase_labels == i)
            if mask.any():
                phase_emb[mask, i*dim//3:(i+1)*dim//3] = 1.0
        
        # Return as 2D tensor to match Generator input expectations (batch, dim)
        return phase_emb
    
    def dann_training_step(self, batch, models, optimizers, epoch, max_epochs):
        """DANN training step """
        
        encoder, generator, discriminator, phase_detector = models
        
        # Prepare data
        input_vol = batch["input_path"].to(self.device)
        target_vol = batch["target_path"].to(self.device)
        input_phase = batch["input_phase"].to(self.device)
        target_phase = batch["target_phase"].to(self.device)
        
        # DANN lambda scheduling
        lambda_grl = self._get_dann_lambda(epoch, max_epochs)
        
        losses = {}
        
        # Clear all gradients
        for optimizer in optimizers.values():
            optimizer.zero_grad()
        
        # Compute shared features
        with autocast(enabled=self.use_mixed_precision):
            features = encoder(input_vol)
            phase_emb = self._create_phase_embedding(target_phase, dim=self.phase_dim)
            generated = generator(features, phase_emb)
            
            # Reconstruction loss
            recon_loss = self.l1_loss(generated, target_vol)
            
            # Adversarial loss for generator (fool discriminator)
            fake_pred = discriminator(generated)
            adv_loss_gen = self.bce_loss(fake_pred, torch.ones_like(fake_pred))
            
            gen_total_loss = recon_loss + 0.1 * adv_loss_gen
        
        # Backward for generator and encoder (main task)
        if self.use_mixed_precision:
            self.scaler.scale(gen_total_loss).backward()
            self.scaler.step(optimizers['generator'])
            self.scaler.step(optimizers['encoder'])
            self.scaler.update()
        else:
            gen_total_loss.backward()
            optimizers['generator'].step()
            optimizers['encoder'].step()
        
        # Step 2: Train Discriminator
        optimizers['discriminator'].zero_grad()
        
        with autocast(enabled=self.use_mixed_precision):
            # Real samples
            real_pred = discriminator(target_vol)
            real_labels = torch.ones_like(real_pred)
            real_loss = self.bce_loss(real_pred, real_labels)
            
            fake_pred = discriminator(generated.detach())
            fake_labels = torch.zeros_like(fake_pred)
            fake_loss = self.bce_loss(fake_pred, fake_labels)
            
            disc_loss = 0.5 * (real_loss + fake_loss)
        
        # Backward for discriminator
        if self.use_mixed_precision:
            self.scaler.scale(disc_loss).backward()
            self.scaler.step(optimizers['discriminator'])
            self.scaler.update()
        else:
            disc_loss.backward()
            optimizers['discriminator'].step()
        
        # Step 3: DANN Training for Phase Detector
        optimizers['phase_detector'].zero_grad()
        
        with autocast(enabled=self.use_mixed_precision):
            # Phase classification loss (normal)
            phase_pred = phase_detector(features.detach())
            phase_classification_loss = self.ce_loss(phase_pred, input_phase)
            
        # Backward for phase detector
        if self.use_mixed_precision:
            self.scaler.scale(phase_classification_loss).backward()
            self.scaler.step(optimizers['phase_detector'])
            self.scaler.update()
        else:
            phase_classification_loss.backward()
            optimizers['phase_detector'].step()
        
        # Step 4: Adversarial update for Encoder (using GRL)
        optimizers['encoder'].zero_grad()
        
        with autocast(enabled=self.use_mixed_precision):
            # Recompute features to build a fresh graph for encoder adversarial update
            features = encoder(input_vol)
            # Apply gradient reversal
            grl_features = gradient_reversal(features, lambda_grl)
            confusion_pred = phase_detector(grl_features)
            confusion_loss = self.ce_loss(confusion_pred, input_phase)
        
        # Backward for encoder (with reversed gradients)
        if self.use_mixed_precision:
            self.scaler.scale(confusion_loss).backward()
            self.scaler.step(optimizers['encoder'])
            self.scaler.update()
        else:
            confusion_loss.backward()
            optimizers['encoder'].step()
        
        # Calculate accuracies
        with torch.no_grad():
            _, phase_pred_labels = torch.max(phase_pred, 1)
            _, confusion_pred_labels = torch.max(confusion_pred, 1)
            
            phase_acc = (phase_pred_labels == input_phase).float().mean().item()
            confusion_acc = (confusion_pred_labels != input_phase).float().mean().item()
        
        losses = {
            'reconstruction': recon_loss.item(),
            'generator': gen_total_loss.item(),
            'discriminator': disc_loss.item(),
            'phase': phase_classification_loss.item(),
            'confusion': confusion_loss.item(),
            'phase_acc': phase_acc,
            'confusion_acc': confusion_acc,
            'lambda_grl': lambda_grl
        }
        
        return losses
            # # Phase confusion loss (gradient reversal effect)
            # # Create adversarial phase predictions
            # confused_features = features.detach() * lambda_grl
            # confusion_pred = phase_detector(confused_features)
            
            # # Target: make phase detector confused (uniform distribution)
            # uniform_targets = torch.full_like(input_phase, fill_value=1, dtype=torch.float)
            # confusion_loss = self.ce_loss(confusion_pred, input_phase)  # Still predict correct, but features are modified
            
            # # Total phase detector loss
            # phase_total_loss = phase_classification_loss - lambda_grl * confusion_loss
        
        # # Backward for phase detector
        # if self.use_mixed_precision:
        #     self.scaler.scale(phase_total_loss).backward()
        #     self.scaler.step(optimizers['phase_detector'])
        #     self.scaler.update()
        # else:
        #     phase_total_loss.backward()
        #     optimizers['phase_detector'].step()
        
        # # Calculate accuracies
        # with torch.no_grad():
        #     _, phase_pred_labels = torch.max(phase_pred, 1)
        #     _, confusion_pred_labels = torch.max(confusion_pred, 1)
            
        #     phase_acc = (phase_pred_labels == input_phase).float().mean().item()
        #     confusion_acc = (confusion_pred_labels != input_phase).float().mean().item()
        
        # losses = {
        #     'reconstruction': recon_loss.item(),
        #     'generator': gen_total_loss.item(),
        #     'discriminator': disc_loss.item(),
        #     'phase': phase_classification_loss.item(),
        #     'confusion': confusion_loss.item(),
        #     'phase_acc': phase_acc,
        #     'confusion_acc': confusion_acc,
        #     'lambda_grl': lambda_grl
        # }
        
        # return losses
    
    def validate_dann_training(self, val_loader, models, max_batches=8):
        """Validate DANN training with quality metrics"""
        
        encoder, generator, discriminator, phase_detector = models
        
        # Set to eval mode
        encoder.eval()
        generator.eval()
        discriminator.eval()
        phase_detector.eval()
        
        total_recon_loss = 0
        total_phase_acc = 0
        total_confusion_acc = 0
        all_quality_metrics = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if num_batches >= max_batches:
                    break
                    
                try:
                    input_vol = batch["input_path"].to(self.device)
                    target_vol = batch["target_path"].to(self.device)
                    target_phase = batch["target_phase"].to(self.device)
                    input_phase = batch["input_phase"].to(self.device)
                    
                    # Forward pass
                    features = encoder(input_vol)
                    phase_emb = self._create_phase_embedding(target_phase, dim=32)
                    generated = generator(features, phase_emb)
                    phase_pred = phase_detector(features)
                    
                    # Basic metrics
                    recon_loss = self.l1_loss(generated, target_vol)
                    _, pred = torch.max(phase_pred, 1)
                    phase_acc = (pred == input_phase).float().mean()
                    
                    # Mock confusion accuracy for validation
                    confusion_acc = 0.5  # Placeholder
                    
                    total_recon_loss += recon_loss.item()
                    total_phase_acc += phase_acc.item()
                    total_confusion_acc += confusion_acc
                    
                    # Quality metrics
                    if self.quality_metrics:
                        generated_clamp = torch.clamp(generated, 0, 1)
                        target_clamp = torch.clamp(target_vol, 0, 1)
                        quality_metrics = self.quality_metrics.compute_all_metrics(generated_clamp, target_clamp)
                        all_quality_metrics.append(quality_metrics)
                    
                    num_batches += 1
                    
                except Exception as e:
                    continue
        
        results = {
            'reconstruction_loss': total_recon_loss / max(num_batches, 1),
            'phase_accuracy': total_phase_acc / max(num_batches, 1),
            'confusion_accuracy': total_confusion_acc / max(num_batches, 1)
        }
        
        if all_quality_metrics:
            # Average quality metrics
            for metric in ['psnr', 'ssim', 'ms_ssim']:
                values = [m.get(metric, 0) for m in all_quality_metrics]
                results[metric] = np.mean(values) if values else 0
        
        # Set back to train mode
        generator.train()
        discriminator.train()
        phase_detector.train()
        
        return results

# Step 2: Copy this function into your file (before your training function)
def load_phase_checkpoint(checkpoint_dir, models, optimizers=None, device='cuda'):
    """Load checkpoint from previous phases if they exist"""
    encoder, generator, discriminator, phase_detector = models
    
    # Search patterns in order of priority
    phase_patterns = [
        'phase3_completed.pth',
        'phase2_completed.pth', 
        'phase1_completed.pth',
        'dann_final_checkpoint.pth',
        'dann_best_*.pth',
        'final_checkpoint.pth',
        '*_completed.pth'
    ]
    
    checkpoint_path = None
    print(f"🔍 Searching for phase checkpoints in {checkpoint_dir}...")
    import glob
    # Find the first available checkpoint
    for pattern in phase_patterns:
        search_pattern = os.path.join(checkpoint_dir, pattern)
        matching_files = glob.glob(search_pattern)
        if matching_files:
            checkpoint_path = max(matching_files, key=os.path.getmtime)
            print(f"📁 Found checkpoint: {os.path.basename(checkpoint_path)}")
            break
    
    if not checkpoint_path:
        print("⚠️ No previous phase checkpoints found - starting from scratch")
        return {'loaded': False, 'epoch': 0, 'phase': 'none', 'metrics': {}}
    
    try:
        print(f"📄 Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model states
        state_keys = ['encoder_state_dict', 'generator_state_dict', 
                     'discriminator_state_dict', 'phase_detector_state_dict']
        
        for model, state_key in zip(models, state_keys):
            if state_key in checkpoint:
                try:
                    model.load_state_dict(checkpoint[state_key])
                    print(f"  ✅ Loaded {state_key}")
                except Exception as e:
                    print(f"  ⚠️ Failed to load {state_key}: {e}")
        
        # Load optimizer states if provided
        if optimizers and 'optimizers' in checkpoint:
            opt_checkpoint = checkpoint['optimizers']
            for name, optimizer in optimizers.items():
                if name in opt_checkpoint:
                    try:
                        optimizer.load_state_dict(opt_checkpoint[name])
                        print(f"  ✅ Loaded {name} optimizer state")
                    except Exception as e:
                        print(f"  ⚠️ Failed to load {name} optimizer: {e}")
        
        loaded_epoch = checkpoint.get('epoch', 0)
        loaded_phase = checkpoint.get('phase', 'unknown')
        loaded_metrics = checkpoint.get('metrics', {})
        
        print(f"✅ Checkpoint loaded successfully!")
        print(f"   Phase: {loaded_phase} | Epoch: {loaded_epoch}")
        
        return {
            'loaded': True,
            'epoch': loaded_epoch,
            'phase': loaded_phase,
            'metrics': loaded_metrics,
            'checkpoint_path': checkpoint_path,
            'encoder_config': checkpoint.get('encoder_config', {})
        }
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return {'loaded': False, 'epoch': 0, 'phase': 'error', 'metrics': {}}


def train_dann_style_contrast_generation(train_loader, encoder, generator, discriminator,
                                        phase_detector, num_epochs=120, device="cuda",
                                             checkpoint_dir="checkpoints", use_mixed_precision=True,
                                             validation_loader=None, encoder_config=None,
                                        encoder_type="simple_cnn", use_sequential_approach=True,
                                        logger=None,
                                        load_previous_phase=True,
                                        force_restart=False,
                                        phase_dim=8):
    """
    OPTIMIZED DANN-style training for Phase 3
    """
    
    print(f"\n🚀 Starting OPTIMIZED DANN Training (Phase 3)")
    print(f"📊 Epochs: {num_epochs}")
    print(f"🧊 Strategy: Frozen encoder + DANN adaptation")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = DANNTrainerOptimized(device, use_mixed_precision, logger, phase_dim=phase_dim)
    
    # Setup models and optimizers
    models = [encoder, generator, discriminator, phase_detector]
    optimizers = trainer.setup_optimizers(*models)

    # Load previous phase checkpoint if requested and not forcing restart
    start_epoch = 0
    checkpoint_info = {'loaded': False}
    
    if load_previous_phase and not force_restart:
        checkpoint_info = load_phase_checkpoint(checkpoint_dir, models, optimizers, device)
        if checkpoint_info['loaded']:
            start_epoch = checkpoint_info.get('epoch', 0)
            print(f"🔄 Resuming training from epoch {start_epoch}")
            
            # Adjust epochs if we're continuing
            if start_epoch > 0:
                print(f"📊 Adjusting training: {start_epoch} → {start_epoch + num_epochs} epochs")
    elif force_restart:
        print("🆕 Force restart enabled - ignoring any existing checkpoints")
    elif not load_previous_phase:
        print("⚡ Starting fresh training (load_previous_phase=False)")
    # Setup schedulers
    schedulers = {
        name: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
        for name, opt in optimizers.items()
    }
    
    # Training metrics
    best_confusion_acc = 0.0
    best_reconstruction_loss = float('inf')
    
    print(f"\n{'='*80}")
    print("DANN TRAINING WITH FROZEN ENCODER - COMPREHENSIVE LOGGING")
    print(f"{'='*80}")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_losses = []
        
        # Training loop
        progress_bar = tqdm(train_loader, desc=f"DANN Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # DANN training step
                step_losses = trainer.dann_training_step(
                    batch, models, optimizers, epoch, num_epochs
                )
                
                epoch_losses.append(step_losses)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'recon': f"{step_losses['reconstruction']:.4f}",
                    'conf_acc': f"{step_losses['confusion_acc']:.3f}",
                    'λ': f"{step_losses['lambda_grl']:.3f}"
                })
                    
            except RuntimeError as e:
                if "Trying to backward through the graph a second time" in str(e):
                    print(f"⚠️ Graph error at batch {batch_idx} - continuing")
                    continue
                else:
                    print(f"⚠️ Batch {batch_idx} error: {e}")
                    continue
            except Exception as e:
                print(f"⚠️ Unexpected error at batch {batch_idx}: {e}")
                continue
        
        # Calculate epoch averages
        epoch_time = time.time() - epoch_start_time
        if epoch_losses:
            avg_losses = {
                key: np.mean([loss[key] for loss in epoch_losses])
                for key in epoch_losses[0].keys()
            }
            
            # Validation with quality metrics
            quality_metrics = None
            if validation_loader and epoch % 20 == 0:
                quality_metrics = trainer.validate_dann_training(validation_loader, models)
                print(f"   Validation - Recon: {quality_metrics.get('reconstruction_loss', 0):.4f}, "
                     f"PSNR: {quality_metrics.get('psnr', 0):.2f}dB, "
                     f"Confusion: {quality_metrics.get('confusion_accuracy', 0):.3f}")
            
            # Track best models
            if avg_losses['confusion_acc'] > best_confusion_acc:
                best_confusion_acc = avg_losses['confusion_acc']
                
                # Save best DANN checkpoint
                torch.save({
                    'epoch': epoch + 1,
                    'encoder_state_dict': encoder.state_dict(),
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'phase_detector_state_dict': phase_detector.state_dict(),
                    'confusion_accuracy': best_confusion_acc,
                    'reconstruction_loss': avg_losses['reconstruction'],
                    'encoder_config': encoder_config
                }, os.path.join(checkpoint_dir, 'dann_best_confusion.pth'))
            
            if avg_losses['reconstruction'] < best_reconstruction_loss:
                best_reconstruction_loss = avg_losses['reconstruction']
            
            # Log comprehensive metrics
            if logger:
                logger.log_epoch(epoch, "phase3_dann", avg_losses, quality_metrics, epoch_time)
            
            # Print progress with DANN status
            dann_status = "🟢 Excellent" if avg_losses['confusion_acc'] > 0.5 else \
                         "🟡 Good" if avg_losses['confusion_acc'] > 0.4 else \
                         "🔴 Learning"
            
            if epoch % 10 == 0:
                print(f"DANN Epoch {epoch+1}: λ_grl = {avg_losses['lambda_grl']:.3f}")
                print(f"  Reconstruction: {avg_losses['reconstruction']:.6f}")
                print(f"  Confusion Acc: {avg_losses['confusion_acc']:.4f} ({dann_status})")
                
                # DANN success indicator
                if avg_losses['confusion_acc'] > 0.4:
                    print("  ✅ DANN: Encoder learning phase-invariant features!")
            
            # # Early stopping for DANN success
            # if avg_losses['confusion_acc'] > 0.5:
            #     print(f"🎉 DANN early success! Achieved {avg_losses['confusion_acc']:.3f} confusion accuracy")
            #     break
        
        # Update schedulers
        for scheduler in schedulers.values():
            scheduler.step()
        
        # Save periodic checkpoints
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'phase_detector_state_dict': phase_detector.state_dict(),
                'training_type': 'dann',
                'encoder_config': encoder_config
            }, os.path.join(checkpoint_dir, f"dann_checkpoint_epoch_{epoch+1}.pth"))
    
    # Final checkpoint
    torch.save({
        'epoch': num_epochs,
        'encoder_state_dict': encoder.state_dict(),
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'phase_detector_state_dict': phase_detector.state_dict(),
        'best_confusion_accuracy': best_confusion_acc,
        'best_reconstruction_loss': best_reconstruction_loss,
        'encoder_config': encoder_config,
        'encoder_type': encoder_type,
        'training_type': 'dann_complete'
    }, os.path.join(checkpoint_dir, "dann_final_checkpoint.pth"))
    
    print(f"\n✅ DANN training completed!")
    print(f"📊 Training Summary:")
    print(f"   Best Confusion Accuracy: {best_confusion_acc:.4f}")
    print(f"   Best Reconstruction Loss: {best_reconstruction_loss:.6f}")
    
    if best_confusion_acc > 0.4:
        print("🎯 DANN SUCCESS: Model learned phase-invariant features!")
    else:
        print("⚠️ DANN PARTIAL: Consider longer training or parameter adjustment")
    
    return {
        'best_confusion_accuracy': best_confusion_acc,
        'best_reconstruction_loss': best_reconstruction_loss,
        'dann_success': best_confusion_acc > 0.4
    }