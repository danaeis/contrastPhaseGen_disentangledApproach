import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import os
import time
import numpy as np
from tqdm import tqdm
import gc
from collections import defaultdict, deque
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


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
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.phase_dim = phase_dim
        
        # Memory-efficient metrics tracking
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=100))
        
        # Quality metrics
        if HAS_QUALITY_METRICS:
            self.quality_metrics = ImageQualityMetrics(device=device)
        else:
            self.quality_metrics = None
        
        # Initialize missing attributes
        self.gradient_accumulation_steps = 1
        self.global_step = 0
        self.best_metrics = {}
    
    def setup_optimizers(self, encoder, generator, discriminator, phase_detector, lr=1e-4, weight_decay=1e-5):
        """Setup optimizers"""
        
        # Enable training mode for all components
        for model in [encoder, generator, discriminator, phase_detector]:
            model.train()
            for param in model.parameters():
                param.requires_grad = True
        
        print("🧊 DANN training")
        
        # optimize components
        optimizers = {
            'encoder': optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999)),
            'generator': optim.Adam(generator.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999)),
            'discriminator': optim.Adam(discriminator.parameters(), lr=lr*2, weight_decay=1e-5, betas=(0.9, 0.999)),
            'phase_detector': optim.Adam(phase_detector.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999))
        }
        # Learning rate schedulers
        schedulers = {
            'encoder': ReduceLROnPlateau(optimizers['encoder'], mode='min', factor=0.5, patience=10),
            'generator': ReduceLROnPlateau(optimizers['generator'], mode='min', factor=0.5, patience=10),
            'discriminator': CosineAnnealingLR(optimizers['discriminator'], T_max=50),
            'phase_detector': ReduceLROnPlateau(optimizers['phase_detector'], mode='max', factor=0.5, patience=15)
        }
        
        return optimizers, schedulers
    
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
        input_vol = batch["input_path"].to(self.device, non_blocking=True)
        target_vol = batch["target_path"].to(self.device, non_blocking=True)
        input_phase = batch["input_phase"].to(self.device, non_blocking=True)
        target_phase = batch["target_phase"].to(self.device, non_blocking=True)
        
        # DANN lambda scheduling
        lambda_grl = self._get_dann_lambda(epoch, max_epochs)
        
        losses = {}
        batch_size = input_vol.size(0)
        
        # # Clear all gradients
        # for optimizer in optimizers.values():
        #     optimizer.zero_grad()
        
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
            # Phase classification loss
            reversed_features = gradient_reversal(features, lambda_grl)
            phase_pred = phase_detector(reversed_features)
            phase_confusion_loss = self.ce_loss(phase_pred, input_phase)
            
            # Combined generator loss
            gen_total_loss = recon_loss + 0.1 * adv_loss_gen + 0.5 * phase_confusion_loss
        
            # gen_total_loss = recon_loss + 0.1 * adv_loss_gen
        # Backward pass for generator
        
        if self.use_mixed_precision and self.scaler is not None:
            self.scaler.scale(gen_total_loss).backward(retain_graph=True)
        else:
            gen_total_loss.backward()
        # # Backward for generator and encoder (main task)
        # if self.use_mixed_precision:
        #     self.scaler.scale(gen_total_loss).backward()
        #     self.scaler.step(optimizers['generator'])
        #     self.scaler.step(optimizers['encoder'])
        #     self.scaler.update()
        # else:
        #     gen_total_loss.backward()
        #     optimizers['generator'].step()
        #     optimizers['encoder'].step()
        
        # Step 2: Train Discriminator
        # optimizers['discriminator'].zero_grad()
        
        with autocast(enabled=self.use_mixed_precision):
            # Real samples
            real_pred = discriminator(target_vol)
            real_loss = self.bce_loss(real_pred, torch.ones_like(real_pred))
            
           # Fake samples (with fresh forward pass)
            with torch.no_grad():
                features_detached = encoder(input_vol)
                generated_detached = generator(features_detached, phase_emb)
            
            fake_pred = discriminator(generated_detached)
            fake_loss = self.bce_loss(fake_pred, torch.zeros_like(fake_pred))
            
            disc_loss = (real_loss + fake_loss) * 0.5
        
        # Backward pass for discriminator
        if self.use_mixed_precision and self.scaler is not None:
            self.scaler.scale(disc_loss).backward(retain_graph=True)
        else:
            disc_loss.backward()
        
        # # Backward for discriminator
        # if self.use_mixed_precision:
        #     self.scaler.scale(disc_loss).backward()
        #     self.scaler.step(optimizers['discriminator'])
        #     self.scaler.update()
        # else:
        #     disc_loss.backward()
        #     optimizers['discriminator'].step()
        
        # Step 3: DANN Training for Phase Detector
        # optimizers['phase_detector'].zero_grad()
        # === Phase Detector Training ===
        with autocast(enabled=self.use_mixed_precision):
            # Direct phase classification (without gradient reversal)
            features_direct = encoder(input_vol)
            phase_pred_direct = phase_detector(features_direct.detach())
            phase_classification_loss = self.ce_loss(phase_pred_direct, input_phase)
        
        # Backward pass for phase detector
        if self.use_mixed_precision and self.scaler is not None:
            self.scaler.scale(phase_classification_loss).backward(retain_graph=True)
        else:
            phase_classification_loss.backward()
        

        # with autocast(enabled=self.use_mixed_precision):
        #     # Phase classification loss (normal)
        #     phase_pred = phase_detector(features.detach())
        #     phase_classification_loss = self.ce_loss(phase_pred, input_phase)
            
        # # Backward for phase detector
        # if self.use_mixed_precision:
        #     self.scaler.scale(phase_classification_loss).backward()
        #     self.scaler.step(optimizers['phase_detector'])
        #     self.scaler.update()
        # else:
        #     phase_classification_loss.backward()
        #     optimizers['phase_detector'].step()
        
        # # Step 4: Adversarial update for Encoder (using GRL)
        # optimizers['encoder'].zero_grad()
        
        # with autocast(enabled=self.use_mixed_precision):
        #     # Recompute features to build a fresh graph for encoder adversarial update
        #     features = encoder(input_vol)
        #     # Apply gradient reversal
        #     grl_features = gradient_reversal(features, lambda_grl)
        #     confusion_pred = phase_detector(grl_features)
        #     confusion_loss = self.ce_loss(confusion_pred, input_phase)
        
        # # Backward for encoder (with reversed gradients)
        # if self.use_mixed_precision:
        #     self.scaler.scale(confusion_loss).backward()
        #     self.scaler.step(optimizers['encoder'])
        #     self.scaler.update()
        # else:
        #     confusion_loss.backward()
        #     optimizers['encoder'].step()
        
        # Optimizer steps with gradient clipping
        for optimizer_name, optimizer in optimizers.items():
            if self.use_mixed_precision and self.scaler is not None:
                # Mixed precision path - use scaler
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    getattr(models[list(optimizers.keys()).index(optimizer_name)], 'parameters')(), 
                    max_norm=1.0
                )
                self.scaler.step(optimizer)
            else:
                # Non-mixed precision path - no scaler
                torch.nn.utils.clip_grad_norm_(
                    getattr(models[list(optimizers.keys()).index(optimizer_name)], 'parameters')(), 
                    max_norm=1.0
                )
                optimizer.step()
        
            optimizer.zero_grad(set_to_none=True)

        if self.use_mixed_precision and self.scaler is not None:
            self.scaler.update()
        
        # Calculate metrics
        with torch.no_grad():
            phase_acc = (phase_pred_direct.argmax(dim=1) == input_phase).float().mean()
            disc_acc = ((real_pred > 0.5).float().mean() + (fake_pred < 0.5).float().mean()) * 0.5
        
        losses.update({
            'reconstruction': recon_loss.item(),
            'adversarial_gen': adv_loss_gen.item(),
            'discriminator': disc_loss.item(),
            'phase_confusion': phase_confusion_loss.item(),
            'phase_classification': phase_classification_loss.item(),
            'confusion_acc': phase_acc.item(),
            'discriminator_acc': disc_acc.item(),
            'lambda_grl': lambda_grl,
            'generator_total': gen_total_loss.item()
        })
        
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
    def validate_dann_training(self, validation_loader, models):
        """Optimized validation with comprehensive metrics"""
        encoder, generator, discriminator, phase_detector = models
        
        # Set models to eval mode
        for model in models:
            model.eval()
        
        val_losses = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(validation_loader, desc="Validation", leave=False):
                try:
                    input_vol = batch["input_path"].to(self.device, non_blocking=True)
                    target_vol = batch["target_path"].to(self.device, non_blocking=True)
                    input_phase = batch["input_phase"].to(self.device, non_blocking=True)
                    target_phase = batch["target_phase"].to(self.device, non_blocking=True)
                    
                    with autocast(enabled=self.use_mixed_precision):
                        features = encoder(input_vol)
                        phase_emb = self._create_phase_embedding(target_phase, dim=self.phase_dim)
                        generated = generator(features, phase_emb)
                        
                        # Validation metrics
                        recon_loss = self.l1_loss(generated, target_vol)
                        phase_pred = phase_detector(features)
                        phase_acc = (phase_pred.argmax(dim=1) == input_phase).float().mean()
                        
                        # Quality metrics if available
                        if self.quality_metrics:
                            psnr = self.quality_metrics.psnr(generated, target_vol)
                            ssim = self.quality_metrics.ssim(generated, target_vol)
                            val_losses['psnr'].append(psnr)
                            val_losses['ssim'].append(ssim)
                    
                    val_losses['reconstruction_loss'].append(recon_loss.item())
                    val_losses['confusion_accuracy'].append(phase_acc.item())
                    
                except Exception as e:
                    print(f"⚠️ Validation batch error: {e}")
                    continue
        
        # Set models back to train mode
        for model in models:
            model.train()
        
        # Calculate averages
        avg_metrics = {key: np.mean(values) for key, values in val_losses.items()}
        return avg_metrics


    def save_checkpoint(self, models, optimizers, schedulers, epoch, metrics, checkpoint_dir, phase="dann"):
        """Memory-efficient checkpoint saving"""
        encoder, generator, discriminator, phase_detector = models
        
        checkpoint = {
            'epoch': epoch,
            'phase': phase,
            'metrics': metrics,
            'model_states': {
                'encoder': encoder.state_dict(),
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'phase_detector': phase_detector.state_dict()
            },
            'optimizer_states': {name: opt.state_dict() for name, opt in optimizers.items()},
            'scheduler_states': {name: sch.state_dict() for name, sch in schedulers.items()},
            'global_step': self.global_step,
            'best_metrics': self.best_metrics
        }
        # FIXED: Only add scaler state if it exists
        if self.scaler is not None:
            checkpoint['scaler_state'] = self.scaler.state_dict()
        
        # Save with atomic write
        checkpoint_path = os.path.join(checkpoint_dir, f'{phase}_epoch_{epoch}.pth')
        temp_path = checkpoint_path + '.tmp'
        torch.save(checkpoint, temp_path)
        os.rename(temp_path, checkpoint_path)
        
        # Keep only last 3 checkpoints to save space
        self._cleanup_old_checkpoints(checkpoint_dir, phase, keep_last=3)
        
        return checkpoint_path
    
    def _cleanup_old_checkpoints(self, checkpoint_dir, phase, keep_last=3):
        """Clean up old checkpoints to save disk space"""
        try:
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith(f'{phase}_epoch_')]
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            for checkpoint in checkpoints[:-keep_last]:
                os.remove(os.path.join(checkpoint_dir, checkpoint))
        except Exception as e:
            print(f"⚠️ Checkpoint cleanup warning: {e}")

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
                                        phase_dim=8, gradient_accumulation_steps=1):
    
    """
    HIGHLY OPTIMIZED DANN-style training with advanced techniques
    """
    
    print(f"\n🚀 Starting HIGHLY OPTIMIZED DANN Training")
    print(f"📊 Epochs: {num_epochs} | Device: {device}")
    print(f"🔧 Mixed Precision: {use_mixed_precision} | Gradient Accumulation: {gradient_accumulation_steps}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize optimized trainer
    trainer = DANNTrainerOptimized(device, use_mixed_precision, logger, phase_dim)
    trainer.gradient_accumulation_steps = gradient_accumulation_steps
    

    # Model compilation for PyTorch 2.0+ speed boost
    if hasattr(torch, 'compile'):
        print("⚡ Torch compile available but disabled due to dynamic indexing issues")
        print("   To enable, set ENABLE_TORCH_COMPILE=True environment variable")
        
        # Only compile if explicitly enabled
        enable_compile = os.environ.get('ENABLE_TORCH_COMPILE', 'False').lower() == 'true'
        
        if enable_compile:
            print("⚡ Compiling models with torch.compile...")
            try:
                # Use a more permissive compilation mode
                encoder = torch.compile(encoder, mode='default', fullgraph=False)
                generator = torch.compile(generator, mode='default', fullgraph=False)
                discriminator = torch.compile(discriminator, mode='default', fullgraph=False)
                phase_detector = torch.compile(phase_detector, mode='default', fullgraph=False)
                print("✅ Model compilation successful")
            except Exception as e:
                print(f"⚠️ Model compilation failed: {e}, continuing without compilation")
        else:
            print("ℹ️ Model compilation disabled to avoid dynamic indexing issues")
            
    
    models = [encoder, generator, discriminator, phase_detector]
    optimizers, schedulers = trainer.setup_optimizers(*models)
    start_epoch = 0
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
    # Training metrics tracking
    training_history = {
        'reconstruction_losses': [],
        'confusion_accuracies': [],
        'discriminator_losses': [],
        'validation_metrics': []
    }
    
    best_confusion_acc = 0.0
    best_recon_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    total_start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        # Set models to training mode
        for model in models:
            model.train()
        
        epoch_losses = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Training step
                step_losses = trainer.dann_training_step(
                    batch, models, optimizers, epoch, num_epochs
                )
                
                epoch_losses.append(step_losses)
                trainer.global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'recon': f"{step_losses['reconstruction']:.4f}",
                    'conf_acc': f"{step_losses['confusion_acc']:.3f}",
                    'disc_acc': f"{step_losses['discriminator_acc']:.3f}",
                    'λ': f"{step_losses['lambda_grl']:.3f}"
                })
                
                # Memory cleanup every 50 steps
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                print(f"⚠️ Batch {batch_idx} error: {e}")
                continue
        
        # Calculate epoch averages
        if epoch_losses:
            avg_losses = {
                key: np.mean([loss[key] for loss in epoch_losses])
                for key in epoch_losses[0].keys()
            }
            
            # Update learning rate schedulers
            schedulers['encoder'].step(avg_losses['reconstruction'])
            schedulers['generator'].step(avg_losses['reconstruction'])
            schedulers['discriminator'].step()
            schedulers['phase_detector'].step(avg_losses['confusion_acc'])
            
            # Validation
            quality_metrics = None
            if validation_loader and epoch % 5 == 0:
                quality_metrics = trainer.validate_dann_training(validation_loader, models)
                training_history['validation_metrics'].append(quality_metrics)
                
                print(f"   Validation - Recon: {quality_metrics.get('reconstruction_loss', 0):.4f}, "
                     f"Confusion Acc: {quality_metrics.get('confusion_accuracy', 0):.3f}")
            
            # Track best models
            if avg_losses['confusion_acc'] > best_confusion_acc:
                best_confusion_acc = avg_losses['confusion_acc']
                patience_counter = 0
                # Save best model
                trainer.save_checkpoint(
                    models, optimizers, schedulers, epoch, avg_losses, 
                    checkpoint_dir, phase="dann_best"
                )
            else:
                patience_counter += 1
            
            if avg_losses['reconstruction'] < best_recon_loss:
                best_recon_loss = avg_losses['reconstruction']
            
            # Update training history
            training_history['reconstruction_losses'].append(avg_losses['reconstruction'])
            training_history['confusion_accuracies'].append(avg_losses['confusion_acc'])
            training_history['discriminator_losses'].append(avg_losses['discriminator'])
            
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) - "
                  f"Recon: {avg_losses['reconstruction']:.4f}, "
                  f"Conf Acc: {avg_losses['confusion_acc']:.3f}, "
                  f"Disc Loss: {avg_losses['discriminator']:.4f}")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                trainer.save_checkpoint(
                    models, optimizers, schedulers, epoch, avg_losses, 
                    checkpoint_dir, phase="dann"
                )
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"🛑 Early stopping at epoch {epoch+1} (patience={max_patience})")
                break
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print("🎉 OPTIMIZED DANN TRAINING COMPLETED!")
    print(f"⏱️  Total Time: {total_time/3600:.2f} hours")
    print(f"📊 Best Confusion Accuracy: {best_confusion_acc:.4f}")
    print(f"📊 Best Reconstruction Loss: {best_recon_loss:.6f}")
    print(f"{'='*80}")
    
    return training_history