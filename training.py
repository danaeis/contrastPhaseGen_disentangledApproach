import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import os
import glob
import time
import numpy as np
from tqdm import tqdm
import csv
from pathlib import Path

# Import your existing quality metrics
try:
    from image_quality_metrics import ImageQualityMetrics, compute_metrics_for_volume_pair
    HAS_QUALITY_METRICS = True
except ImportError:
    print("⚠️ Image quality metrics not available - using basic validation")
    HAS_QUALITY_METRICS = False


def load_phase_checkpoint(checkpoint_dir, phase_name, models, optimizers=None, device='cuda'):
    """
    Load checkpoint from a specific phase if it exists
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        phase_name: Name of the phase ('phase1', 'phase2', 'phase3')
        models: List or dict of models to load
        optimizers: Optional dict of optimizers to restore
        device: Device to load on
    
    Returns:
        dict: Checkpoint info with loaded status and metrics
    """
    
    # Define checkpoint patterns for this phase
    phase_patterns = [
        f'{phase_name}_complete.pth',
        f'{phase_name}_complete.pth', 
        f'{phase_name}_best.pth',
        f'{phase_name}_final.pth'
    ]
    
    checkpoint_path = None
    print(f"🔍 Searching for {phase_name} checkpoints in {checkpoint_dir}...")
    
    # Find the first available checkpoint for this phase
    for pattern in phase_patterns:
        # search_pattern = os.path.join(checkpoint_dir, pattern)
        # matching_files = glob.glob(search_pattern)
        # print('matchong file', search_pattern,matching_files)
        manual_path = os.path.join(checkpoint_dir, pattern)
        print("manual path", manual_path)
        if os.path.isfile(manual_path):
            checkpoint_path = manual_path
            print(f"📁 Found {phase_name} checkpoint via manual check: {checkpoint_path}")
            break
    
    if not checkpoint_path:
        print(f"⚠️ No {phase_name} checkpoint found - will run this phase")
        return {'loaded': False, 'skip_phase': False, 'metrics': {}}
    
    try:
        print(f"📄 Loading {phase_name} checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle both list and dict of models
        if isinstance(models, list):
            model_names = ['encoder', 'generator', 'discriminator', 'phase_detector']
            model_dict = {name: model for name, model in zip(model_names, models)}
        else:
            model_dict = models
        
        # Load model states
        state_keys = {
            'encoder': 'encoder_state_dict',
            'generator': 'generator_state_dict', 
            'discriminator': 'discriminator_state_dict',
            'phase_detector': 'phase_detector_state_dict'
        }
        
        loaded_models = []
        for model_name, model in model_dict.items():
            state_key = state_keys.get(model_name)
            if state_key and state_key in checkpoint:
                try:
                    model.load_state_dict(checkpoint[state_key])
                    loaded_models.append(model_name)
                    print(f"  ✅ Loaded {model_name} from {phase_name}")
                except Exception as e:
                    print(f"  ⚠️ Failed to load {model_name}: {e}")
        
        # Load optimizer states if provided
        if optimizers and 'optimizers' in checkpoint:
            opt_checkpoint = checkpoint['optimizers']
            for name, optimizer in optimizers.items():
                if name in opt_checkpoint:
                    try:
                        optimizer.load_state_dict(opt_checkpoint[name])
                        print(f"  ✅ Loaded {name} optimizer from {phase_name}")
                    except Exception as e:
                        print(f"  ⚠️ Failed to load {name} optimizer: {e}")
        
        loaded_metrics = {
            'phase_accuracy': checkpoint.get('phase_accuracy', 0),
            'reconstruction_loss': checkpoint.get('reconstruction_loss', float('inf')),
            'confusion_accuracy': checkpoint.get('confusion_accuracy', 0)
        }
        
        print(f"✅ {phase_name} checkpoint loaded successfully!")
        print(f"   Models loaded: {loaded_models}")
        if loaded_metrics['phase_accuracy'] > 0:
            print(f"   Phase accuracy: {loaded_metrics['phase_accuracy']:.4f}")
        if loaded_metrics['reconstruction_loss'] < float('inf'):
            print(f"   Reconstruction loss: {loaded_metrics['reconstruction_loss']:.6f}")
        
        return {
            'loaded': True,
            'skip_phase': True,  # Skip this phase since it's already completed
            'metrics': loaded_metrics,
            'checkpoint_path': checkpoint_path,
            'loaded_models': loaded_models
        }
        
    except Exception as e:
        print(f"❌ Error loading {phase_name} checkpoint: {e}")
        return {'loaded': False, 'skip_phase': False, 'metrics': {}, 'error': str(e)}


def check_training_progress(checkpoint_dir):
    """
    Check which phases have been completed and return training status
    
    Returns:
        dict: Status of each phase and recommended starting point
    """
    phases = ['phase1', 'phase2', 'phase3']
    status = {}
    
    print(f"🔍 Checking training progress in {checkpoint_dir}...")
    
    for phase in phases:
        patterns = [f'{phase}_completed.pth', f'{phase}_complete.pth']
        phase_completed = False
        
        for pattern in patterns:
            if glob.glob(os.path.join(checkpoint_dir, pattern)):
                phase_completed = True
                break
        
        status[phase] = phase_completed
        print(f"  {phase}: {'✅ Completed' if phase_completed else '❌ Not completed'}")
    
    # Determine starting phase
    if status['phase3']:
        start_phase = 'completed'
        print("🎉 All phases completed!")
    elif status['phase2']:
        start_phase = 'phase3'
        print("🚀 Will start from Phase 3 (DANN training)")
    elif status['phase1']:
        start_phase = 'phase2'
        print("🚀 Will start from Phase 2 (Encoder + Generator)")
    else:
        start_phase = 'phase1'
        print("🚀 Will start from Phase 1 (Phase Detector)")
    
    return {
        'phase_status': status,
        'start_phase': start_phase,
        'completed_phases': sum(status.values())
    }


class OptimizedLogger:
    """Simple but comprehensive logger"""
    
    def __init__(self, checkpoint_dir, experiment_name):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = self.checkpoint_dir / "logs"
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        self.csv_file = self.log_dir / f"{experiment_name}_metrics.csv"
        self.log_file = self.log_dir / f"{experiment_name}_training.log"
        
        # Initialize CSV with comprehensive headers
        headers = [
            'epoch', 'phase', 'reconstruction_loss', 'generator_loss', 'discriminator_loss',
            'phase_classification_loss', 'phase_accuracy', 'confusion_accuracy', 'lambda_grl',
            'psnr', 'ssim', 'ms_ssim', 'clinical_status', 'training_time', 'memory_mb'
        ]
        
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        
        print(f"📊 Logger initialized - saving to {self.log_dir}")
    
    def log_epoch(self, epoch, phase, losses, quality_metrics=None, epoch_time=0):
        """Log epoch with all metrics"""
        
        # Get memory usage
        memory_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        
        # Clinical assessment
        clinical_status = "unknown"
        if quality_metrics:
            psnr = quality_metrics.get('psnr', 0)
            ssim = quality_metrics.get('ssim', 0)
            if psnr >= 30 and ssim >= 0.8:
                clinical_status = "excellent"
            elif psnr >= 25 and ssim >= 0.7:
                clinical_status = "good"
            elif psnr >= 20 and ssim >= 0.6:
                clinical_status = "acceptable"
            else:
                clinical_status = "poor"
        
        # Prepare row
        row = [
            epoch, phase,
            losses.get('reconstruction', 0),
            losses.get('generator', 0), 
            losses.get('discriminator', 0),
            losses.get('phase', 0),
            losses.get('phase_acc', 0),
            losses.get('confusion_acc', 0),
            losses.get('lambda_grl', 0),
            quality_metrics.get('psnr', 0) if quality_metrics else 0,
            quality_metrics.get('ssim', 0) if quality_metrics else 0,
            quality_metrics.get('ms_ssim', 0) if quality_metrics else 0,
            clinical_status,
            epoch_time,
            memory_mb
        ]
        
        # Write to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # Write to log file
        log_msg = f"Epoch {epoch+1} ({phase}): "
        log_msg += f"Recon={losses.get('reconstruction', 0):.4f}, "
        if quality_metrics:
            log_msg += f"PSNR={quality_metrics.get('psnr', 0):.1f}dB, "
            log_msg += f"SSIM={quality_metrics.get('ssim', 0):.3f}, "
        log_msg += f"Clinical={clinical_status}, "
        log_msg += f"Time={epoch_time:.2f}s, Mem={memory_mb:.0f}MB\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_msg)
        
        # Print to console
        print(log_msg.strip())


class OptimizedLogger:
    """Simple but comprehensive logger"""
    
    def __init__(self, checkpoint_dir, experiment_name):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = self.checkpoint_dir / "logs"
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        self.csv_file = self.log_dir / f"{experiment_name}_metrics.csv"
        self.log_file = self.log_dir / f"{experiment_name}_training.log"
        
        # Initialize CSV with comprehensive headers
        headers = [
            'epoch', 'phase', 'reconstruction_loss', 'generator_loss', 'discriminator_loss',
            'phase_classification_loss', 'phase_accuracy', 'confusion_accuracy', 'lambda_grl',
            'psnr', 'ssim', 'ms_ssim', 'clinical_status', 'training_time', 'memory_mb'
        ]
        
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        
        print(f"📊 Logger initialized - saving to {self.log_dir}")
    
    def log_epoch(self, epoch, phase, losses, quality_metrics=None, epoch_time=0):
        """Log epoch with all metrics"""
        
        # Get memory usage
        memory_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        
        # Clinical assessment
        clinical_status = "unknown"
        if quality_metrics:
            psnr = quality_metrics.get('psnr', 0)
            ssim = quality_metrics.get('ssim', 0)
            if psnr >= 30 and ssim >= 0.8:
                clinical_status = "excellent"
            elif psnr >= 25 and ssim >= 0.7:
                clinical_status = "good"
            elif psnr >= 20 and ssim >= 0.6:
                clinical_status = "acceptable"
            else:
                clinical_status = "poor"
        
        # Prepare row
        row = [
            epoch, phase,
            losses.get('reconstruction', 0),
            losses.get('generator', 0), 
            losses.get('discriminator', 0),
            losses.get('phase', 0),
            losses.get('phase_acc', 0),
            losses.get('confusion_acc', 0),
            losses.get('lambda_grl', 0),
            quality_metrics.get('psnr', 0) if quality_metrics else 0,
            quality_metrics.get('ssim', 0) if quality_metrics else 0,
            quality_metrics.get('ms_ssim', 0) if quality_metrics else 0,
            clinical_status,
            epoch_time,
            memory_mb
        ]
        
        # Write to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # Write to log file
        log_msg = f"Epoch {epoch+1} ({phase}): "
        log_msg += f"Recon={losses.get('reconstruction', 0):.4f}, "
        if quality_metrics:
            log_msg += f"PSNR={quality_metrics.get('psnr', 0):.1f}dB, "
            log_msg += f"SSIM={quality_metrics.get('ssim', 0):.3f}, "
        log_msg += f"Clinical={clinical_status}, "
        log_msg += f"Time={epoch_time:.2f}s, Mem={memory_mb:.0f}MB\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_msg)
        
        # Print to console
        print(log_msg.strip())


def create_phase_embeddings(phase_labels, dim=32, device='cuda'):
    """Create simple phase embeddings"""
    batch_size = phase_labels.size(0)
    phase_emb = torch.zeros(batch_size, dim, device=device)
    
    for i in range(3):  # 3 phases
        mask = (phase_labels == i)
        if mask.any():
            phase_emb[mask, i*dim//3:(i+1)*dim//3] = 1.0
    
    return phase_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


def validate_with_quality_metrics(val_loader, encoder, generator, phase_detector, device, max_batches=10):
    """Quick validation with image quality metrics"""
    encoder.eval()
    generator.eval()
    phase_detector.eval()
    
    total_recon_loss = 0
    total_phase_acc = 0
    all_quality_metrics = []
    num_batches = 0
    
    if HAS_QUALITY_METRICS:
        quality_eval = ImageQualityMetrics(device=device)
    
    with torch.no_grad():
        for batch in val_loader:
            if num_batches >= max_batches:
                break
                
            try:
                input_vol = batch["input_path"].to(device)
                target_vol = batch["target_path"].to(device)
                target_phase = batch["target_phase"].to(device)
                input_phase = batch["input_phase"].to(device)
                
                # Forward pass
                features = encoder(input_vol)
                phase_emb = create_phase_embeddings(target_phase, dim=32, device=device)
                generated = generator(features, phase_emb)
                phase_pred = phase_detector(features)
                
                # Basic metrics
                recon_loss = nn.L1Loss()(generated, target_vol)
                _, pred = torch.max(phase_pred, 1)
                phase_acc = (pred == input_phase).float().mean()
                
                total_recon_loss += recon_loss.item()
                total_phase_acc += phase_acc.item()
                
                # Quality metrics
                if HAS_QUALITY_METRICS:
                    generated_clamp = torch.clamp(generated, 0, 1)
                    target_clamp = torch.clamp(target_vol, 0, 1)
                    quality_metrics = quality_eval.compute_all_metrics(generated_clamp, target_clamp)
                    all_quality_metrics.append(quality_metrics)
                
                num_batches += 1
                
            except Exception as e:
                print(f"⚠️ Validation batch error: {e}")
                continue
    
    results = {
        'reconstruction_loss': total_recon_loss / max(num_batches, 1),
        'phase_accuracy': total_phase_acc / max(num_batches, 1)
    }
    
    if HAS_QUALITY_METRICS and all_quality_metrics:
        # Average quality metrics
        for metric in ['psnr', 'ssim', 'ms_ssim', 'nmse', 'ncc']:
            values = [m.get(metric, 0) for m in all_quality_metrics]
            results[metric] = np.mean(values) if values else 0
    
    encoder.train()
    generator.train()
    phase_detector.train()
    
    return results


def train_phase_detector_only(train_loader, encoder, phase_detector, num_epochs=30, 
                            device="cuda", logger=None, checkpoint_dir="checkpoints"):
    """Phase 1: Pretrain phase detector with frozen encoder"""
    print(f"\n{'='*60}")
    print("PHASE 1: PRETRAINING PHASE DETECTOR ONLY")
    print(f"{'='*60}")
    # Check if this phase is already completed
    models = {'encoder': encoder, 'phase_detector': phase_detector}
    checkpoint_info = load_phase_checkpoint(checkpoint_dir, 'phase1', models, device=device)
    
    if checkpoint_info['skip_phase']:
        print("✅ Phase 1 already completed - skipping")
        return checkpoint_info['metrics'].get('phase_accuracy', 0.85)
    
    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    
    # Train phase detector
    phase_detector.train()
    optimizer = optim.Adam(phase_detector.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_losses = []
        epoch_accuracies = []
        
        progress_bar = tqdm(train_loader, desc=f"Phase 1 Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                input_vol = batch["input_path"].to(device)
                input_phase = batch["input_phase"].to(device)
                
                optimizer.zero_grad()
                
                # Extract features with frozen encoder
                with torch.no_grad():
                    features = encoder(input_vol)
                
                # Train phase detector
                phase_pred = phase_detector(features)
                loss = nn.CrossEntropyLoss()(phase_pred, input_phase)
                
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                _, pred_labels = torch.max(phase_pred, 1)
                accuracy = (pred_labels == input_phase).float().mean().item()
                
                epoch_losses.append(loss.item())
                epoch_accuracies.append(accuracy)
                
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{accuracy:.3f}"
                })
                
            except Exception as e:
                print(f"⚠️ Phase 1 batch error: {e}")
                continue
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        avg_acc = np.mean(epoch_accuracies) if epoch_accuracies else 0.0
        
        # Update best
        if avg_acc > best_acc:
            best_acc = avg_acc
        
        # Log epoch
        if logger:
            losses = {'phase': avg_loss, 'phase_acc': avg_acc}
            logger.log_epoch(epoch, "phase1_pretrain", losses, epoch_time=epoch_time)
        
        scheduler.step()
        
        # Early stopping
        if avg_acc >= 0.85:
            print(f"🎉 Phase 1 early success! Achieved {avg_acc:.3f} accuracy")
            break
    
    print(f"✅ Phase 1 completed! Best accuracy: {best_acc:.4f}")
    return best_acc


def train_encoder_generator(train_loader, encoder, generator, discriminator, phase_detector, 
                          num_epochs=75, device="cuda", logger=None, validation_loader=None, checkpoint_dir='checkpoints'):
    """Phase 2: Train encoder + generator with frozen phase detector"""
    print(f"\n{'='*60}")
    print("PHASE 2: TRAINING ENCODER + GENERATOR")
    print(f"{'='*60}")
    # Check if this phase is already completed
    models = [encoder, generator, discriminator, phase_detector]
    checkpoint_info = load_phase_checkpoint(checkpoint_dir, 'phase2', models, device=device)
    
    if checkpoint_info['skip_phase']:
        print("✅ Phase 2 already completed - skipping")
        return checkpoint_info['metrics'].get('reconstruction_loss', 0.1)
    # If phase 1 is completed, load it first
    if not checkpoint_info['loaded']:
        phase1_info = load_phase_checkpoint(checkpoint_dir, 'phase1', models, device=device)
        if phase1_info['loaded']:
            print("📄 Loaded Phase 1 results as prerequisite")
    
    # Freeze phase detector
    for param in phase_detector.parameters():
        param.requires_grad = False
    phase_detector.eval()
    
    # Unfreeze encoder
    for param in encoder.parameters():
        param.requires_grad = True
    encoder.train()
    generator.train()
    discriminator.train()
    
    # Optimizers
    optimizer_enc = optim.Adam(encoder.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer_gen = optim.Adam(generator.parameters(), lr=2e-3, weight_decay=1e-5)
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=2e-3, weight_decay=1e-5)
    
    schedulers = [
        optim.lr_scheduler.CosineAnnealingLR(optimizer_enc, T_max=num_epochs),
        optim.lr_scheduler.CosineAnnealingLR(optimizer_gen, T_max=num_epochs),
        optim.lr_scheduler.CosineAnnealingLR(optimizer_disc, T_max=num_epochs)
    ]
    
    best_recon_loss = float('inf')
    l1_loss = nn.L1Loss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_losses = []
        
        progress_bar = tqdm(train_loader, desc=f"Phase 2 Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                input_vol = batch["input_path"].to(device)
                target_vol = batch["target_path"].to(device)
                target_phase = batch["target_phase"].to(device)
                
                # Train Generator + Encoder
                optimizer_enc.zero_grad()
                optimizer_gen.zero_grad()
                
                features = encoder(input_vol)
                phase_emb = create_phase_embeddings(target_phase, dim=32, device=device)
                generated = generator(features, phase_emb)
                
                # Reconstruction loss
                recon_loss = l1_loss(generated, target_vol)
                
                # Adversarial loss
                fake_pred = discriminator(generated)
                real_labels = torch.ones_like(fake_pred)
                adv_loss = bce_loss(fake_pred, real_labels)
                
                gen_loss = recon_loss + 0.1 * adv_loss
                gen_loss.backward()
                
                optimizer_enc.step()
                optimizer_gen.step()
                
                # Train Discriminator
                optimizer_disc.zero_grad()
                
                real_pred = discriminator(target_vol)
                real_labels = torch.ones_like(real_pred)
                real_loss = bce_loss(real_pred, real_labels)
                
                with torch.no_grad():
                    fake_vol = generator(encoder(input_vol), phase_emb)
                
                fake_pred = discriminator(fake_vol.detach())
                fake_labels = torch.zeros_like(fake_pred)
                fake_loss = bce_loss(fake_pred, fake_labels)
                
                disc_loss = 0.5 * (real_loss + fake_loss)
                disc_loss.backward()
                optimizer_disc.step()
                
                # Store losses
                step_losses = {
                    'reconstruction': recon_loss.item(),
                    'generator': gen_loss.item(),
                    'discriminator': disc_loss.item()
                }
                epoch_losses.append(step_losses)
                
                progress_bar.set_postfix({
                    'recon': f"{recon_loss.item():.4f}",
                    'gen': f"{gen_loss.item():.4f}",
                    'disc': f"{disc_loss.item():.4f}"
                })
                
            except Exception as e:
                print(f"⚠️ Phase 2 batch error: {e}")
                continue
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        if epoch_losses:
            avg_losses = {
                key: np.mean([loss[key] for loss in epoch_losses])
                for key in epoch_losses[0].keys()
            }
            
            # Validation with quality metrics
            quality_metrics = None
            if validation_loader and epoch % 15 == 0:
                quality_metrics = validate_with_quality_metrics(
                    validation_loader, encoder, generator, phase_detector, device
                )
                print(f"   Validation - PSNR: {quality_metrics.get('psnr', 0):.2f}dB, "
                     f"SSIM: {quality_metrics.get('ssim', 0):.4f}")
            
            # Update best
            if avg_losses['reconstruction'] < best_recon_loss:
                best_recon_loss = avg_losses['reconstruction']
            
            # Log epoch
            if logger:
                logger.log_epoch(epoch, "phase2_encoder_gen", avg_losses, 
                               quality_metrics, epoch_time)
            
            # Early stopping
            if avg_losses['reconstruction'] < 0.1:
                print(f"🎉 Phase 2 early success! Achieved {avg_losses['reconstruction']:.4f} loss")
                break
        
        # Update schedulers
        for scheduler in schedulers:
            scheduler.step()
    
    print(f"✅ Phase 2 completed! Best loss: {best_recon_loss:.6f}")
    return best_recon_loss

def train_contrast_phase_generation_optimized(train_loader, encoder, generator, discriminator, 
                                            phase_detector, num_epochs=225, device="cuda",
                                            checkpoint_dir="checkpoints", use_mixed_precision=True,
                                            validation_loader=None, encoder_config=None, 
                                            encoder_type="simple_cnn",
                                            load_previous_phase=True,
                                            force_restart=False):
    """
    OPTIMIZED Sequential training:
    Phase 1: Pretrain phase detector (30 epochs)
    Phase 2: Train encoder + generator (75 epochs)  
    Phase 3: DANN training (120 epochs)
    """
    
    print(f"🚀 Starting OPTIMIZED Sequential Training ({num_epochs} total epochs)")
    print(f"📊 Strategy: Phase Detector → Encoder/Generator → DANN")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize logger
    logger = OptimizedLogger(checkpoint_dir, f"sequential_training")
    
    total_start_time = time.time()
    
    # Phase 1: Pretrain Phase Detector (1/5 num epochs)
    phase_1_epochs = num_epochs//5
    best_phase_acc = train_phase_detector_only(
        train_loader, encoder, phase_detector, phase_1_epochs, device, logger, checkpoint_dir
    )
    
    # Save Phase 1 checkpoint
    torch.save({
        'phase': 'phase1_complete',
        'encoder_state_dict': encoder.state_dict(),
        'phase_detector_state_dict': phase_detector.state_dict(),
        'phase_accuracy': best_phase_acc,
        'encoder_config': encoder_config
    }, os.path.join(checkpoint_dir, 'phase1_complete.pth'))
    
    # # Phase 2: Train Encoder + Generator (75 epochs)  
    # phase_2_epochs = 75
    # best_recon_loss = train_encoder_generator(
    #     train_loader, encoder, generator, discriminator, phase_detector,
    #     phase_2_epochs, device, logger, validation_loader
    # )
    
    # # Save Phase 2 checkpoint
    # torch.save({
    #     'phase': 'phase2_complete',
    #     'encoder_state_dict': encoder.state_dict(),
    #     'generator_state_dict': generator.state_dict(),
    #     'discriminator_state_dict': discriminator.state_dict(),
    #     'phase_detector_state_dict': phase_detector.state_dict(),
    #     'reconstruction_loss': best_recon_loss,
    #     'encoder_config': encoder_config
    # }, os.path.join(checkpoint_dir, 'phase2_complete.pth'))
    
    # Phase 3: DANN Training (120 epochs) - Use existing optimized DANN
    print(f"\n{'='*60}")
    print("PHASE 3: DANN TRAINING")
    print(f"{'='*60}")
    
    # Import and use your optimized DANN training
    from training_dann_style import train_dann_style_contrast_generation
    dann_epochs = num_epochs - phase_1_epochs
    dann_results = train_dann_style_contrast_generation(
        train_loader, encoder, generator, discriminator, phase_detector,
        num_epochs=dann_epochs, device=device, checkpoint_dir=checkpoint_dir,
        use_mixed_precision=use_mixed_precision, validation_loader=validation_loader,
        encoder_config=encoder_config, encoder_type=encoder_type,
        use_sequential_approach=True, logger=logger
    )
    
    total_time = time.time() - total_start_time
    
    # Final summary
    print(f"\n{'='*100}")
    print("🎉 OPTIMIZED SEQUENTIAL TRAINING COMPLETED!")
    print(f"{'='*100}")
    print(f"⏱️  Total Time: {total_time:.2f}s ({total_time/3600:.2f}h)")
    print(f"🎯 Phase 1 - Best Phase Accuracy: {best_phase_acc:.4f}")
    # print(f"🎯 Phase 2 - Best Reconstruction Loss: {best_recon_loss:.6f}")
    
    # Save final checkpoint
    torch.save({
        'training_completed': True,
        'total_time': total_time,
        'encoder_state_dict': encoder.state_dict(),
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'phase_detector_state_dict': phase_detector.state_dict(),
        'best_metrics': {
            'phase_accuracy': best_phase_acc,
            # 'reconstruction_loss': best_recon_loss
        },
        'encoder_config': encoder_config,
        'encoder_type': encoder_type,
        'training_type': 'optimized_sequential'
    }, os.path.join(checkpoint_dir, "final_optimized_checkpoint.pth"))
    
    print(f"📁 All results saved to: {checkpoint_dir}")
    print(f"📊 Detailed logs: {checkpoint_dir}/logs/")
    
    return {
        'phase_1_accuracy': best_phase_acc,
        # 'phase_2_loss': best_recon_loss,
        'total_time': total_time
    }


# Keep your existing functions but add the optimized sequential training as default
def train_contrast_phase_generation(train_loader, encoder, generator, discriminator, 
                                   phase_detector, num_epochs=225, device="cuda",
                                   checkpoint_dir="checkpoints", use_mixed_precision=True,
                                   validation_loader=None, encoder_config=None, 
                                   encoder_type="simple_cnn"):
    """Main training function - now uses optimized sequential approach"""
    
    return train_contrast_phase_generation_optimized(
        train_loader, encoder, generator, discriminator, phase_detector,
        num_epochs, device, checkpoint_dir, use_mixed_precision,
        validation_loader, encoder_config, encoder_type
    )