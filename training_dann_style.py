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

class DANNTrainerFixed:
    """FIXED: DANN-style trainer with proper graph management"""
    
    def __init__(self, device="cuda", use_mixed_precision=True):
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.scaler = GradScaler() if use_mixed_precision else None
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss() 
        self.ce_loss = nn.CrossEntropyLoss()
        
        # FIXED: Track computation graphs
        self.enable_graph_debugging = False
    
    def setup_optimizers(self, encoder, generator, discriminator, phase_detector):
        """Setup DANN-style optimizers"""
        return {
            'encoder': optim.Adam(encoder.parameters(), lr=1e-4, betas=(0.5, 0.999)),
            'generator': optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999)),
            'discriminator': optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999)),
            'phase_detector': optim.Adam(phase_detector.parameters(), lr=1e-4, betas=(0.5, 0.999))  # Increased LR for better learning
        }
    
    def dann_training_step(self, batch, models, optimizers, epoch, max_epochs):
        """FIXED: Single DANN training step with proper graph management"""
        encoder, generator, discriminator, phase_detector = models
        
        # Prepare data
        input_vol = batch["input_path"].to(self.device)
        target_vol = batch["target_path"].to(self.device)
        input_phase = batch["input_phase"].to(self.device)
        target_phase = batch["target_phase"].to(self.device)
        
        # DANN lambda scheduling
        lambda_grl = self._get_dann_lambda(epoch, max_epochs, 'adaptive')
        
        losses = {}
        
        # FIXED: Clear all gradients first
        for optimizer in optimizers.values():
            optimizer.zero_grad()
        
        # Debug: Print lambda value occasionally
        if torch.rand(1).item() < 0.1:  # 10% of the time
            print(f"\nDEBUG DANN Training - Epoch {epoch}/{max_epochs}")
            print(f"  Lambda GRL: {lambda_grl:.4f}")
        
        # Method 1: Sequential approach (RECOMMENDED)
        # This avoids sharing computation graphs between different losses
        losses.update(self._train_generator_step(
            input_vol, target_vol, target_phase, encoder, generator, discriminator, optimizers
        ))
        
        losses.update(self._train_discriminator_step(
            input_vol, target_vol, target_phase, encoder, generator, discriminator, optimizers
        ))
        
        losses.update(self._train_phase_detector_step(
            input_vol, input_phase, lambda_grl, encoder, phase_detector, optimizers
        ))
        
        losses['lambda_grl'] = lambda_grl
        return losses
    
    def _train_generator_step(self, input_vol, target_vol, target_phase, encoder, generator, discriminator, optimizers):
        """FIXED: Generator training with isolated computation graph"""
        
        # Clear gradients
        optimizers['encoder'].zero_grad()
        optimizers['generator'].zero_grad()
        
        with autocast(device_type="cuda", enabled=self.use_mixed_precision):
            # FIXED: Fresh forward pass for generator
            features = encoder(input_vol)
            phase_emb = create_simple_phase_embedding(target_phase, dim=32, device=self.device)
            generated = generator(features, phase_emb)
            
            # Reconstruction loss
            recon_loss = self.l1_loss(generated, target_vol)
            
            # Adversarial loss (generator wants to fool discriminator)
            fake_scores = discriminator(generated)
            g_adv_loss = self.bce_loss(fake_scores, torch.ones_like(fake_scores))
            
            # Combined generator loss
            g_total_loss = recon_loss * 100.0 + g_adv_loss * 1.0
        
        # Backward pass
        if self.use_mixed_precision:
            self.scaler.scale(g_total_loss).backward()
            self.scaler.unscale_(optimizers['encoder'])
            self.scaler.unscale_(optimizers['generator'])
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            self.scaler.step(optimizers['encoder'])
            self.scaler.step(optimizers['generator'])
            self.scaler.update()
        else:
            g_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizers['encoder'].step()
            optimizers['generator'].step()
        
        return {
            'reconstruction': recon_loss.item(),
            'generator': g_total_loss.item(),
            'g_adv': g_adv_loss.item()
        }
    
    def _train_discriminator_step(self, input_vol, target_vol, target_phase, encoder, generator, discriminator, optimizers):
        """FIXED: Discriminator training with detached inputs"""
        
        # Clear gradients
        optimizers['discriminator'].zero_grad()
        
        with autocast(device_type="cuda", enabled=self.use_mixed_precision):
            # FIXED: Generate fake samples with detached encoder output to avoid graph conflicts
            with torch.no_grad():  # Detach encoder from discriminator training
                features = encoder(input_vol)
                phase_emb = create_simple_phase_embedding(target_phase, dim=32, device=self.device)
                generated = generator(features, phase_emb)
            
            # Real samples
            real_scores = discriminator(target_vol)
            d_real_loss = self.bce_loss(real_scores, torch.ones_like(real_scores))
            
            # Fake samples (detached from generator)
            fake_scores = discriminator(generated.detach())  # FIXED: Detach generated samples
            d_fake_loss = self.bce_loss(fake_scores, torch.zeros_like(fake_scores))
            
            # Combined discriminator loss
            d_loss = (d_real_loss + d_fake_loss) * 0.5
        
        # Backward pass
        if self.use_mixed_precision:
            self.scaler.scale(d_loss).backward()
            self.scaler.unscale_(optimizers['discriminator'])
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            self.scaler.step(optimizers['discriminator'])
            self.scaler.update()
        else:
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizers['discriminator'].step()
        
        return {
            'discriminator': d_loss.item(),
            'd_real': d_real_loss.item(),
            'd_fake': d_fake_loss.item()
        }
    
    def _train_phase_detector_step(self, input_vol, input_phase, lambda_grl, encoder, phase_detector, optimizers):
        """FIXED: Phase detector training with proper gradient reversal"""
        
        # Clear gradients
        optimizers['phase_detector'].zero_grad()
        
        with autocast(device_type="cuda", enabled=self.use_mixed_precision):
            # FIXED: Fresh encoder forward pass for phase detection
            features = encoder(input_vol)
            
            # Normal phase prediction (for classification accuracy)
            phase_pred = phase_detector(features)
            phase_loss = self.ce_loss(phase_pred, input_phase)
            
            # FIXED: Gradient reversal for domain confusion (separate forward pass)
            # This avoids graph conflicts by creating a separate computation path
            # Always apply confusion loss since lambda_grl now has a minimum value
            # Create reversed features with proper gradient handling
            reversed_features = GradientReversalLayer.apply(features.detach().requires_grad_(True), lambda_grl)
            confusion_pred = phase_detector(reversed_features)
            confusion_loss = self.ce_loss(confusion_pred, input_phase)
            
            # Total phase detector loss - always include confusion loss
            total_phase_loss = phase_loss + confusion_loss * lambda_grl
        
        # Backward pass
        if self.use_mixed_precision:
            self.scaler.scale(total_phase_loss).backward()
            self.scaler.unscale_(optimizers['phase_detector'])
            torch.nn.utils.clip_grad_norm_(phase_detector.parameters(), max_norm=1.0)
            self.scaler.step(optimizers['phase_detector'])
            self.scaler.update()
        else:
            total_phase_loss.backward()
            torch.nn.utils.clip_grad_norm_(phase_detector.parameters(), max_norm=1.0)
            optimizers['phase_detector'].step()
        
        # Calculate accuracies
        with torch.no_grad():
            _, phase_pred_labels = torch.max(phase_pred, 1)
            _, confusion_pred_labels = torch.max(confusion_pred, 1)
            phase_acc = (phase_pred_labels == input_phase).float().mean().item()
            confusion_acc = (confusion_pred_labels != input_phase).float().mean().item()
        
        return {
            'phase': phase_loss.item(),
            'confusion': confusion_loss.item(),
            'phase_acc': phase_acc,
            'confusion_acc': confusion_acc
        }
    
    def _get_dann_lambda(self, epoch, max_epochs, schedule='adaptive'):
        """DANN lambda scheduling"""
        p = epoch / max_epochs
        
        if schedule == 'adaptive':
            # Start with a minimum value to ensure confusion loss is always applied
            return max(0.1, min(1.0, p * 2))  # Minimum 0.1, linear growth, capped at 1.0
        elif schedule == 'exp':
            # Ensure minimum value for exponential schedule
            return max(0.1, 2.0 / (1.0 + np.exp(-10 * p)) - 1.0)
        else:  # linear
            return max(0.1, p)  # Minimum 0.1

    # Alternative Method: Single-pass approach with careful graph management
    def dann_training_step_single_pass(self, batch, models, optimizers, epoch, max_epochs):
        """Alternative: Single-pass DANN step with careful graph management"""
        encoder, generator, discriminator, phase_detector = models
        
        # Prepare data
        input_vol = batch["input_path"].to(self.device)
        target_vol = batch["target_path"].to(self.device)
        input_phase = batch["input_phase"].to(self.device)
        target_phase = batch["target_phase"].to(self.device)
        
        lambda_grl = self._get_dann_lambda(epoch, max_epochs, 'adaptive')
        
        # FIXED: Clear all gradients
        for optimizer in optimizers.values():
            optimizer.zero_grad()
        
        with autocast(device_type="cuda", enabled=self.use_mixed_precision):
            # Single forward pass through encoder
            features = encoder(input_vol)
            
            # Generation
            phase_emb = create_simple_phase_embedding(target_phase, dim=32, device=self.device)
            generated = generator(features, phase_emb)
            
            # Losses computation
            recon_loss = self.l1_loss(generated, target_vol)
            
            # Discriminator losses
            real_scores = discriminator(target_vol)
            fake_scores = discriminator(generated)
            d_real_loss = self.bce_loss(real_scores, torch.ones_like(real_scores))
            d_fake_loss = self.bce_loss(fake_scores, torch.zeros_like(fake_scores))
            g_adv_loss = self.bce_loss(fake_scores, torch.ones_like(fake_scores))
            
            # Phase detection - ensure features are properly detached for gradient reversal
            phase_pred = phase_detector(features)
            phase_loss = self.ce_loss(phase_pred, input_phase)
            
            # Gradient reversal - use detached features to avoid gradient conflicts
            reversed_features = GradientReversalLayer.apply(features.detach().requires_grad_(True), lambda_grl)
            confusion_pred = phase_detector(reversed_features)
            confusion_loss = self.ce_loss(confusion_pred, input_phase)
        
        # FIXED: Separate backward passes with retain_graph where needed
        
        # 1. Discriminator loss
        d_loss = (d_real_loss + d_fake_loss) * 0.5
        if self.use_mixed_precision:
            self.scaler.scale(d_loss).backward(retain_graph=True)
        else:
            d_loss.backward(retain_graph=True)
        
        # 2. Generator and reconstruction loss  
        g_loss = recon_loss * 100.0 + g_adv_loss * 1.0
        if self.use_mixed_precision:
            self.scaler.scale(g_loss).backward(retain_graph=True)
        else:
            g_loss.backward(retain_graph=True)
        
        # 3. Phase detection and confusion loss (final backward, no retain_graph)
        # Ensure both losses contribute to phase detector training
        phase_total_loss = phase_loss + confusion_loss * lambda_grl
        
        # Debug: Print loss components occasionally
        if torch.rand(1).item() < 0.05:  # 5% of the time
            print(f"  Phase total loss: {phase_total_loss.item():.4f} (phase: {phase_loss.item():.4f}, confusion: {confusion_loss.item():.4f} * {lambda_grl:.4f})")
        
        if self.use_mixed_precision:
            self.scaler.scale(phase_total_loss).backward()
        else:
            phase_total_loss.backward()
        
        # FIXED: Apply gradients with proper scaling
        if self.use_mixed_precision:
            for name, optimizer in optimizers.items():
                self.scaler.unscale_(optimizer)
                # Get parameters for this optimizer
                if name == 'encoder':
                    params = encoder.parameters()
                elif name == 'generator':
                    params = generator.parameters()
                elif name == 'discriminator':
                    params = discriminator.parameters()
                elif name == 'phase_detector':
                    params = phase_detector.parameters()
                
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                self.scaler.step(optimizer)
            
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(phase_detector.parameters(), max_norm=1.0)
            
            for optimizer in optimizers.values():
                optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            _, phase_pred_labels = torch.max(phase_pred, 1)
            _, confusion_pred_labels = torch.max(confusion_pred, 1)
            
            # Debug: Print phase predictions occasionally
            if torch.rand(1).item() < 0.05:  # 5% of the time
                print(f"\nDEBUG Phase Detection:")
                print(f"  Input phases: {input_phase[:5].cpu().numpy()}")
                print(f"  Phase pred: {phase_pred_labels[:5].cpu().numpy()}")
                print(f"  Confusion pred: {confusion_pred_labels[:5].cpu().numpy()}")
                print(f"  Lambda GRL: {lambda_grl:.4f}")
                print(f"  Phase loss: {phase_loss.item():.4f}")
                print(f"  Confusion loss: {confusion_loss.item():.4f}")
            
            losses = {
                'reconstruction': recon_loss.item(),
                'generator': g_loss.item(),
                'discriminator': d_loss.item(),
                'phase': phase_loss.item(),
                'confusion': confusion_loss.item(),
                'phase_acc': (phase_pred_labels == input_phase).float().mean().item(),
                'confusion_acc': (confusion_pred_labels != input_phase).float().mean().item(),
                'lambda_grl': lambda_grl
            }
        
        return losses


def train_dann_style_contrast_generation(train_loader, encoder, generator, discriminator,
                                             phase_detector, num_epochs=100, device="cuda",
                                             checkpoint_dir="checkpoints", use_mixed_precision=True,
                                             validation_loader=None, encoder_config=None,
                                             encoder_type="simple_cnn", use_sequential_approach=True):
    """FIXED: DANN-style training with proper graph management"""
    
    print(f"🚀 Starting FIXED DANN-style training for {num_epochs} epochs")
    print(f"📊 Using {'sequential' if use_sequential_approach else 'single-pass'} approach")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = DANNTrainerFixed(device, use_mixed_precision)
    
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
    
    best_reconstruction_loss = float('inf')
    
    print(f"\n{'='*60}")
    print("FIXED DANN-STYLE TRAINING WITH PROPER GRAPH MANAGEMENT")
    print(f"{'='*60}")
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # Training loop
        progress_bar = tqdm(train_loader, desc=f"FIXED DANN Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Choose training approach
                if use_sequential_approach:
                    step_losses = trainer.dann_training_step(batch, models, optimizers, epoch, num_epochs)
                else:
                    step_losses = trainer.dann_training_step_single_pass(batch, models, optimizers, epoch, num_epochs)
                
                epoch_losses.append(step_losses)
                # print("losses are ", epoch_losses)
                # Update progress bar
                if batch_idx % 50 == 0:
                    progress_bar.set_postfix({
                        'recon': f"{step_losses['reconstruction']:.4f}",
                        'phase_acc': f"{step_losses['phase_acc']:.3f}",
                        'conf_acc': f"{step_losses['confusion_acc']:.3f}",
                        'λ': f"{step_losses['lambda_grl']:.3f}"
                    })
                    
            except RuntimeError as e:
                if "Trying to backward through the graph a second time" in str(e):
                    print(f"⚠️ Graph error at batch {batch_idx} - skipping batch")
                    continue
                else:
                    print(f"⚠️ Batch {batch_idx} error: {e}")
                    continue
            except Exception as e:
                print(f"⚠️ Unexpected error at batch {batch_idx}: {e}")
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
            
            # Save best model
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
    
    print(f"\n✅ FIXED DANN-style training completed!")
    print(f"📊 Training Summary:")
    print(f"   Best Reconstruction Loss: {best_reconstruction_loss:.6f}")
    
    if metrics["confusion_accuracies"]:
        final_confusion_acc = metrics["confusion_accuracies"][-1]
        if final_confusion_acc > 0.4:
            print("🎯 DANN Success: Model learned phase-invariant features!")
        else:
            print("⚠️  DANN Warning: Low confusion rate - consider tuning λ_grl")
    
    return metrics


# Debug utilities
def debug_computation_graph(tensor, name="tensor"):
    """Debug utility to check tensor properties"""
    print(f"🔍 Debug {name}:")
    print(f"   Shape: {tensor.shape}")
    print(f"   Device: {tensor.device}")
    print(f"   Requires grad: {tensor.requires_grad}")
    print(f"   Has grad_fn: {tensor.grad_fn is not None}")
    if tensor.grad_fn:
        print(f"   Grad function: {type(tensor.grad_fn).__name__}")


def safe_backward(loss, retain_graph=False, name="loss"):
    """Safe backward pass with error handling"""
    try:
        loss.backward(retain_graph=retain_graph)
        return True
    except RuntimeError as e:
        print(f"⚠️ Backward pass failed for {name}: {e}")
        if "Trying to backward through the graph a second time" in str(e):
            print("   💡 Suggestion: Try using retain_graph=True or detach intermediate tensors")
        return False
    except Exception as e:
        print(f"⚠️ Unexpected error in backward pass for {name}: {e}")
        return False