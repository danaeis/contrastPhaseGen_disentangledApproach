import torch
import torch.nn as nn
import numpy as np
import os
from collections import defaultdict
from typing import Dict, Optional, Union, List
import warnings


class AdvancedEarlyStopping:
    """
    Advanced early stopping with oscillation detection and gradient monitoring
    """
    
    def __init__(self, 
                 patience: int = 15,
                 min_delta: float = 1e-4,
                 oscillation_patience: int = 10,
                 oscillation_threshold: float = 0.005,
                 restore_best_weights: bool = True,
                 mode: str = 'min',
                 verbose: bool = True):
        """
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in the monitored quantity to qualify as an improvement
            oscillation_patience: Number of epochs to detect oscillation pattern
            oscillation_threshold: Threshold for detecting oscillation
            restore_best_weights: Whether to restore model weights from the best epoch
            mode: 'min' for loss, 'max' for accuracy/metrics
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.oscillation_patience = oscillation_patience
        self.oscillation_threshold = oscillation_threshold
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.verbose = verbose
        
        # State tracking
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.best_weights = None
        self.score_history = []
        self.stopped = False
        self.stop_reason = ""
        
        # Oscillation detection
        self.oscillation_counter = 0
        
        # Set comparison function based on mode
        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
            self.best_score = float('inf')
        else:
            self.is_better = lambda current, best: current > best + min_delta
            self.best_score = float('-inf')
    
    def __call__(self, 
                 score: float, 
                 model: Optional[nn.Module] = None, 
                 epoch: int = 0,
                 gradient_norm: Optional[float] = None) -> bool:
        """
        Check if training should stop
        
        Args:
            score: Current validation score
            model: Model to save weights from
            epoch: Current epoch
            gradient_norm: Current gradient norm for additional monitoring
            
        Returns:
            True if training should stop
        """
        self.score_history.append(score)
        
        # Check for improvement
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self.oscillation_counter = 0
            
            # Save best weights
            if model is not None and self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            if self.verbose:
                print(f"   📈 New best score: {score:.6f} (epoch {epoch})")
                
        else:
            self.counter += 1
            
            if self.verbose and self.counter % 5 == 0:
                print(f"   ⏰ No improvement for {self.counter} epochs (patience: {self.patience})")
        
        # Check for oscillation pattern
        if len(self.score_history) >= self.oscillation_patience:
            recent_scores = self.score_history[-self.oscillation_patience:]
            score_std = np.std(recent_scores)
            
            if score_std < self.oscillation_threshold:
                self.oscillation_counter += 1
                if self.oscillation_counter >= self.oscillation_patience // 2:
                    self.stopped = True
                    self.stop_reason = f"oscillation_detected_std_{score_std:.6f}"
                    if self.verbose:
                        print(f"   🔄 Oscillation detected - stopping early")
                    return True
            else:
                self.oscillation_counter = 0
        
        # Check gradient norm for vanishing gradients
        if gradient_norm is not None and gradient_norm < 1e-8:
            if self.verbose:
                print(f"   ⚠️  Very small gradient norm detected: {gradient_norm:.2e}")
        
        # Standard patience check
        if self.counter >= self.patience:
            self.stopped = True
            self.stop_reason = f"patience_exceeded_{self.counter}"
            if self.verbose:
                print(f"   🛑 Early stopping triggered - no improvement for {self.counter} epochs")
            return True
        
        return False
    
    def restore_weights(self, model: nn.Module):
        """Restore best weights to model"""
        if self.best_weights is not None:
            device = next(model.parameters()).device
            model.load_state_dict({k: v.to(device) for k, v in self.best_weights.items()})
            if self.verbose:
                print(f"   🔄 Restored weights from epoch {self.best_epoch} (score: {self.best_score:.6f})")
        else:
            if self.verbose:
                print("   ⚠️  No best weights to restore")
    
    def get_summary(self) -> Dict:
        """Get summary of early stopping behavior"""
        return {
            'stopped': self.stopped,
            'stop_reason': self.stop_reason,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'patience_counter': self.counter,
            'total_epochs_monitored': len(self.score_history),
            'oscillation_counter': self.oscillation_counter
        }


class ModelSpecificEarlyStopping:
    """
    Early stopping system that tracks multiple models independently
    """
    
    def __init__(self):
        self.models = {}
        self.global_stopped = False
        
    def add_model(self, 
                  name: str,
                  patience: int = 20,
                  min_delta: float = 1e-5,
                  restore_best_weights: bool = True,
                  mode: str = 'min'):
        """Add a model to track"""
        self.models[name] = AdvancedEarlyStopping(
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=restore_best_weights,
            mode=mode,
            verbose=True
        )
        print(f"   📋 Added {name} to early stopping (patience: {patience})")
    
    def check_model(self, 
                    name: str, 
                    score: float, 
                    model: Optional[nn.Module] = None, 
                    epoch: int = 0,
                    gradient_norm: Optional[float] = None) -> bool:
        """
        Check early stopping for a specific model
        
        Returns:
            True if this specific model should stop
        """
        if name not in self.models:
            warnings.warn(f"Model '{name}' not found in early stopping tracker")
            return False
        
        return self.models[name](score, model, epoch, gradient_norm)
    
    def restore_best_weights(self, name: str, model: nn.Module):
        """Restore best weights for a specific model"""
        if name in self.models:
            self.models[name].restore_weights(model)
        else:
            print(f"   ⚠️  Model '{name}' not found for weight restoration")
    
    def get_summary(self) -> Dict:
        """Get summary for all tracked models"""
        summary = {}
        for name, early_stopping in self.models.items():
            summary[name] = early_stopping.get_summary()
        return summary
    
    def any_stopped(self) -> bool:
        """Check if any model has stopped"""
        return any(es.stopped for es in self.models.values())
    
    def all_stopped(self) -> bool:
        """Check if all models have stopped"""
        return all(es.stopped for es in self.models.values())


def compute_gradient_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    """
    Compute the gradient norm for a model
    
    Args:
        model: PyTorch model
        norm_type: Type of norm to compute (2.0 for L2 norm)
        
    Returns:
        Gradient norm value
    """
    total_norm = 0.0
    param_count = 0
    
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
            param_count += 1
    
    if param_count == 0:
        return 0.0
    
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def compute_parameter_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    """
    Compute the parameter norm for a model
    
    Args:
        model: PyTorch model
        norm_type: Type of norm to compute
        
    Returns:
        Parameter norm value
    """
    total_norm = 0.0
    
    for param in model.parameters():
        param_norm = param.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


class GradientMonitor:
    """
    Monitor gradient flow and detect training issues
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.gradient_history = defaultdict(list)
        self.param_history = defaultdict(list)
        
    def update(self, model: nn.Module, model_name: str):
        """Update gradient and parameter statistics"""
        grad_norm = compute_gradient_norm(model)
        param_norm = compute_parameter_norm(model)
        
        self.gradient_history[model_name].append(grad_norm)
        self.param_history[model_name].append(param_norm)
        
        # Keep only recent history
        if len(self.gradient_history[model_name]) > self.window_size:
            self.gradient_history[model_name] = self.gradient_history[model_name][-self.window_size:]
        if len(self.param_history[model_name]) > self.window_size:
            self.param_history[model_name] = self.param_history[model_name][-self.window_size:]
    
    def get_statistics(self, model_name: str) -> Dict:
        """Get gradient and parameter statistics for a model"""
        if model_name not in self.gradient_history:
            return {}
        
        grad_history = self.gradient_history[model_name]
        param_history = self.param_history[model_name]
        
        return {
            'gradient_norm_mean': np.mean(grad_history),
            'gradient_norm_std': np.std(grad_history),
            'gradient_norm_latest': grad_history[-1] if grad_history else 0,
            'parameter_norm_mean': np.mean(param_history),
            'parameter_norm_std': np.std(param_history),
            'parameter_norm_latest': param_history[-1] if param_history else 0,
            'gradient_stability': np.std(grad_history) / (np.mean(grad_history) + 1e-8)
        }
    
    def detect_issues(self, model_name: str) -> List[str]:
        """Detect potential training issues"""
        issues = []
        stats = self.get_statistics(model_name)
        
        if not stats:
            return issues
        
        # Check for vanishing gradients
        if stats['gradient_norm_latest'] < 1e-7:
            issues.append("vanishing_gradients")
        
        # Check for exploding gradients
        if stats['gradient_norm_latest'] > 100:
            issues.append("exploding_gradients")
        
        # Check for unstable gradients
        if stats['gradient_stability'] > 5.0:
            issues.append("unstable_gradients")
        
        # Check for dead parameters (no gradient change)
        if len(self.gradient_history[model_name]) >= 5:
            recent_grads = self.gradient_history[model_name][-5:]
            if all(abs(g - recent_grads[0]) < 1e-10 for g in recent_grads):
                issues.append("static_gradients")
        
        return issues


def setup_early_stopping_for_training(config: Dict) -> ModelSpecificEarlyStopping:
    """
    Factory function to setup early stopping based on configuration
    
    Args:
        config: Dictionary with early stopping configuration
        
    Returns:
        Configured ModelSpecificEarlyStopping instance
    """
    early_stopping = ModelSpecificEarlyStopping()
    
    # Default configurations for different models
    default_configs = {
        'encoder': {'patience': 25, 'min_delta': 1e-5, 'restore_best_weights': True},
        'generator': {'patience': 20, 'min_delta': 1e-5, 'restore_best_weights': True},
        'discriminator': {'patience': 30, 'min_delta': 1e-5, 'restore_best_weights': False},
        'phase_detector': {'patience': 15, 'min_delta': 1e-4, 'restore_best_weights': True},
        'overall': {'patience': 35, 'min_delta': 1e-6, 'restore_best_weights': False}
    }
    
    # Add models based on config
    for model_name, model_config in config.get('models', default_configs).items():
        early_stopping.add_model(
            name=model_name,
            patience=model_config.get('patience', 20),
            min_delta=model_config.get('min_delta', 1e-5),
            restore_best_weights=model_config.get('restore_best_weights', True),
            mode=model_config.get('mode', 'min')
        )
    
    return early_stopping


class LossTracker:
    """
    Track and analyze loss curves for early stopping decisions
    """
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.train_losses = []
        self.val_losses = []
        
    def update(self, train_loss: float, val_loss: Optional[float] = None):
        """Update loss tracking"""
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        
        # Keep only recent history
        if len(self.train_losses) > self.window_size * 2:
            self.train_losses = self.train_losses[-self.window_size:]
        if len(self.val_losses) > self.window_size * 2:
            self.val_losses = self.val_losses[-self.window_size:]
    
    def detect_overfitting(self, threshold: float = 0.1) -> bool:
        """
        Detect overfitting by comparing train and validation loss trends
        
        Args:
            threshold: Threshold for overfitting detection
            
        Returns:
            True if overfitting is detected
        """
        if len(self.train_losses) < self.window_size or len(self.val_losses) < self.window_size:
            return False
        
        # Compare recent trends
        recent_train = self.train_losses[-self.window_size//2:]
        recent_val = self.val_losses[-self.window_size//2:]
        
        train_trend = np.mean(recent_train) - np.mean(self.train_losses[-self.window_size:-self.window_size//2])
        val_trend = np.mean(recent_val) - np.mean(self.val_losses[-self.window_size:-self.window_size//2])
        
        # Overfitting: train loss decreasing, val loss increasing
        if train_trend < -threshold and val_trend > threshold:
            return True
        
        return False
    
    def get_convergence_status(self) -> Dict:
        """Get convergence status information"""
        if len(self.train_losses) < 10:
            return {'status': 'insufficient_data'}
        
        recent_losses = self.train_losses[-10:]
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        
        # Determine convergence status
        if loss_std / (loss_mean + 1e-8) < 0.01:
            status = 'converged'
        elif loss_std / (loss_mean + 1e-8) > 0.1:
            status = 'unstable'
        else:
            status = 'improving'
        
        return {
            'status': status,
            'loss_std': loss_std,
            'loss_mean': loss_mean,
            'coefficient_of_variation': loss_std / (loss_mean + 1e-8),
            'recent_trend': recent_losses[-1] - recent_losses[0]
        }


# Example usage and testing
if __name__ == "__main__":
    # Test AdvancedEarlyStopping
    print("🧪 Testing AdvancedEarlyStopping...")
    
    early_stopping = AdvancedEarlyStopping(patience=5, min_delta=0.01)
    
    # Simulate training with improving then stable loss
    test_losses = [1.0, 0.8, 0.6, 0.5, 0.49, 0.48, 0.48, 0.48, 0.48, 0.48]
    
    for epoch, loss in enumerate(test_losses):
        should_stop = early_stopping(loss, epoch=epoch)
        print(f"Epoch {epoch}: Loss = {loss}, Stop = {should_stop}")
        if should_stop:
            break
    
    print("Summary:", early_stopping.get_summary())
    
    # Test ModelSpecificEarlyStopping
    print("\n🧪 Testing ModelSpecificEarlyStopping...")
    
    multi_stopping = ModelSpecificEarlyStopping()
    multi_stopping.add_model('generator', patience=3)
    multi_stopping.add_model('discriminator', patience=5)
    
    # Simulate different convergence rates
    gen_losses = [1.0, 0.5, 0.3, 0.29, 0.29, 0.29]
    disc_losses = [1.5, 1.2, 1.0, 0.8, 0.6, 0.4]
    
    for epoch, (g_loss, d_loss) in enumerate(zip(gen_losses, disc_losses)):
        gen_stop = multi_stopping.check_model('generator', g_loss, epoch=epoch)
        disc_stop = multi_stopping.check_model('discriminator', d_loss, epoch=epoch)
        
        print(f"Epoch {epoch}: Gen={g_loss}(stop={gen_stop}), Disc={d_loss}(stop={disc_stop})")
        
        if gen_stop and disc_stop:
            break
    
    print("Multi-model summary:", multi_stopping.get_summary())
    print("✅ Early stopping system tests completed!")