import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import csv
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import warnings
from collections import defaultdict
import json


class ImageQualityMetrics:
    """
    Comprehensive image quality metrics for 3D medical volumes
    """
    
    def __init__(self, data_range: float = 1.0, device: str = 'cuda'):
        """
        Args:
            data_range: Range of the image data (e.g., 1.0 for [0,1] normalized images)
            device: Device to run computations on
        """
        self.data_range = data_range
        self.device = device
    
    def psnr(self, img1: torch.Tensor, img2: torch.Tensor, data_range: Optional[float] = None) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR)
        
        Args:
            img1: First image tensor
            img2: Second image tensor
            data_range: Range of the data (if None, uses self.data_range)
            
        Returns:
            PSNR value in dB
        """
        if data_range is None:
            data_range = self.data_range
            
        mse = torch.mean((img1 - img2) ** 2)
        
        if mse == 0:
            return float('inf')
        
        psnr_value = 20 * torch.log10(data_range / torch.sqrt(mse))
        return psnr_value.item()
    
    def ssim(self, img1: torch.Tensor, img2: torch.Tensor, 
             window_size: int = 11, sigma: float = 1.5,
             data_range: Optional[float] = None) -> float:
        """
        Calculate Structural Similarity Index (SSIM) for 3D volumes
        
        Args:
            img1: First image tensor (B, C, D, H, W)
            img2: Second image tensor (B, C, D, H, W)
            window_size: Size of the sliding window
            sigma: Standard deviation of the Gaussian window
            data_range: Range of the data
            
        Returns:
            SSIM value
        """
        if data_range is None:
            data_range = self.data_range
            
        # Constants for stability
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        # Create 3D Gaussian window
        window = self._create_3d_window(window_size, sigma, img1.device)
        
        # Convert to same dtype
        img1 = img1.to(dtype=torch.float32)
        img2 = img2.to(dtype=torch.float32)
        
        # Calculate local means
        mu1 = F.conv3d(img1, window, padding=window_size//2, groups=img1.shape[1])
        mu2 = F.conv3d(img2, window, padding=window_size//2, groups=img2.shape[1])
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Calculate local variances and covariance
        sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size//2, groups=img1.shape[1]) - mu1_sq
        sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size//2, groups=img2.shape[1]) - mu2_sq
        sigma12 = F.conv3d(img1 * img2, window, padding=window_size//2, groups=img1.shape[1]) - mu1_mu2
        
        # Calculate SSIM
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim_map = numerator / denominator
        return ssim_map.mean().item()
    
    def ms_ssim(self, img1: torch.Tensor, img2: torch.Tensor,
                levels: int = 5, weights: Optional[List[float]] = None,
                data_range: Optional[float] = None) -> float:
        """
        Calculate Multi-Scale SSIM (MS-SSIM)
        
        Args:
            img1: First image tensor
            img2: Second image tensor
            levels: Number of scales
            weights: Weights for each scale
            data_range: Range of the data
            
        Returns:
            MS-SSIM value
        """
        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]  # Standard weights
        
        if len(weights) != levels:
            weights = [1.0 / levels] * levels
        
        if data_range is None:
            data_range = self.data_range
        
        mcs = []
        
        for i in range(levels):
            if i > 0:
                # Downsample by factor of 2
                img1 = F.avg_pool3d(img1, kernel_size=2, stride=2)
                img2 = F.avg_pool3d(img2, kernel_size=2, stride=2)
            
            ssim_val = self.ssim(img1, img2, data_range=data_range)
            mcs.append(ssim_val)
        
        # Weighted average
        ms_ssim_val = sum(w * mc for w, mc in zip(weights, mcs))
        return ms_ssim_val
    
    def nmse(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Calculate Normalized Mean Squared Error (NMSE)
        
        Args:
            img1: Reference image
            img2: Compared image
            
        Returns:
            NMSE value
        """
        mse = torch.mean((img1 - img2) ** 2)
        norm_factor = torch.mean(img1 ** 2)
        
        if norm_factor == 0:
            return float('inf') if mse > 0 else 0.0
        
        nmse_val = mse / norm_factor
        return nmse_val.item()
    
    def ncc(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Calculate Normalized Cross Correlation (NCC)
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            NCC value
        """
        # Flatten images
        img1_flat = img1.view(-1)
        img2_flat = img2.view(-1)
        
        # Center the data
        img1_centered = img1_flat - torch.mean(img1_flat)
        img2_centered = img2_flat - torch.mean(img2_flat)
        
        # Calculate correlation
        numerator = torch.sum(img1_centered * img2_centered)
        denominator = torch.sqrt(torch.sum(img1_centered ** 2) * torch.sum(img2_centered ** 2))
        
        if denominator == 0:
            return 0.0
        
        ncc_val = numerator / denominator
        return ncc_val.item()
    
    def mutual_information(self, img1: torch.Tensor, img2: torch.Tensor, bins: int = 256) -> float:
        """
        Calculate Mutual Information between two images
        
        Args:
            img1: First image
            img2: Second image
            bins: Number of histogram bins
            
        Returns:
            Mutual information value
        """
        # Convert to numpy for histogram computation
        img1_np = img1.cpu().numpy().flatten()
        img2_np = img2.cpu().numpy().flatten()
        
        # Create joint histogram
        hist_2d, _, _ = np.histogram2d(img1_np, img2_np, bins=bins)
        
        # Normalize to get probabilities
        pxy = hist_2d / hist_2d.sum()
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        
        # Calculate mutual information
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
        
        return mi
    
    def lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Calculate LPIPS (Learned Perceptual Image Patch Similarity)
        Note: This is a simplified version for 3D medical images
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            LPIPS-like score
        """
        # Use L2 distance in feature space as a proxy
        # This could be replaced with actual LPIPS if needed
        diff = img1 - img2
        lpips_score = torch.mean(torch.sqrt(torch.sum(diff ** 2, dim=1)))
        return lpips_score.item()
    
    def _create_3d_window(self, window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
        """Create 3D Gaussian window for SSIM calculation"""
        coords = torch.arange(window_size, dtype=torch.float32, device=device)
        coords -= window_size // 2
        
        g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g1d /= g1d.sum()
        
        # Create 3D window
        g3d = g1d[:, None, None] * g1d[None, :, None] * g1d[None, None, :]
        g3d = g3d.unsqueeze(0).unsqueeze(0)
        
        return g3d
    
    def compute_all_metrics(self, img1: torch.Tensor, img2: torch.Tensor) -> Dict[str, float]:
        """
        Compute all available metrics
        
        Args:
            img1: Reference image
            img2: Compared image
            
        Returns:
            Dictionary with all computed metrics
        """
        metrics = {}
        
        try:
            metrics['psnr'] = self.psnr(img1, img2)
        except Exception as e:
            metrics['psnr'] = 0.0
            warnings.warn(f"PSNR calculation failed: {e}")
        
        try:
            metrics['ssim'] = self.ssim(img1, img2)
        except Exception as e:
            metrics['ssim'] = 0.0
            warnings.warn(f"SSIM calculation failed: {e}")
        
        try:
            metrics['ms_ssim'] = self.ms_ssim(img1, img2)
        except Exception as e:
            metrics['ms_ssim'] = 0.0
            warnings.warn(f"MS-SSIM calculation failed: {e}")
        
        try:
            metrics['nmse'] = self.nmse(img1, img2)
        except Exception as e:
            metrics['nmse'] = float('inf')
            warnings.warn(f"NMSE calculation failed: {e}")
        
        try:
            metrics['ncc'] = self.ncc(img1, img2)
        except Exception as e:
            metrics['ncc'] = 0.0
            warnings.warn(f"NCC calculation failed: {e}")
        
        try:
            metrics['mi'] = self.mutual_information(img1, img2)
        except Exception as e:
            metrics['mi'] = 0.0
            warnings.warn(f"Mutual Information calculation failed: {e}")
        
        return metrics


class ValidationMetricsTracker:
    """
    Track validation metrics over time and save to CSV
    """
    
    def __init__(self, save_path: str, metrics_list: Optional[List[str]] = None):
        """
        Args:
            save_path: Path to save metrics CSV
            metrics_list: List of metrics to track
        """
        self.save_path = save_path
        self.metrics_list = metrics_list or ['psnr', 'ssim', 'ms_ssim', 'nmse', 'ncc', 'mi']
        
        # Initialize tracking
        self.epoch_metrics = defaultdict(list)
        self.best_metrics = {}
        self.metrics_history = []
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Initialize CSV with headers
        self._initialize_csv()
        
        print(f"📊 Initialized metrics tracker - saving to {save_path}")
    
    def _initialize_csv(self):
        """Initialize CSV file with headers"""
        headers = ['epoch'] + self.metrics_list + ['best_psnr', 'best_ssim']
        
        with open(self.save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
    
    def update(self, epoch: int, metrics: Dict[str, float]):
        """
        Update metrics for an epoch
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of computed metrics
        """
        # Store metrics for this epoch
        epoch_data = {'epoch': epoch}
        
        for metric_name in self.metrics_list:
            value = metrics.get(metric_name, 0.0)
            epoch_data[metric_name] = value
            self.epoch_metrics[metric_name].append(value)
        
        # Track best metrics
        if 'psnr' in metrics:
            if 'psnr' not in self.best_metrics or metrics['psnr'] > self.best_metrics['psnr']:
                self.best_metrics['psnr'] = metrics['psnr']
        
        if 'ssim' in metrics:
            if 'ssim' not in self.best_metrics or metrics['ssim'] > self.best_metrics['ssim']:
                self.best_metrics['ssim'] = metrics['ssim']
        
        epoch_data['best_psnr'] = self.best_metrics.get('psnr', 0.0)
        epoch_data['best_ssim'] = self.best_metrics.get('ssim', 0.0)
        
        self.metrics_history.append(epoch_data)
        
        # Append to CSV
        self._append_to_csv(epoch_data)
    
    def _append_to_csv(self, epoch_data: Dict):
        """Append epoch data to CSV"""
        with open(self.save_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row = [epoch_data['epoch']]
            
            for metric_name in self.metrics_list:
                row.append(epoch_data.get(metric_name, 0.0))
            
            row.append(epoch_data.get('best_psnr', 0.0))
            row.append(epoch_data.get('best_ssim', 0.0))
            
            writer.writerow(row)
    
    def compute_and_track(self, 
                         generated: torch.Tensor, 
                         target: torch.Tensor, 
                         epoch: int) -> Dict[str, float]:
        """
        Compute metrics and track them
        
        Args:
            generated: Generated images
            target: Target images
            epoch: Current epoch
            
        Returns:
            Computed metrics
        """
        quality_metrics = ImageQualityMetrics(device=generated.device)
        metrics = quality_metrics.compute_all_metrics(generated, target)
        
        self.update(epoch, metrics)
        return metrics
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best metrics achieved so far"""
        return self.best_metrics.copy()
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get latest metrics"""
        if self.metrics_history:
            return self.metrics_history[-1].copy()
        return {}
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics"""
        summary = {}
        
        for metric_name in self.metrics_list:
            values = self.epoch_metrics[metric_name]
            if values:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'latest': values[-1]
                }
            else:
                summary[metric_name] = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'latest': 0}
        
        return summary
    
    def print_summary(self):
        """Print summary of tracked metrics"""
        print(f"\n📊 Validation Metrics Summary:")
        print(f"   Total epochs tracked: {len(self.metrics_history)}")
        
        best_metrics = self.get_best_metrics()
        latest_metrics = self.get_latest_metrics()
        
        for metric_name in self.metrics_list:
            best_val = best_metrics.get(metric_name, 0)
            latest_val = latest_metrics.get(metric_name, 0)
            
            print(f"   {metric_name.upper()}:")
            print(f"     Best: {best_val:.6f}")
            print(f"     Latest: {latest_val:.6f}")
        
        print(f"   Metrics saved to: {self.save_path}")
    
    def save_summary_json(self, filepath: str):
        """Save summary statistics to JSON"""
        summary = {
            'best_metrics': self.get_best_metrics(),
            'latest_metrics': self.get_latest_metrics(),
            'summary_statistics': self.get_summary_statistics(),
            'total_epochs': len(self.metrics_history)
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Summary saved to: {filepath}")


def run_validation_with_metrics(validation_loader, 
                               encoder, 
                               generator, 
                               metrics_tracker: ValidationMetricsTracker,
                               device: str = 'cuda',
                               max_batches: Optional[int] = None) -> Dict[str, float]:
    """
    Run validation and compute comprehensive image quality metrics
    
    Args:
        validation_loader: DataLoader for validation data
        encoder: Encoder model
        generator: Generator model
        metrics_tracker: Metrics tracker instance
        device: Device to run on
        max_batches: Maximum number of batches to process (for speed)
        
    Returns:
        Dictionary with averaged metrics
    """
    encoder.eval()
    generator.eval()
    
    quality_metrics = ImageQualityMetrics(device=device)
    
    all_metrics = defaultdict(list)
    batch_count = 0
    
    print(f"📊 Running validation with image quality metrics...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(validation_loader, desc="Validation Metrics")):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            try:
                # Get batch data
                input_volume = batch["input_path"].to(device)
                target_volume = batch["target_path"].to(device)
                target_phase = batch["target_phase"].to(device)
                
                # Forward pass
                latent = encoder(input_volume)
                
                # Create phase embeddings
                from training import create_phase_embeddings
                phase_emb = create_phase_embeddings(target_phase, dim=32, device=device)
                
                # Generate volume
                generated_volume = generator(latent, phase_emb)
                
                # Ensure both volumes are in the same range [0, 1]
                generated_volume = torch.clamp(generated_volume, 0, 1)
                target_volume = torch.clamp(target_volume, 0, 1)
                
                # Compute metrics for each sample in batch
                for i in range(generated_volume.shape[0]):
                    gen_sample = generated_volume[i:i+1]
                    target_sample = target_volume[i:i+1]
                    
                    sample_metrics = quality_metrics.compute_all_metrics(gen_sample, target_sample)
                    
                    for metric_name, value in sample_metrics.items():
                        if not (np.isnan(value) or np.isinf(value)):
                            all_metrics[metric_name].append(value)
                
                batch_count += 1
                
            except Exception as e:
                print(f"⚠️  Error in validation batch {batch_idx}: {e}")
                continue
    
    # Compute average metrics
    avg_metrics = {}
    for metric_name, values in all_metrics.items():
        if values:
            avg_metrics[metric_name] = np.mean(values)
        else:
            avg_metrics[metric_name] = 0.0
    
    print(f"✅ Validation completed - processed {batch_count} batches")
    print(f"📊 Validation Metrics:")
    for metric_name, value in avg_metrics.items():
        print(f"   {metric_name.upper()}: {value:.6f}")
    
    return avg_metrics


def compute_metrics_for_volume_pair(generated_volume: torch.Tensor, 
                                  target_volume: torch.Tensor,
                                  device: str = 'cuda') -> Dict[str, float]:
    """
    Compute all metrics for a single volume pair
    
    Args:
        generated_volume: Generated volume tensor
        target_volume: Target volume tensor
        device: Device to run on
        
    Returns:
        Dictionary with computed metrics
    """
    quality_metrics = ImageQualityMetrics(device=device)
    
    # Ensure tensors are on the correct device
    generated_volume = generated_volume.to(device)
    target_volume = target_volume.to(device)
    
    # Ensure both volumes are in [0, 1] range
    generated_volume = torch.clamp(generated_volume, 0, 1)
    target_volume = torch.clamp(target_volume, 0, 1)
    
    return quality_metrics.compute_all_metrics(generated_volume, target_volume)


class MetricsComparison:
    """
    Compare metrics between different models or epochs
    """
    
    def __init__(self):
        self.model_metrics = defaultdict(list)
    
    def add_model_metrics(self, model_name: str, metrics: Dict[str, float]):
        """Add metrics for a model"""
        self.model_metrics[model_name].append(metrics)
    
    def compare_models(self, metric_name: str) -> Dict[str, Dict[str, float]]:
        """
        Compare models on a specific metric
        
        Args:
            metric_name: Name of metric to compare
            
        Returns:
            Comparison statistics for each model
        """
        comparison = {}
        
        for model_name, metrics_list in self.model_metrics.items():
            values = [m.get(metric_name, 0) for m in metrics_list]
            
            if values:
                comparison[model_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        return comparison
    
    def get_best_model(self, metric_name: str, higher_is_better: bool = True) -> str:
        """
        Get the best model for a specific metric
        
        Args:
            metric_name: Name of metric
            higher_is_better: Whether higher values are better
            
        Returns:
            Name of best model
        """
        comparison = self.compare_models(metric_name)
        
        if not comparison:
            return ""
        
        if higher_is_better:
            best_model = max(comparison.keys(), key=lambda k: comparison[k]['mean'])
        else:
            best_model = min(comparison.keys(), key=lambda k: comparison[k]['mean'])
        
        return best_model


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Image Quality Metrics...")
    
    # Create test data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img1 = torch.randn(1, 1, 32, 64, 64).to(device)
    img2 = img1 + 0.1 * torch.randn_like(img1)  # Add some noise
    
    # Initialize metrics
    quality_metrics = ImageQualityMetrics(device=device)
    
    # Test individual metrics
    psnr_val = quality_metrics.psnr(img1, img2)
    ssim_val = quality_metrics.ssim(img1, img2)
    nmse_val = quality_metrics.nmse(img1, img2)
    ncc_val = quality_metrics.ncc(img1, img2)
    
    print(f"PSNR: {psnr_val:.4f}")
    print(f"SSIM: {ssim_val:.4f}")
    print(f"NMSE: {nmse_val:.6f}")
    print(f"NCC: {ncc_val:.4f}")
    
    # Test all metrics
    all_metrics = quality_metrics.compute_all_metrics(img1, img2)
    print("\nAll metrics:", all_metrics)
    
    # Test metrics tracker
    tracker = ValidationMetricsTracker('test_metrics.csv')
    tracker.update(1, all_metrics)
    tracker.print_summary()
    
    print("✅ Image quality metrics tests completed!")