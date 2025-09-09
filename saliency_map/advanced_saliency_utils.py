#!/usr/bin/env python3
"""
Advanced saliency analysis utilities for 3D medical image interpretation
Provides enhanced saliency map generation, analysis, and visualization tools
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from scipy.ndimage import label, center_of_mass
from skimage import measure, segmentation, filters
from sklearn.cluster import KMeans
import cv2
import os
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class AdvancedSaliencyAnalyzer:
    """
    Advanced saliency analysis for 3D medical images with multiple techniques
    """
    
    def __init__(self, model, phase_mapping=None):
        self.model = model
        self.phase_mapping = phase_mapping or {0: 'Non-contrast', 1: 'Arterial', 2: 'Venous', 3: 'Delayed', 4: 'Hepatobiliary'}
        self.device = next(model.parameters()).device
        
    def generate_integrated_gradients(self, volume, target_class=None, steps=50, baseline=None):
        """
        Generate Integrated Gradients saliency map
        
        Args:
            volume: Input volume (1, 1, D, H, W)
            target_class: Target class index
            steps: Number of integration steps
            baseline: Baseline image (default: zeros)
        
        Returns:
            integrated_gradients: Saliency map
            predicted_class: Predicted class
            confidence: Prediction confidence
        """
        self.model.eval()
        volume = volume.to(self.device)
        volume.requires_grad_(True)
        
        # Get prediction
        with torch.no_grad():
            logits = self.model(volume)
            probs = F.softmax(logits, dim=1)
            if target_class is None:
                target_class = torch.argmax(logits, dim=1).item()
            confidence = probs[0, target_class].item()
        
        # Set baseline (default: zero image)
        if baseline is None:
            baseline = torch.zeros_like(volume)
        else:
            baseline = baseline.to(self.device)
        
        # Generate path from baseline to input
        alphas = torch.linspace(0, 1, steps).to(self.device)
        
        integrated_grads = torch.zeros_like(volume)
        
        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated = baseline + alpha * (volume - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            logits = self.model(interpolated)
            
            # Backward pass for target class
            self.model.zero_grad()
            class_score = logits[0, target_class]
            class_score.backward(retain_graph=True)
            
            # Accumulate gradients
            integrated_grads += interpolated.grad.data
        
        # Average gradients and multiply by input difference
        integrated_grads = integrated_grads / steps
        integrated_grads = integrated_grads * (volume - baseline)
        
        # Convert to numpy and normalize
        saliency_map = torch.abs(integrated_grads[0, 0]).cpu().numpy()
        saliency_map = self._normalize_saliency_map(saliency_map)
        
        return saliency_map, target_class, confidence
    
    def generate_guided_backprop(self, volume, target_class=None):
        """
        Generate Guided Backpropagation saliency map
        
        Args:
            volume: Input volume (1, 1, D, H, W)
            target_class: Target class index
        
        Returns:
            guided_gradients: Saliency map
            predicted_class: Predicted class
            confidence: Prediction confidence
        """
        # Store original ReLU forward functions
        relu_outputs = []
        
        def relu_backward_hook_function(module, grad_in, grad_out):
            """Hook function for guided backprop"""
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        
        # Register hooks for all ReLU layers
        hooks = []
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                hooks.append(module.register_backward_hook(relu_backward_hook_function))
        
        try:
            self.model.eval()
            volume = volume.to(self.device)
            volume.requires_grad_(True)
            
            # Forward pass
            logits = self.model(volume)
            probs = F.softmax(logits, dim=1)
            
            if target_class is None:
                target_class = torch.argmax(logits, dim=1).item()
            confidence = probs[0, target_class].item()
            
            # Backward pass
            self.model.zero_grad()
            class_score = logits[0, target_class]
            class_score.backward()
            
            # Get guided gradients
            guided_gradients = volume.grad.data
            saliency_map = torch.abs(guided_gradients[0, 0]).cpu().numpy()
            saliency_map = self._normalize_saliency_map(saliency_map)
            
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        return saliency_map, target_class, confidence
    
    def generate_smoothgrad(self, volume, target_class=None, noise_level=0.1, n_samples=50):
        """
        Generate SmoothGrad saliency map by averaging gradients over noisy inputs
        
        Args:
            volume: Input volume (1, 1, D, H, W)
            target_class: Target class index
            noise_level: Standard deviation of noise
            n_samples: Number of noisy samples
        
        Returns:
            smoothgrad: Averaged saliency map
            predicted_class: Predicted class
            confidence: Prediction confidence
        """
        self.model.eval()
        volume = volume.to(self.device)
        
        # Get clean prediction
        with torch.no_grad():
            clean_logits = self.model(volume)
            if target_class is None:
                target_class = torch.argmax(clean_logits, dim=1).item()
            confidence = F.softmax(clean_logits, dim=1)[0, target_class].item()
        
        # Accumulate gradients over noisy samples
        accumulated_gradients = torch.zeros_like(volume)
        
        for _ in range(n_samples):
            # Add noise
            noise = torch.randn_like(volume) * noise_level
            noisy_volume = volume + noise
            noisy_volume.requires_grad_(True)
            
            # Forward pass
            logits = self.model(noisy_volume)
            
            # Backward pass
            self.model.zero_grad()
            class_score = logits[0, target_class]
            class_score.backward()
            
            # Accumulate gradients
            accumulated_gradients += noisy_volume.grad.data
        
        # Average gradients
        smoothgrad = accumulated_gradients / n_samples
        saliency_map = torch.abs(smoothgrad[0, 0]).cpu().numpy()
        saliency_map = self._normalize_saliency_map(saliency_map)
        
        return saliency_map, target_class, confidence
    
    def generate_lime_3d(self, volume, target_class=None, n_samples=1000, n_features=100):
        """
        Generate LIME-based explanation for 3D volume
        
        Args:
            volume: Input volume (1, 1, D, H, W)
            target_class: Target class index
            n_samples: Number of perturbation samples
            n_features: Number of superpixel features
        
        Returns:
            lime_map: LIME importance map
            predicted_class: Predicted class
            confidence: Prediction confidence
        """
        self.model.eval()
        volume_np = volume[0, 0].cpu().numpy()
        
        # Get original prediction
        with torch.no_grad():
            original_logits = self.model(volume.to(self.device))
            if target_class is None:
                target_class = torch.argmax(original_logits, dim=1).item()
            confidence = F.softmax(original_logits, dim=1)[0, target_class].item()
        
        # Create 3D superpixels using watershed
        print("Creating 3D superpixels...")
        superpixels = self._create_3d_superpixels(volume_np, n_features)
        n_superpixels = len(np.unique(superpixels))
        
        # Generate random perturbations
        print(f"Generating {n_samples} perturbation samples...")
        perturbations = np.random.binomial(1, 0.5, (n_samples, n_superpixels))
        predictions = []
        
        for i, perturbation in enumerate(perturbations):
            if i % 100 == 0:
                print(f"Processing sample {i}/{n_samples}")
            
            # Create perturbed image
            perturbed_volume = self._apply_perturbation_3d(volume_np, superpixels, perturbation)
            perturbed_tensor = torch.tensor(perturbed_volume).unsqueeze(0).unsqueeze(0).float().to(self.device)
            
            # Get prediction
            with torch.no_grad():
                logits = self.model(perturbed_tensor)
                prob = F.softmax(logits, dim=1)[0, target_class].item()
                predictions.append(prob)
        
        # Fit linear model
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(perturbations, predictions)
        
        # Create importance map
        importance_scores = ridge.coef_
        lime_map = np.zeros_like(volume_np)
        
        for i, score in enumerate(importance_scores):
            lime_map[superpixels == i] = score
        
        # Normalize
        lime_map = self._normalize_saliency_map(lime_map)
        
        return lime_map, target_class, confidence
    
    def _create_3d_superpixels(self, volume, n_segments):
        """Create 3D superpixels using watershed algorithm"""
        # Apply Gaussian filter for smoothing
        smoothed = filters.gaussian(volume, sigma=1.0)
        
        # Use gradient magnitude for watershed
        gradient_magnitude = np.sqrt(np.sum([np.gradient(smoothed, axis=i)**2 for i in range(3)], axis=0))
        
        # Create markers using local minima
        from skimage.feature import peak_local_minima
        from skimage.morphology import h_minima
        
        # Find local minima
        local_minima = h_minima(gradient_magnitude, 0.1)
        markers, _ = label(local_minima)
        
        # Limit number of markers
        if markers.max() > n_segments:
            # Keep only the strongest markers
            marker_values = []
            for i in range(1, markers.max() + 1):
                mask = markers == i
                if np.sum(mask) > 0:
                    avg_grad = np.mean(gradient_magnitude[mask])
                    marker_values.append((i, avg_grad))
            
            # Sort by gradient strength and keep top n_segments
            marker_values.sort(key=lambda x: x[1])
            keep_markers = [x[0] for x in marker_values[:n_segments]]
            
            new_markers = np.zeros_like(markers)
            for i, old_idx in enumerate(keep_markers):
                new_markers[markers == old_idx] = i + 1
            markers = new_markers
        
        # Apply watershed
        from skimage.segmentation import watershed
        superpixels = watershed(gradient_magnitude, markers)
        
        return superpixels
    
    def _apply_perturbation_3d(self, volume, superpixels, perturbation):
        """Apply LIME perturbation to 3D volume"""
        perturbed = volume.copy()
        
        for i, keep in enumerate(perturbation):
            if not keep:
                # Replace superpixel with mean value
                mask = superpixels == i
                if np.sum(mask) > 0:
                    perturbed[mask] = np.mean(volume[mask])
        
        return perturbed
    
    def analyze_saliency_regions(self, saliency_map, volume, threshold=0.7):
        """
        Analyze important regions in saliency map
        
        Args:
            saliency_map: 3D saliency map
            volume: Original 3D volume
            threshold: Threshold for important regions
        
        Returns:
            analysis: Dictionary with region analysis
        """
        # Threshold saliency map
        binary_map = saliency_map > threshold
        
        # Find connected components
        labeled_regions, n_regions = label(binary_map)
        
        analysis = {
            'n_regions': n_regions,
            'total_volume': np.sum(binary_map),
            'relative_volume': np.sum(binary_map) / np.prod(saliency_map.shape),
            'regions': []
        }
        
        for region_id in range(1, n_regions + 1):
            region_mask = labeled_regions == region_id
            
            # Calculate region properties
            region_volume = np.sum(region_mask)
            centroid = center_of_mass(region_mask)
            
            # Calculate intensity statistics in original volume
            region_intensities = volume[region_mask]
            
            # Calculate saliency statistics
            region_saliency = saliency_map[region_mask]
            
            region_info = {
                'id': region_id,
                'volume': region_volume,
                'centroid': centroid,
                'intensity_mean': np.mean(region_intensities),
                'intensity_std': np.std(region_intensities),
                'saliency_mean': np.mean(region_saliency),
                'saliency_max': np.max(region_saliency),
                'bbox': self._get_region_bbox(region_mask)
            }
            
            analysis['regions'].append(region_info)
        
        # Sort regions by volume (largest first)
        analysis['regions'].sort(key=lambda x: x['volume'], reverse=True)
        
        return analysis
    
    def _get_region_bbox(self, mask):
        """Get bounding box of a region"""
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return None
        
        bbox = []
        for dim in range(3):
            bbox.extend([np.min(coords[dim]), np.max(coords[dim])])
        
        return bbox
    
    def compare_saliency_methods(self, volume, target_class=None, save_path=None):
        """
        Compare different saliency methods on the same volume
        
        Args:
            volume: Input volume (1, 1, D, H, W)
            target_class: Target class index
            save_path: Path to save comparison plot
        
        Returns:
            comparison_results: Dictionary with all saliency maps and metrics
        """
        print("Generating saliency maps using different methods...")
        
        methods = {
            'GradCAM': lambda: self.model.generate_gradcam(volume, target_class),
            'Guided Backprop': lambda: self.generate_guided_backprop(volume, target_class),
            'Integrated Gradients': lambda: self.generate_integrated_gradients(volume, target_class),
            'SmoothGrad': lambda: self.generate_smoothgrad(volume, target_class, n_samples=30)
        }
        
        results = {}
        
        for method_name, method_func in methods.items():
            print(f"Generating {method_name}...")
            try:
                saliency_map, pred_class, confidence = method_func()
                results[method_name] = {
                    'saliency_map': saliency_map,
                    'predicted_class': pred_class,
                    'confidence': confidence
                }
            except Exception as e:
                print(f"Failed to generate {method_name}: {e}")
                results[method_name] = None
        
        # Create comparison visualization
        if save_path:
            self._plot_saliency_comparison(volume, results, save_path)
        
        return results
    
    def _plot_saliency_comparison(self, volume, results, save_path):
        """Plot comparison of different saliency methods"""
        volume_np = volume[0, 0].cpu().numpy()
        
        # Select middle slices for visualization
        depth = volume_np.shape[0]
        slice_indices = [depth//4, depth//2, 3*depth//4]
        
        # Filter out failed methods
        valid_methods = {k: v for k, v in results.items() if v is not None}
        n_methods = len(valid_methods)
        
        if n_methods == 0:
            print("No valid saliency methods to plot")
            return
        
        # Create figure
        fig, axes = plt.subplots(len(slice_indices), n_methods + 1, figsize=(4*(n_methods+1), 12))
        if len(slice_indices) == 1:
            axes = axes.reshape(1, -1)
        
        for row, slice_idx in enumerate(slice_indices):
            # Original slice
            axes[row, 0].imshow(volume_np[slice_idx], cmap='gray')
            axes[row, 0].set_title(f'Original Slice {slice_idx}')
            axes[row, 0].axis('off')
            
            # Saliency maps
            for col, (method_name, result) in enumerate(valid_methods.items()):
                saliency_map = result['saliency_map']
                confidence = result['confidence']
                
                im = axes[row, col+1].imshow(saliency_map[slice_idx], cmap='hot')
                axes[row, col+1].set_title(f'{method_name}\n(Conf: {confidence:.3f})')
                axes[row, col+1].axis('off')
                
                # Add colorbar for first row
                if row == 0:
                    plt.colorbar(im, ax=axes[row, col+1], fraction=0.046, pad=0.04)
        
        plt.suptitle('Saliency Method Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saliency comparison saved to: {save_path}")
    
    def generate_attribution_report(self, volume, volume_id=None, save_dir=None):
        """
        Generate comprehensive attribution report for a volume
        
        Args:
            volume: Input volume (1, 1, D, H, W)
            volume_id: Identifier for the volume
            save_dir: Directory to save report
        
        Returns:
            report: Comprehensive attribution report
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        volume_id = volume_id or f"volume_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"Generating attribution report for {volume_id}...")
        
        # Basic prediction
        with torch.no_grad():
            logits = self.model(volume.to(self.device))
            probs = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probs[0, predicted_class].item()
            
        phase_name = self.phase_mapping.get(predicted_class, f'Phase_{predicted_class}')
        
        # Generate multiple saliency maps
        saliency_results = self.compare_saliency_methods(
            volume, 
            predicted_class, 
            save_path=save_dir / f'{volume_id}_saliency_comparison.png' if save_dir else None
        )
        
        # Analyze regions for each method
        region_analyses = {}
        for method_name, result in saliency_results.items():
            if result is not None:
                analysis = self.analyze_saliency_regions(
                    result['saliency_map'], 
                    volume[0, 0].cpu().numpy()
                )
                region_analyses[method_name] = analysis
        
        # Create report
        report = {
            'volume_id': volume_id,
            'timestamp': datetime.now().isoformat(),
            'prediction': {
                'predicted_class': predicted_class,
                'phase_name': phase_name,
                'confidence': confidence,
                'all_probabilities': probs[0].cpu().numpy().tolist()
            },
            'volume_info': {
                'shape': list(volume.shape),
                'intensity_range': [float(volume.min()), float(volume.max())],
                'intensity_mean': float(volume.mean()),
                'intensity_std': float(volume.std())
            },
            'saliency_methods': {},
            'region_analyses': region_analyses
        }
        
        # Add saliency method results
        for method_name, result in saliency_results.items():
            if result is not None:
                report['saliency_methods'][method_name] = {
                    'confidence': result['confidence'],
                    'saliency_stats': {
                        'mean': float(np.mean(result['saliency_map'])),
                        'std': float(np.std(result['saliency_map'])),
                        'max': float(np.max(result['saliency_map'])),
                        'min': float(np.min(result['saliency_map']))
                    }
                }
        
        # Save report
        if save_dir:
            report_path = save_dir / f'{volume_id}_attribution_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Attribution report saved to: {report_path}")
        
        return report
    
    def _normalize_saliency_map(self, saliency_map):
        """Normalize saliency map to [0, 1]"""
        saliency_map = saliency_map - saliency_map.min()
        if saliency_map.max() > 0:
            saliency_map = saliency_map / saliency_map.max()
        return saliency_map


class SaliencyEvaluator:
    """
    Evaluate saliency maps using various metrics
    """
    
    def __init__(self, model, ground_truth_masks=None):
        self.model = model
        self.ground_truth_masks = ground_truth_masks
    
    def pointing_game_accuracy(self, saliency_maps, ground_truth_points):
        """
        Evaluate pointing game accuracy
        
        Args:
            saliency_maps: List of saliency maps
            ground_truth_points: List of ground truth points (3D coordinates)
        
        Returns:
            accuracy: Pointing game accuracy
        """
        hits = 0
        total = len(saliency_maps)
        
        for saliency_map, gt_point in zip(saliency_maps, ground_truth_points):
            # Find peak in saliency map
            peak_location = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
            
            # Check if peak is close to ground truth point
            distance = np.sqrt(np.sum([(p - g)**2 for p, g in zip(peak_location, gt_point)]))
            
            # Consider hit if within reasonable distance (e.g., 10 voxels)
            if distance <= 10:
                hits += 1
        
        accuracy = hits / total if total > 0 else 0
        return accuracy
    
    def calculate_iou(self, saliency_map, ground_truth_mask, threshold=0.5):
        """
        Calculate Intersection over Union between saliency map and ground truth
        
        Args:
            saliency_map: Saliency map
            ground_truth_mask: Ground truth binary mask
            threshold: Threshold for binarizing saliency map
        
        Returns:
            iou: Intersection over Union score
        """
        # Binarize saliency map
        binary_saliency = saliency_map > threshold
        
        # Calculate IoU
        intersection = np.logical_and(binary_saliency, ground_truth_mask)
        union = np.logical_or(binary_saliency, ground_truth_mask)
        
        iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
        return iou
    
    def sensitivity_analysis(self, volume, target_class, perturbation_sizes=[0.1, 0.2, 0.3]):
        """
        Analyze sensitivity of saliency maps to input perturbations
        
        Args:
            volume: Input volume
            target_class: Target class
            perturbation_sizes: List of perturbation magnitudes
        
        Returns:
            sensitivity_scores: Sensitivity scores for each perturbation size
        """
        # Generate original saliency map
        original_saliency, _, _ = self.model.generate_gradcam(volume, target_class)
        
        sensitivity_scores = []
        
        for pert_size in perturbation_sizes:
            similarities = []
            
            # Generate multiple perturbed versions
            for _ in range(10):
                # Add random noise
                noise = torch.randn_like(volume) * pert_size
                perturbed_volume = volume + noise
                
                # Generate saliency map for perturbed input
                perturbed_saliency, _, _ = self.model.generate_gradcam(perturbed_volume, target_class)
                
                # Calculate similarity (correlation)
                similarity = np.corrcoef(original_saliency.flatten(), perturbed_saliency.flatten())[0, 1]
                similarities.append(similarity)
            
            # Average similarity for this perturbation size
            avg_similarity = np.mean(similarities)
            sensitivity_scores.append(avg_similarity)
        
        return dict(zip(perturbation_sizes, sensitivity_scores))


def create_saliency_analyzer(model, phase_mapping=None):
    """
    Factory function to create saliency analyzer
    
    Args:
        model: Trained ContrastPhaseMLPClassifier
        phase_mapping: Mapping from class indices to phase names
    
    Returns:
        analyzer: AdvancedSaliencyAnalyzer instance
    """
    return AdvancedSaliencyAnalyzer(model, phase_mapping)


# Example usage and testing
def test_advanced_saliency():
    """Test advanced saliency analysis"""
    print("Testing Advanced Saliency Analysis...")
    
    # Create dummy model for testing
    from contrast_phase_mlp_classifier import ContrastPhaseMLPClassifier
    import torch.nn as nn
    
    class DummyEncoder(nn.Module):
        def __init__(self, latent_dim=256):
            super().__init__()
            self.latent_dim = latent_dim
            self.conv = nn.Conv3d(1, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool3d((2, 2, 2))
            self.fc = nn.Linear(32 * 2 * 2 * 2, latent_dim)
            
        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    # Create model
    encoder = DummyEncoder()
    model = ContrastPhaseMLPClassifier(
        encoder=encoder,
        encoder_name="TestEncoder",
        n_classes=5,
        enable_gradcam=True
    )
    
    # Create analyzer
    analyzer = create_saliency_analyzer(model)
    
    # Test data
    test_volume = torch.randn(1, 1, 32, 32, 32)
    
    # Test different saliency methods
    print("Testing Integrated Gradients...")
    try:
        ig_map, pred, conf = analyzer.generate_integrated_gradients(test_volume, steps=10)
        print(f"✅ Integrated Gradients: shape {ig_map.shape}, pred {pred}, conf {conf:.3f}")
    except Exception as e:
        print(f"❌ Integrated Gradients failed: {e}")
    
    print("Testing Guided Backprop...")
    try:
        gb_map, pred, conf = analyzer.generate_guided_backprop(test_volume)
        print(f"✅ Guided Backprop: shape {gb_map.shape}, pred {pred}, conf {conf:.3f}")
    except Exception as e:
        print(f"❌ Guided Backprop failed: {e}")
    
    print("Testing SmoothGrad...")
    try:
        sg_map, pred, conf = analyzer.generate_smoothgrad(test_volume, n_samples=5)
        print(f"✅ SmoothGrad: shape {sg_map.shape}, pred {pred}, conf {conf:.3f}")
    except Exception as e:
        print(f"❌ SmoothGrad failed: {e}")
    
    print("Testing method comparison...")
    try:
        comparison = analyzer.compare_saliency_methods(test_volume)
        print(f"✅ Method comparison: {len(comparison)} methods compared")
    except Exception as e:
        print(f"❌ Method comparison failed: {e}")
    
    print("Testing region analysis...")
    try:
        # Use any available saliency map
        saliency_map = np.random.rand(32, 32, 32)
        volume_np = test_volume[0, 0].numpy()
        analysis = analyzer.analyze_saliency_regions(saliency_map, volume_np)
        print(f"✅ Region analysis: found {analysis['n_regions']} regions")
    except Exception as e:
        print(f"❌ Region analysis failed: {e}")
    
    print("Advanced saliency testing completed!")


if __name__ == "__main__":
    test_advanced_saliency()