import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import accuracy_score
from captum import attr
from captum.attr import IntegratedGradients, Saliency, GradCam, Occlusion
import copy

class SaliencyMapGenerator:
    """
    Generate saliency maps for contrast phase detection pipeline
    """
    
    def __init__(self, encoder, lda_classifier, scaler=None, device='cuda'):
        self.encoder = encoder
        self.lda_classifier = lda_classifier
        self.scaler = scaler
        self.device = device
        
        # Create end-to-end model for gradient computation
        self.end_to_end_model = self._create_end_to_end_model()
        
    def _create_end_to_end_model(self):
        """Create end-to-end differentiable model"""
        
        class EndToEndModel(nn.Module):
            def __init__(self, encoder, lda_classifier, scaler):
                super().__init__()
                self.encoder = encoder
                # Convert LDA to PyTorch for differentiability
                self.lda_weights = torch.tensor(lda_classifier.coef_, dtype=torch.float32)
                self.lda_intercept = torch.tensor(lda_classifier.intercept_, dtype=torch.float32)
                self.scaler = scaler
                
            def forward(self, x):
                # Enable gradients for encoder
                features = self.encoder(x)
                
                # Apply scaling if used
                if self.scaler is not None:
                    # Convert sklearn scaler to torch operations
                    mean = torch.tensor(self.scaler.mean_, dtype=torch.float32, device=x.device)
                    scale = torch.tensor(self.scaler.scale_, dtype=torch.float32, device=x.device)
                    features = (features - mean) / scale
                
                # Apply LDA transformation (simplified to linear layer)
                logits = F.linear(features, self.lda_weights, self.lda_intercept)
                return logits
        
        return EndToEndModel(self.encoder, self.lda_classifier, self.scaler).to(self.device)
    
    def modify_encoder_for_gradients(self):
        """Remove torch.no_grad() from encoder to enable gradient flow"""
        
        # For DINO encoder, we need to modify the forward pass
        if hasattr(self.encoder, 'dino'):
            # Create a wrapper that enables gradients
            original_forward = self.encoder.forward
            
            def gradient_enabled_forward(volume_3d):
                batch_size, _, depth, height, width = volume_3d.shape
                device = volume_3d.device
                
                slice_indices = self.encoder._sample_slices(volume_3d)
                slice_features = []
                
                for idx in slice_indices:
                    slice_2d = volume_3d[:, :, idx, :, :]
                    slice_2d = self.encoder._preprocess_slice(slice_2d)
                    
                    # ENABLE GRADIENTS HERE - Remove torch.no_grad()
                    features = self.encoder.dino(slice_2d)
                    
                    # Handle different output formats
                    if isinstance(features, dict):
                        features = features.get('x_norm_clstoken', 
                                               features.get('cls_token', 
                                                           features.get('x', features)))
                    elif isinstance(features, tuple):
                        features = features[0]
                    
                    if features.dim() > 2:
                        features = features.view(features.shape[0], -1)
                    
                    slice_latent = self.encoder.slice_projection(features.float())
                    slice_latent = self.encoder.slice_norm(slice_latent)
                    slice_features.append(slice_latent)
                
                # Rest of the forward pass remains the same
                volume_features = torch.stack(slice_features, dim=1)
                num_slices = volume_features.shape[1]
                
                if num_slices <= self.encoder.pos_encoding.shape[0]:
                    pos_enc = self.encoder.pos_encoding[:num_slices].unsqueeze(0).expand(batch_size, -1, -1)
                    volume_features = volume_features + pos_enc.to(device)
                
                aggregated_features = self.encoder.slice_aggregator(volume_features)
                final_features = aggregated_features.mean(dim=1)
                projected = self.encoder.final_projection(final_features)
                final_features = final_features + projected
                final_features = self.encoder.final_norm(final_features)
                
                return final_features
            
            # Replace the forward method
            self.encoder.forward = gradient_enabled_forward
    
    def generate_gradient_saliency(self, input_volume, target_phase=None, smooth_grad=False, n_samples=50):
        """
        Generate gradient-based saliency maps
        
        Args:
            input_volume: Input 3D volume [1, 1, D, H, W]
            target_phase: Target phase for saliency (if None, uses predicted phase)
            smooth_grad: Whether to use SmoothGrad
            n_samples: Number of samples for SmoothGrad
        """
        
        self.modify_encoder_for_gradients()
        self.end_to_end_model.eval()
        
        input_volume = input_volume.to(self.device).requires_grad_(True)
        
        if smooth_grad:
            # SmoothGrad: Add noise and average gradients
            saliency_maps = []
            
            for _ in range(n_samples):
                # Add noise
                noise = torch.randn_like(input_volume) * 0.1
                noisy_input = input_volume + noise
                noisy_input.requires_grad_(True)
                
                # Forward pass
                output = self.end_to_end_model(noisy_input)
                
                if target_phase is None:
                    target_phase = output.argmax(dim=1)
                
                # Backward pass
                self.end_to_end_model.zero_grad()
                loss = output[0, target_phase]
                loss.backward()
                
                # Get gradients
                saliency_map = noisy_input.grad.abs()
                saliency_maps.append(saliency_map.cpu().detach())
            
            # Average saliency maps
            saliency_map = torch.stack(saliency_maps).mean(dim=0)
        
        else:
            # Standard gradient saliency
            output = self.end_to_end_model(input_volume)
            
            if target_phase is None:
                target_phase = output.argmax(dim=1)
            
            self.end_to_end_model.zero_grad()
            loss = output[0, target_phase]
            loss.backward()
            
            saliency_map = input_volume.grad.abs().cpu().detach()
        
        return saliency_map, target_phase.item() if torch.is_tensor(target_phase) else target_phase
    
    def generate_integrated_gradients(self, input_volume, target_phase=None, n_steps=50):
        """
        Generate Integrated Gradients saliency maps
        """
        
        self.modify_encoder_for_gradients()
        
        # Use Captum's IntegratedGradients
        ig = IntegratedGradients(self.end_to_end_model)
        
        input_volume = input_volume.to(self.device)
        
        # Get prediction if target not specified
        if target_phase is None:
            with torch.no_grad():
                output = self.end_to_end_model(input_volume)
                target_phase = output.argmax(dim=1).item()
        
        # Create baseline (zeros or mean)
        baseline = torch.zeros_like(input_volume)
        
        # Compute integrated gradients
        attributions = ig.attribute(
            input_volume, 
            baseline, 
            target=target_phase, 
            n_steps=n_steps
        )
        
        return attributions.cpu().detach(), target_phase
    
    def generate_occlusion_saliency(self, input_volume, target_phase=None, window_size=(8, 16, 16), stride=(4, 8, 8)):
        """
        Generate occlusion-based saliency maps
        """
        
        input_volume = input_volume.to(self.device)
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.end_to_end_model(input_volume)
            if target_phase is None:
                target_phase = baseline_output.argmax(dim=1).item()
            baseline_score = F.softmax(baseline_output, dim=1)[0, target_phase].item()
        
        # Initialize saliency map
        _, _, D, H, W = input_volume.shape
        saliency_map = torch.zeros_like(input_volume)
        
        # Occlude different regions
        for d in range(0, D - window_size[0] + 1, stride[0]):
            for h in range(0, H - window_size[1] + 1, stride[1]):
                for w in range(0, W - window_size[2] + 1, stride[2]):
                    # Create occluded input
                    occluded_input = input_volume.clone()
                    occluded_input[:, :, 
                                 d:d+window_size[0], 
                                 h:h+window_size[1], 
                                 w:w+window_size[2]] = 0
                    
                    # Get prediction with occlusion
                    with torch.no_grad():
                        occluded_output = self.end_to_end_model(occluded_input)
                        occluded_score = F.softmax(occluded_output, dim=1)[0, target_phase].item()
                    
                    # Compute importance as drop in confidence
                    importance = baseline_score - occluded_score
                    
                    # Assign importance to occluded region
                    saliency_map[:, :, 
                               d:d+window_size[0], 
                               h:h+window_size[1], 
                               w:w+window_size[2]] += importance
        
        return saliency_map.cpu().detach(), target_phase
    
    def generate_attention_maps(self, input_volume):
        """
        Generate attention maps from Vision Transformer (if applicable)
        """
        
        if not hasattr(self.encoder, 'dino'):
            print("Attention maps only available for Vision Transformer encoders")
            return None
        
        # This would require modifying the ViT to return attention weights
        # For now, return placeholder
        print("Attention map extraction requires ViT modification to return attention weights")
        return None
    
    def visualize_saliency_3d(self, saliency_map, original_volume, save_path=None, method_name="Saliency"):
        """
        Visualize 3D saliency maps
        """
        
        # Normalize saliency map
        saliency_map = saliency_map.squeeze()
        original_volume = original_volume.squeeze()
        
        # Select middle slices for visualization
        D, H, W = saliency_map.shape
        middle_slices = [D//4, D//2, 3*D//4]
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        for i, slice_idx in enumerate(middle_slices):
            # Original slice
            axes[i, 0].imshow(original_volume[slice_idx], cmap='gray')
            axes[i, 0].set_title(f'Original - Slice {slice_idx}')
            axes[i, 0].axis('off')
            
            # Saliency map
            saliency_slice = saliency_map[slice_idx]
            im1 = axes[i, 1].imshow(saliency_slice, cmap='hot')
            axes[i, 1].set_title(f'Saliency - Slice {slice_idx}')
            axes[i, 1].axis('off')
            plt.colorbar(im1, ax=axes[i, 1])
            
            # Overlay
            axes[i, 2].imshow(original_volume[slice_idx], cmap='gray', alpha=0.7)
            axes[i, 2].imshow(saliency_slice, cmap='hot', alpha=0.3)
            axes[i, 2].set_title(f'Overlay - Slice {slice_idx}')
            axes[i, 2].axis('off')
        
        plt.suptitle(f'{method_name} Maps for Phase Detection')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saliency visualization saved to: {save_path}")
        
        plt.show()
    
    def compare_saliency_methods(self, input_volume, target_phase=None, save_dir=None):
        """
        Compare different saliency methods
        """
        
        print("Generating saliency maps using different methods...")
        
        results = {}
        
        # Gradient saliency
        try:
            print("1. Gradient Saliency...")
            grad_saliency, pred_phase = self.generate_gradient_saliency(
                input_volume.clone(), target_phase
            )
            results['gradient'] = grad_saliency
            print(f"   Predicted phase: {pred_phase}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # SmoothGrad
        try:
            print("2. SmoothGrad...")
            smooth_saliency, _ = self.generate_gradient_saliency(
                input_volume.clone(), target_phase, smooth_grad=True, n_samples=20
            )
            results['smoothgrad'] = smooth_saliency
        except Exception as e:
            print(f"   Error: {e}")
        
        # Integrated Gradients
        try:
            print("3. Integrated Gradients...")
            ig_saliency, _ = self.generate_integrated_gradients(
                input_volume.clone(), target_phase, n_steps=30
            )
            results['integrated_gradients'] = ig_saliency
        except Exception as e:
            print(f"   Error: {e}")
        
        # Occlusion
        try:
            print("4. Occlusion...")
            occ_saliency, _ = self.generate_occlusion_saliency(
                input_volume.clone(), target_phase
            )
            results['occlusion'] = occ_saliency
        except Exception as e:
            print(f"   Error: {e}")
        
        # Visualize all methods
        if results:
            fig, axes = plt.subplots(len(results), 3, figsize=(15, 5*len(results)))
            
            original = input_volume.squeeze()
            middle_slice = original.shape[0] // 2
            
            for i, (method_name, saliency_map) in enumerate(results.items()):
                saliency_slice = saliency_map.squeeze()[middle_slice]
                
                # Original
                if len(results) == 1:
                    ax_row = axes
                else:
                    ax_row = axes[i]
                
                ax_row[0].imshow(original[middle_slice], cmap='gray')
                ax_row[0].set_title(f'Original (Slice {middle_slice})')
                ax_row[0].axis('off')
                
                # Saliency
                im = ax_row[1].imshow(saliency_slice, cmap='hot')
                ax_row[1].set_title(f'{method_name.replace("_", " ").title()}')
                ax_row[1].axis('off')
                plt.colorbar(im, ax=ax_row[1])
                
                # Overlay
                ax_row[2].imshow(original[middle_slice], cmap='gray', alpha=0.7)
                ax_row[2].imshow(saliency_slice, cmap='hot', alpha=0.3)
                ax_row[2].set_title('Overlay')
                ax_row[2].axis('off')
            
            plt.tight_layout()
            
            if save_dir:
                comparison_path = f"{save_dir}/saliency_comparison.png"
                plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
                print(f"Comparison saved to: {comparison_path}")
            
            plt.show()
        
        return results


# Example usage function
def demo_saliency_generation():
    """
    Demo function showing how to use saliency maps
    """
    
    # Assume you have a trained model
    # encoder = your_trained_encoder
    # lda_classifier = your_trained_lda
    # scaler = your_trained_scaler
    
    # Create saliency generator
    # saliency_gen = SaliencyMapGenerator(encoder, lda_classifier, scaler)
    
    # Load test volume
    # test_volume = torch.randn(1, 1, 64, 128, 128)  # Example volume
    
    # Generate saliency maps
    # results = saliency_gen.compare_saliency_methods(test_volume, save_dir="saliency_results")
    
    print("Demo function - replace with actual trained models")


if __name__ == "__main__":
    demo_saliency_generation()
    