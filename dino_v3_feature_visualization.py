# dino_v3_feature_visualization.py - Extract and visualize DINO v3 features

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from datetime import datetime
import warnings
import time
warnings.filterwarnings('ignore')

# Import your existing modules
from dino_encoder import DinoV3Encoder  # Make sure this import works from your models.py
from data import prepare_dataset_from_folders, prepare_data

# Import visualization functions from your existing file
from feature_visualization import (
    apply_dimensionality_reduction_flexible,
    plot_combined_2d_3d_visualization,
    create_interactive_combined_plot,
    explain_silhouette_score,
    compute_comprehensive_metrics,
    save_features_cache,
    load_features_cache
)

def extract_dino_v3_features_exact(data_loader, encoder, device, encoder_name, 
                                   use_cache=True):
    """
    Extract features using DINO v3 encoder - following the exact pattern from feature_visualization.py
    """
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_name = f"dino_v3_{encoder_name}_features.pkl"
    cache_path = os.path.join(cache_dir, cache_name)
    
    # Try to load from cache if enabled
    if use_cache and os.path.exists(cache_path):
        print(f"📁 Loading cached DINO v3 features from {cache_path}")
        try:
            return load_features_cache(cache_path)
        except Exception as e:
            print(f"⚠️  Cache loading failed: {e}, extracting fresh features")
    
    print(f"🔧 Extracting features with DINO v3 {encoder_name} encoder...")
    encoder.eval()
    
    all_features = []
    all_phases = []
    all_scan_ids = []
    
    processed_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Extracting DINO v3 features")):
                
            try:
                # Handle different batch formats
                if isinstance(batch, dict):
                    volumes = batch['volume'].to(device)
                    phases = batch['phase']
                    scan_ids = batch.get('scan_id', [f'scan_{batch_idx}_{i}' for i in range(len(phases))])
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    volumes, phases = batch[0].to(device), batch[1]
                    scan_ids = batch[2] if len(batch) > 2 else [f'scan_{batch_idx}_{i}' for i in range(len(phases))]
                else:
                    print(f"⚠️  Unexpected batch format: {type(batch)}")
                    continue
                
                # Extract features using DINO v3
                features = encoder(volumes)
                
                # Move to CPU and convert to numpy
                features_np = features.detach().cpu().numpy()
                
                # Handle phases (ensure they're numpy arrays)
                if torch.is_tensor(phases):
                    phases_np = phases.detach().cpu().numpy()
                else:
                    phases_np = np.array(phases)
                
                # Ensure scan_ids are strings
                if isinstance(scan_ids, torch.Tensor):
                    scan_ids = scan_ids.tolist()
                scan_ids = [str(sid) for sid in scan_ids]
                
                all_features.append(features_np)
                all_phases.extend(phases_np)
                all_scan_ids.extend(scan_ids)
                
                processed_samples += len(features_np)
                
                
            except Exception as e:
                print(f"⚠️  Error processing batch {batch_idx}: {e}")
                continue
    
    # Concatenate all features
    if all_features:
        features = np.vstack(all_features)
        phases = np.array(all_phases)
        scan_ids = np.array(all_scan_ids)
        
        print(f"✅ Extracted {features.shape[0]} feature vectors of dimension {features.shape[1]}")
        
        # Cache the results
        if use_cache:
            save_features_cache(features, phases, scan_ids, cache_path)
        
        return features, phases, scan_ids
    else:
        raise ValueError("No features were extracted successfully")

def create_dino_analysis_with_exact_functions(features, phases, scan_ids, encoder_name, output_dir, 
                                              create_interactive=False):
    """
    Create DINO v3 analysis using EXACT functions from feature_visualization.py
    """
    print(f"\n🎨 Creating comprehensive DINO v3 feature visualizations using exact functions...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Apply dimensionality reduction methods - using exact same list as feature_visualization.py
    methods = ['PCA', 'TSNE', 'LDA']
    
    for method in methods:
        print(f"\n📊 Applying 2D/3D {method} to DINO v3 {encoder_name} features...")
        
        try:
            # Use the EXACT function signature from feature_visualization.py
            kwargs = {}
            if method.upper() == 'TSNE':
                # Use correct parameters for t-SNE (not n_iter but max_iter)
                kwargs = {
                    'max_iter': 1000,  # Not n_iter!
                    'perplexity': min(30, len(features) // 4)
                }
            
            # Apply dimensionality reduction - EXACT function call
            reduced_2d, reduced_3d, reducers, dr_metrics = apply_dimensionality_reduction_flexible(
                features, phases, method=method.lower(), **kwargs
            )
            
            # Compute comprehensive metrics - EXACT function call
            comp_metrics = compute_comprehensive_metrics(reduced_2d, reduced_3d, phases)
            
            # Combine all metrics
            all_metrics = {**dr_metrics, **comp_metrics}
            
            # Create combined 2D/3D visualization - EXACT function call
            save_path = os.path.join(output_dir, f"dino_v3_{encoder_name}_{method.lower()}_combined.png")
            df = plot_combined_2d_3d_visualization(
                reduced_2d, reduced_3d, phases, scan_ids, method.upper(), 
                f"DINO_v3_{encoder_name}", all_metrics, save_path
            )
            
            # Create interactive plot if requested - EXACT function call
            if create_interactive:
                interactive_path = os.path.join(output_dir, f"dino_v3_{encoder_name}_{method.lower()}_interactive.html")
                create_interactive_combined_plot(
                    reduced_2d, reduced_3d, phases, scan_ids, method.upper(), 
                    f"DINO_v3_{encoder_name}", all_metrics, interactive_path
                )
            
            # Store results - same structure as feature_visualization.py
            results[f"dino_v3_{encoder_name}_{method.lower()}"] = {
                'reduced_2d': reduced_2d,
                'reduced_3d': reduced_3d,
                'phases': phases,
                'scan_ids': scan_ids,
                'metrics': all_metrics,
                'dataframe': df
            }
            
            # Print metrics - same format as feature_visualization.py
            print(f"✅ {method.upper()} Analysis Results:")
            print(f"   2D Silhouette Score: {all_metrics.get('silhouette_2d', 0):.4f}")
            print(f"   3D Silhouette Score: {all_metrics.get('silhouette_3d', 0):.4f}")
            if 'total_variance_explained_2d' in all_metrics:
                print(f"   2D Variance Explained: {all_metrics['total_variance_explained_2d']:.4f}")
            if 'total_variance_explained_3d' in all_metrics:
                print(f"   3D Variance Explained: {all_metrics['total_variance_explained_3d']:.4f}")
            if 'cv_accuracy_mean' in all_metrics:
                print(f"   Classification Accuracy: {all_metrics['cv_accuracy_mean']:.4f} ± {all_metrics['cv_accuracy_std']:.4f}")
                
        except Exception as e:
            print(f"❌ Error with {method}: {e}")
            continue
    
    return results

def create_additional_dino_plots(features, phases, scan_ids, encoder_name, output_dir):
    """
    Create additional DINO-specific analysis plots with robust error handling
    """
    print(f"\n📈 Creating additional DINO v3 feature analysis plots...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Feature magnitude distribution by phase - WITH ROBUST BINNING
    ax1 = axes[0, 0]
    phase_names = ['Non-contrast', 'Arterial', 'Venous', 'Delayed']
    feature_norms = np.linalg.norm(features, axis=1)
    
    unique_phases = np.unique(phases)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_phases)))
    
    print(f"   Feature norms: min={feature_norms.min():.4f}, max={feature_norms.max():.4f}, range={feature_norms.max()-feature_norms.min():.4f}")
    
    # Adaptive binning based on data range
    data_range = feature_norms.max() - feature_norms.min()
    if data_range < 1e-10:  # Very small range
        print(f"   ⚠️  Feature norms have very small range ({data_range:.2e}), using single value plot")
        # Create a scatter plot instead of histogram
        for i, phase in enumerate(unique_phases):
            phase_mask = phases == phase
            if np.any(phase_mask):
                phase_name = phase_names[phase] if phase < len(phase_names) else f'Phase_{phase}'
                y_vals = np.ones(np.sum(phase_mask)) * i
                y_vals += np.random.normal(0, 0.1, size=len(y_vals))  # Add small jitter
                ax1.scatter(feature_norms[phase_mask], y_vals, alpha=0.6, label=phase_name, color=colors[i])
        
        ax1.set_xlabel('Feature Vector Magnitude')
        ax1.set_ylabel('Phase (jittered)')
        ax1.set_title(f'DINO v3 {encoder_name}: Feature Magnitudes (Constant Values)')
        ax1.set_yticks(range(len(unique_phases)))
        ax1.set_yticklabels([phase_names[p] if p < len(phase_names) else f'Phase_{p}' for p in unique_phases])
    else:
        # Use adaptive number of bins
        n_samples = len(feature_norms)
        max_bins = min(30, max(5, n_samples // 10))  # Adaptive bin count
        
        try:
            for i, phase in enumerate(unique_phases):
                phase_mask = phases == phase
                if np.any(phase_mask):
                    phase_name = phase_names[phase] if phase < len(phase_names) else f'Phase_{phase}'
                    phase_norms = feature_norms[phase_mask]
                    
                    # Determine appropriate number of bins for this phase
                    phase_range = phase_norms.max() - phase_norms.min()
                    if phase_range < 1e-10:
                        # This phase has constant values, use fewer bins
                        bins = 1
                    else:
                        bins = min(max_bins, max(1, len(phase_norms) // 5))
                    
                    ax1.hist(phase_norms, alpha=0.7, label=phase_name, 
                            bins=bins, color=colors[i], density=True)
            
            ax1.set_xlabel('Feature Vector Magnitude')
            ax1.set_ylabel('Density')
            ax1.set_title(f'DINO v3 {encoder_name}: Feature Magnitude Distribution by Phase')
        except Exception as e:
            print(f"   ⚠️  Histogram failed ({e}), using box plot instead")
            # Fallback to box plot
            ax1.clear()
            box_data = []
            box_labels = []
            for phase in unique_phases:
                phase_mask = phases == phase
                if np.any(phase_mask):
                    box_data.append(feature_norms[phase_mask])
                    phase_name = phase_names[phase] if phase < len(phase_names) else f'Phase_{phase}'
                    box_labels.append(phase_name)
            
            ax1.boxplot(box_data, labels=box_labels)
            ax1.set_xlabel('Phase')
            ax1.set_ylabel('Feature Vector Magnitude')
            ax1.set_title(f'DINO v3 {encoder_name}: Feature Magnitude Distribution by Phase (Box Plot)')
            plt.setp(ax1.get_xticklabels(), rotation=45)
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Feature dimension variance - WITH ERROR HANDLING
    ax2 = axes[0, 1]
    try:
        feature_vars = np.var(features, axis=0)
        print(f"   Feature variances: min={feature_vars.min():.4f}, max={feature_vars.max():.4f}")
        
        if feature_vars.max() - feature_vars.min() < 1e-10:
            # Very small variance range, show as constant line
            ax2.axhline(y=feature_vars.mean(), color='blue', linestyle='-', linewidth=2, label='Constant Variance')
            ax2.fill_between(range(len(feature_vars)), feature_vars.min(), feature_vars.max(), alpha=0.3)
            ax2.set_title(f'DINO v3 {encoder_name}: Variance Across Features (Nearly Constant)')
        else:
            ax2.plot(feature_vars, linewidth=1)
            ax2.set_title(f'DINO v3 {encoder_name}: Variance Across Feature Dimensions')
        
        ax2.set_xlabel('Feature Dimension')
        ax2.set_ylabel('Variance')
        ax2.grid(True, alpha=0.3)
        
    except Exception as e:
        print(f"   ⚠️  Feature variance plot failed: {e}")
        ax2.text(0.5, 0.5, f'Variance plot failed:\n{str(e)[:50]}...', 
                transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('Feature Variance Analysis (Failed)')
    
    # 3. Inter-phase distances - WITH ERROR HANDLING
    ax3 = axes[1, 0]
    try:
        phase_means = []
        valid_phases = []
        
        for phase in unique_phases:
            phase_mask = phases == phase
            if np.any(phase_mask) and np.sum(phase_mask) > 0:
                phase_mean = np.mean(features[phase_mask], axis=0)
                phase_means.append(phase_mean)
                valid_phases.append(phase)
        
        if len(phase_means) > 1:
            phase_means = np.array(phase_means)
            distance_matrix = np.zeros((len(phase_means), len(phase_means)))
            
            for i in range(len(phase_means)):
                for j in range(len(phase_means)):
                    distance_matrix[i, j] = np.linalg.norm(phase_means[i] - phase_means[j])
            
            print(f"   Inter-phase distances: min={distance_matrix[distance_matrix>0].min():.4f}, max={distance_matrix.max():.4f}")
            
            im = ax3.imshow(distance_matrix, cmap='viridis')
            ax3.set_title(f'DINO v3 {encoder_name}: Inter-Phase Centroid Distances')
            ax3.set_xlabel('Phase')
            ax3.set_ylabel('Phase')
            
            # Add labels
            phase_labels = [phase_names[i] if i < len(phase_names) else f'Phase_{i}' 
                           for i in valid_phases]
            ax3.set_xticks(range(len(phase_labels)))
            ax3.set_yticks(range(len(phase_labels)))
            ax3.set_xticklabels(phase_labels, rotation=45)
            ax3.set_yticklabels(phase_labels)
            
            # Add colorbar
            plt.colorbar(im, ax=ax3, label='Euclidean Distance')
            
            # Add text annotations for distances
            for i in range(len(phase_means)):
                for j in range(len(phase_means)):
                    text = ax3.text(j, i, f'{distance_matrix[i, j]:.3f}',
                                   ha="center", va="center", color="white" if distance_matrix[i, j] > distance_matrix.max()/2 else "black",
                                   fontsize=8)
        else:
            ax3.text(0.5, 0.5, f'Need at least 2 phases\nFound: {len(phase_means)} valid phases', 
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('Inter-Phase Distances (Insufficient Data)')
            
    except Exception as e:
        print(f"   ⚠️  Inter-phase distance plot failed: {e}")
        ax3.text(0.5, 0.5, f'Distance plot failed:\n{str(e)[:50]}...', 
                transform=ax3.transAxes, ha='center', va='center')
        ax3.set_title('Inter-Phase Distances (Failed)')
    
    # 4. Phase distribution - WITH ERROR HANDLING
    ax4 = axes[1, 1]
    try:
        # Safe phase counting
        phase_counts = []
        phase_labels_for_plot = []
        
        for phase in unique_phases:
            count = np.sum(phases == phase)
            phase_counts.append(count)
            phase_name = phase_names[phase] if phase < len(phase_names) else f'Phase_{phase}'
            phase_labels_for_plot.append(f'{phase_name}\n(n={count})')
        
        print(f"   Phase distribution: {dict(zip(unique_phases, phase_counts))}")
        
        bars = ax4.bar(range(len(phase_counts)), phase_counts, color=colors[:len(phase_counts)])
        ax4.set_xlabel('Phase')
        ax4.set_ylabel('Sample Count')
        ax4.set_title(f'DINO v3 {encoder_name}: Phase Distribution')
        ax4.set_xticks(range(len(phase_counts)))
        ax4.set_xticklabels(phase_labels_for_plot, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for bar, count in zip(bars, phase_counts):
            if count > 0:  # Only add labels for non-zero bars
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(phase_counts)*0.01, 
                        str(count), ha='center', va='bottom', fontweight='bold')
        
    except Exception as e:
        print(f"   ⚠️  Phase distribution plot failed: {e}")
        ax4.text(0.5, 0.5, f'Phase plot failed:\n{str(e)[:50]}...', 
                transform=ax4.transAxes, ha='center', va='center')
        ax4.set_title('Phase Distribution (Failed)')
    
    plt.tight_layout()
    
    # Save the additional analysis
    additional_path = os.path.join(output_dir, f'dino_v3_{encoder_name}_additional_analysis.png')
    try:
        plt.savefig(additional_path, dpi=300, bbox_inches='tight')
        print(f"📊 Additional analysis saved to: {additional_path}")
    except Exception as e:
        print(f"⚠️  Failed to save additional analysis: {e}")
    
    # Display the plot
    try:
        plt.show()
    except Exception as e:
        print(f"⚠️  Failed to display plot: {e}")
        plt.close()

# Alternative simplified version if the above still has issues
def create_simple_dino_plots(features, phases, scan_ids, encoder_name, output_dir):
    """
    Create simplified, robust plots that should always work
    """
    print(f"\n📈 Creating simplified DINO v3 feature analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    unique_phases = np.unique(phases)
    phase_names = ['Non-contrast', 'Arterial', 'Venous', 'Delayed']
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_phases)))
    
    # 1. Simple feature statistics
    ax1 = axes[0, 0]
    feature_means = np.mean(features, axis=0)
    ax1.plot(feature_means)
    ax1.set_title(f'DINO v3 {encoder_name}: Mean Feature Values')
    ax1.set_xlabel('Feature Dimension')
    ax1.set_ylabel('Mean Value')
    ax1.grid(True, alpha=0.3)
    
    # 2. Feature norms by phase (scatter plot)
    ax2 = axes[0, 1]
    feature_norms = np.linalg.norm(features, axis=1)
    for i, phase in enumerate(unique_phases):
        phase_mask = phases == phase
        if np.any(phase_mask):
            phase_name = phase_names[phase] if phase < len(phase_names) else f'Phase_{phase}'
            x_vals = np.full(np.sum(phase_mask), i)
            x_vals += np.random.normal(0, 0.1, size=len(x_vals))  # Add jitter
            ax2.scatter(x_vals, feature_norms[phase_mask], alpha=0.6, label=phase_name, color=colors[i])
    
    ax2.set_title(f'DINO v3 {encoder_name}: Feature Magnitudes by Phase')
    ax2.set_xlabel('Phase')
    ax2.set_ylabel('Feature Vector Magnitude')
    ax2.set_xticks(range(len(unique_phases)))
    ax2.set_xticklabels([phase_names[p] if p < len(phase_names) else f'Phase_{p}' for p in unique_phases])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Phase sample counts
    ax3 = axes[1, 0]
    phase_counts = [np.sum(phases == phase) for phase in unique_phases]
    bars = ax3.bar(range(len(phase_counts)), phase_counts, color=colors[:len(phase_counts)])
    ax3.set_title(f'DINO v3 {encoder_name}: Sample Count by Phase')
    ax3.set_xlabel('Phase')
    ax3.set_ylabel('Count')
    ax3.set_xticks(range(len(unique_phases)))
    ax3.set_xticklabels([phase_names[p] if p < len(phase_names) else f'Phase_{p}' for p in unique_phases])
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, phase_counts)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(phase_counts)*0.01, 
                str(count), ha='center', va='bottom')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Feature dimension statistics
    ax4 = axes[1, 1]
    feature_stds = np.std(features, axis=0)
    ax4.plot(feature_stds, label='Std Dev')
    ax4.plot(np.abs(feature_means), label='Abs Mean', alpha=0.7)
    ax4.set_title(f'DINO v3 {encoder_name}: Feature Statistics')
    ax4.set_xlabel('Feature Dimension')
    ax4.set_ylabel('Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    simple_path = os.path.join(output_dir, f'dino_v3_{encoder_name}_simple_analysis.png')
    plt.savefig(simple_path, dpi=300, bbox_inches='tight')
    print(f"📊 Simple analysis saved to: {simple_path}")
    plt.show()

    
def main():
    # First explain silhouette scores
    explain_silhouette_score()
    
    parser = argparse.ArgumentParser(description="DINO v3 Feature Extraction and Visualization")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="data", help="Path to data directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for feature extraction")
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[128, 128, 128], help="Input volume size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Model arguments
    parser.add_argument("--dino_v3_size", type=str, default="small", choices=["small", "base", "large"])
    parser.add_argument("--dino_v3_pretrained", action="store_true", default=True)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--max_slices", type=int, default=32)
    parser.add_argument("--slice_sampling", type=str, default="uniform", 
                       choices=["uniform", "adaptive", "random", "all"])
    
    # Visualization arguments
    parser.add_argument("--output_dir", type=str, default="dino_v3_visualizations", 
                       help="Output directory for visualizations")
    parser.add_argument("--methods", type=str, nargs='+', default=["pca", "tsne"],
                       choices=["pca", "tsne", "lda"], help="Dimensionality reduction methods")
    parser.add_argument("--create_interactive", action="store_true", 
                       help="Create interactive HTML plots")
    
    # Debug arguments
    parser.add_argument("--max_samples", type=int, default=None, 
                       help="Maximum number of samples to process (for debugging)")
    parser.add_argument("--use_cache", action="store_true", default=False, 
                       help="Use cached features")
    parser.add_argument("--cache_dir", type=str, default="cache", 
                       help="Cache directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    print(f"🚀 DINO v3 Feature Visualization Pipeline")
    print(f"📊 Model: DINO v3 {args.dino_v3_size}")
    print(f"🎯 Output: {args.output_dir}")
    print("=" * 60)
    
    try:
        # 1. Create DINO v3 encoder
        print(f"🔧 Creating DINO v3 {args.dino_v3_size} encoder...")
        encoder = DinoV3Encoder(
            latent_dim=args.latent_dim,
            model_size=args.dino_v3_size,
            pretrained=args.dino_v3_pretrained,
            max_slices=args.max_slices,
            slice_sampling=args.slice_sampling
        ).to(args.device)
        
        print(f"✅ DINO v3 encoder created successfully")
        
         # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        print("🔄 Preparing dataset...")
        
        labels_csv = os.path.join(args.data_path, "labels.csv")
        if not os.path.exists(labels_csv):
            print(f"❌ Error: labels.csv not found at {labels_csv}")
            return
        
        # Prepare dataset
        train_data_dicts, _ = prepare_dataset_from_folders(
            args.data_path, labels_csv, validation_split=0.2, skip_prep=True
        )
        print(f"✅ Using {len(train_data_dicts)} samples for visualization")
    
        # Create data loader
        img_size = tuple(args.spatial_size)
        data_loader = prepare_data(train_data_dicts, batch_size=args.batch_size, 
                                augmentation=False, spatial_size=img_size)
        
        # Initialize encoders
        print("🏗️ Initializing encoders...")
        # # 2. Prepare data
        # print(f"📁 Loading data from {args.data_path}...")
        # try:
        #     # Try the folder-based approach first
        #     train_loader, val_loader = prepare_dataset_from_folders( 
        #         args.data_path, labels_csv, validation_split=0.2, skip_prep=True
        #         args.data_path, 
        #         batch_size=args.batch_size, 
        #         spatial_size=tuple(args.spatial_size)
        #     )
        #     data_loader = train_loader
        #     print(f"✅ Data loaded using folder-based approach")
        # except:
        #     # Fallback to the other data loading approach
        #     train_loader, val_loader, _ = prepare_data(
        #         args.data_path, 
        #         batch_size=args.batch_size, 
        #         spatial_size=tuple(args.spatial_size)
        #     )
        #     data_loader = train_loader
        #     print(f"✅ Data loaded using fallback approach")
        
        # 3. Extract features using exact pattern
        features, phases, scan_ids = extract_dino_v3_features_exact(
            data_loader, encoder, args.device, args.dino_v3_size,
            use_cache=args.use_cache
        )
        
        print(f"📊 Feature extraction summary:")
        print(f"   Total samples: {len(features)}")
        print(f"   Feature dimension: {features.shape[1]}")
        print(f"   Unique phases: {np.unique(phases)}")
        print(f"   Unique scan IDs: {len(np.unique(scan_ids))}")
        
        # 4. Create comprehensive visualizations using exact functions
        results = create_dino_analysis_with_exact_functions(
            features, phases, scan_ids, 
            encoder_name=args.dino_v3_size,
            output_dir=args.output_dir,
            create_interactive=args.create_interactive
        )
        
        # 5. Create additional DINO-specific plots
        create_additional_dino_plots(
            features, phases, scan_ids, 
            encoder_name=args.dino_v3_size,
            output_dir=args.output_dir
        )
        
        # 6. Summary report - exact same pattern as feature_visualization.py
        print(f"\n🏆 DINO v3 {args.dino_v3_size} Feature Analysis Summary:")
        print("=" * 50)
        
        best_2d_score = -1
        best_3d_score = -1
        best_2d_method = None
        best_3d_method = None
        
        for key, result in results.items():
            encoder_name, method = key.split('_', 1)[-2:]  # Get last two parts
            metrics = result['metrics']
            sil_2d = metrics.get('silhouette_2d', 0)
            sil_3d = metrics.get('silhouette_3d', 0)
            
            print(f"\n{method.upper()} Results:")
            print(f"  2D Silhouette: {sil_2d:.4f}")
            print(f"  3D Silhouette: {sil_3d:.4f}")
            if 'total_variance_explained_2d' in metrics:
                print(f"  2D Variance Explained: {metrics['total_variance_explained_2d']:.4f}")
            if 'total_variance_explained_3d' in metrics:
                print(f"  3D Variance Explained: {metrics['total_variance_explained_3d']:.4f}")
            if 'cv_accuracy_mean' in metrics:
                print(f"  Classification Accuracy: {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
            
            if sil_2d > best_2d_score:
                best_2d_score = sil_2d
                best_2d_method = method
            
            if sil_3d > best_3d_score:
                best_3d_score = sil_3d
                best_3d_method = method
        
        # Only print best performance if we have results
        if best_2d_method and best_3d_method:
            print(f"\n🥇 Best Performance:")
            print(f"   2D: {best_2d_method.upper()} (Silhouette: {best_2d_score:.4f})")
            print(f"   3D: {best_3d_method.upper()} (Silhouette: {best_3d_score:.4f})")
        else:
            print(f"\n⚠️  No successful analyses completed")
        
        # Save summary report - same format as feature_visualization.py
        summary_path = os.path.join(args.output_dir, "dino_v3_analysis_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"DINO v3 {args.dino_v3_size} Feature Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write("SILHOUETTE SCORE EXPLANATION:\n")
            f.write(explain_silhouette_score())
            f.write(f"\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: DINO v3 {args.dino_v3_size}\n")
            f.write(f"Latent Dimension: {args.latent_dim}\n")
            f.write(f"Total Samples: {len(features)}\n")
            f.write(f"Unique Cases: {len(np.unique(scan_ids))}\n\n")
            
            f.write("RESULTS:\n")
            for key, result in results.items():
                method = key.split('_')[-1]
                metrics = result['metrics']
                f.write(f"\n{method.upper()}:\n")
                f.write(f"  2D Silhouette: {metrics.get('silhouette_2d', 0):.4f}\n")
                f.write(f"  3D Silhouette: {metrics.get('silhouette_3d', 0):.4f}\n")
                if 'total_variance_explained_2d' in metrics:
                    f.write(f"  2D Variance Explained: {metrics['total_variance_explained_2d']:.4f}\n")
                if 'total_variance_explained_3d' in metrics:
                    f.write(f"  3D Variance Explained: {metrics['total_variance_explained_3d']:.4f}\n")
            
            if best_2d_method and best_3d_method:
                f.write(f"\nBest Performance:\n")
                f.write(f"  2D: {best_2d_method.upper()} ({best_2d_score:.4f})\n")
                f.write(f"  3D: {best_3d_method.upper()} ({best_3d_score:.4f})\n")
        
        print(f"\n📁 All results saved to: {args.output_dir}")
        print(f"📝 Summary report: {summary_path}")
        print("🎉 DINO v3 feature analysis complete!")
        
    except Exception as e:
        print(f"❌ Error in DINO v3 feature analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()