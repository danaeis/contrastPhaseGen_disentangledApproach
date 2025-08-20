import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import argparse
from tqdm import tqdm

# Import project modules
from models import TimmViTEncoder
from medViT_encoder import create_medvit_encoder
from data import prepare_dataset_from_folders, prepare_data

def extract_features_from_encoder(encoder, data_loader, device='cuda', encoder_name='encoder'):
    """
    Extract features from encoder for all samples in the data loader
    
    Args:
        encoder: The encoder model (ViT or MedViT)
        data_loader: DataLoader containing the dataset
        device: Device to run inference on
        encoder_name: Name for logging purposes
    
    Returns:
        features: numpy array of extracted features
        phases: list of phase labels
        scan_ids: list of scan IDs
    """
    encoder.eval()
    encoder.to(device)
    
    all_features = []
    all_phases = []
    all_scan_ids = []
    
    print(f"Extracting features using {encoder_name}...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Processing {encoder_name}")):
            # Get input volumes and metadata
            input_volumes = batch['input_volume'].to(device)
            input_phases = batch['input_phase']
            scan_ids = batch['scan_id']
            
            # Extract features
            features = encoder(input_volumes)
            
            # Convert to numpy and store
            features_np = features.cpu().numpy()
            all_features.append(features_np)
            
            # Store phase labels and scan IDs
            for i in range(len(input_phases)):
                phase_label = input_phases[i].item() if torch.is_tensor(input_phases[i]) else input_phases[i]
                all_phases.append(phase_label)
                all_scan_ids.append(scan_ids[i])
    
    # Concatenate all features
    features = np.vstack(all_features)
    
    print(f"Extracted {features.shape[0]} feature vectors of dimension {features.shape[1]}")
    
    return features, all_phases, all_scan_ids

def create_phase_mapping():
    """
    Create mapping from phase numbers to phase names
    """
    return {
        0: 'Arterial',
        1: 'Venous', 
        2: 'Delayed',
        3: 'Non-contrast'
    }

def apply_dimensionality_reduction(features, method='tsne', n_components=2, random_state=42, n_iter=1000, **kwargs):
    """
    Apply dimensionality reduction to features
    
    Args:
        features: Feature matrix (n_samples, n_features)
        method: 'tsne' or 'pca'
        n_components: Number of components for reduction
        random_state: Random state for reproducibility
        n_iter: Number of iterations for t-SNE (ignored for PCA)
        **kwargs: Additional keyword arguments for the dimensionality reduction methods
    
    Returns:
        reduced_features: Reduced feature matrix
        reducer: The fitted reducer object
    """
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    if method.lower() == 'tsne':
        # Prepare t-SNE parameters
        tsne_params = {
            'n_components': n_components,
            'random_state': random_state,
            'perplexity': min(30, len(features) // 4),  # Adjust perplexity based on data size
            'max_iter': n_iter,
            'learning_rate': 'auto'
        }
        
        # Add any additional kwargs that are valid for TSNE
        tsne_params.update(kwargs)
        
        reducer = TSNE(**tsne_params)
        reduced_features = reducer.fit_transform(features_scaled)
        
    elif method.lower() == 'pca':
        # Prepare PCA parameters
        pca_params = {
            'n_components': n_components,
            'random_state': random_state
        }
        
        # Add any additional kwargs that are valid for PCA
        pca_params.update(kwargs)
        
        reducer = PCA(**pca_params)
        reduced_features = reducer.fit_transform(features_scaled)
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'pca'")
    
    return reduced_features, reducer

def plot_feature_visualization(reduced_features, phases, scan_ids, method_name, encoder_name, 
                             save_path=None, figsize=(12, 8)):
    """
    Create visualization plot for reduced features
    
    Args:
        reduced_features: 2D reduced features
        phases: Phase labels
        scan_ids: Scan IDs
        method_name: Name of dimensionality reduction method
        encoder_name: Name of encoder
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Create phase mapping
    phase_mapping = create_phase_mapping()
    
    # Convert phase numbers to names
    phase_names = [phase_mapping.get(p, f'Phase_{p}') for p in phases]
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'x': reduced_features[:, 0],
        'y': reduced_features[:, 1],
        'phase': phase_names,
        'phase_num': phases,
        'scan_id': scan_ids
    })
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Define colors for each phase
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    phase_order = ['Arterial', 'Venous', 'Delayed', 'Non-contrast']
    
    # Create scatter plot
    for i, phase in enumerate(phase_order):
        phase_data = df[df['phase'] == phase]
        if len(phase_data) > 0:
            plt.scatter(
                phase_data['x'], phase_data['y'],
                c=colors[i], label=phase,
                alpha=0.7, s=50, edgecolors='black', linewidth=0.5
            )
    
    plt.xlabel(f'{method_name} Component 1', fontsize=12)
    plt.ylabel(f'{method_name} Component 2', fontsize=12)
    plt.title(f'{method_name} Visualization of {encoder_name} Features\nPhase Clustering Analysis', 
              fontsize=14, fontweight='bold')
    plt.legend(title='Contrast Phase', title_fontsize=12, fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add some statistics to the plot
    unique_phases = df['phase'].unique()
    stats_text = f"Total samples: {len(df)}\nPhases: {len(unique_phases)}"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return df

def analyze_clustering_quality(reduced_features, phases, method_name, encoder_name):
    """
    Analyze the quality of phase clustering
    
    Args:
        reduced_features: 2D reduced features
        phases: Phase labels
        method_name: Name of dimensionality reduction method
        encoder_name: Name of encoder
    """
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.cluster import KMeans
    
    # Calculate silhouette score
    if len(np.unique(phases)) > 1:
        silhouette = silhouette_score(reduced_features, phases)
        print(f"\n{encoder_name} - {method_name} Clustering Analysis:")
        print(f"Silhouette Score: {silhouette:.4f}")
        
        # Try K-means clustering and compare with true phases
        n_clusters = len(np.unique(phases))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        predicted_clusters = kmeans.fit_predict(reduced_features)
        
        ari_score = adjusted_rand_score(phases, predicted_clusters)
        print(f"Adjusted Rand Index (vs K-means): {ari_score:.4f}")
        
        # Phase distribution
        phase_mapping = create_phase_mapping()
        phase_counts = pd.Series(phases).value_counts().sort_index()
        print("\nPhase Distribution:")
        for phase_num, count in phase_counts.items():
            phase_name = phase_mapping.get(phase_num, f'Phase_{phase_num}')
            print(f"  {phase_name}: {count} samples")
    
    return silhouette if len(np.unique(phases)) > 1 else 0

def main():
    parser = argparse.ArgumentParser(description="Feature Visualization for CT Contrast Phase Generation")
    parser.add_argument("--data_path", type=str, default="data", help="Path to data directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for feature extraction")
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[128, 128, 128], help="Input volume size D H W")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--output_dir", type=str, default="feature_visualizations", help="Output directory for plots")
    parser.add_argument("--latent_dim", type=int, default=256, help="Latent dimension size")
    
    # MedViT specific arguments
    parser.add_argument('--medvit_size', type=str, default='small', choices=['tiny', 'small', 'base'])
    parser.add_argument('--medvit_pretrained_path', type=str, default='pretrained_medvit_small.pth')
    parser.add_argument('--aggregation_method', type=str, default='lstm', choices=['lstm', 'attention', 'mean', 'max'])
    parser.add_argument('--max_slices', type=int, default=32)
    
    # Timm ViT specific arguments
    parser.add_argument('--timm_model_name', type=str, default='vit_small_patch16_224')
    parser.add_argument('--timm_pretrained', action='store_true', help='Use pretrained weights for Timm model')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🔄 Preparing dataset...")
    
    # Path to labels.csv
    labels_csv = os.path.join(args.data_path, "labels.csv")
    
    if not os.path.exists(labels_csv):
        print(f"❌ Error: labels.csv not found at {labels_csv}")
        return
    
    # Prepare dataset (use validation set for visualization to avoid data leakage)
    train_data_dicts, _ = prepare_dataset_from_folders(
        args.data_path,
        labels_csv,
        validation_split=0.2,
        skip_prep=True  # Use cached data if available
    )
    
    print(f"✅ Using {len(train_data_dicts)} validation samples for visualization")
    
    # Create data loader (no augmentation for feature extraction)
    img_size = tuple(args.spatial_size)
    data_loader = prepare_data(train_data_dicts, batch_size=args.batch_size, 
                              augmentation=False, spatial_size=img_size)
    
    # Initialize encoders
    print("🏗️  Initializing encoders...")
    
    # 1. MedViT Encoder
    medvit_config = {
        'model_size': args.medvit_size,
        'pretrained_path': args.medvit_pretrained_path if os.path.exists(args.medvit_pretrained_path) else None,
        'latent_dim': args.latent_dim,
        'aggregation_method': args.aggregation_method,
        'slice_sampling': 'uniform',
        'max_slices': args.max_slices
    }
    
    medvit_encoder = create_medvit_encoder(medvit_config)
    
    # 2. Timm ViT Encoder
    try:
        timm_encoder = TimmViTEncoder(
            latent_dim=args.latent_dim,
            model_name=args.timm_model_name,
            pretrained=args.timm_pretrained,
            max_slices=args.max_slices,
            slice_sampling='uniform'
        )
    except ImportError:
        print("⚠️  Warning: timm library not found. Skipping ViT encoder.")
        timm_encoder = None
    
    # Extract features from both encoders
    encoders = [
        (medvit_encoder, "MedViT"),
    ]
    
    if timm_encoder is not None:
        encoders.append((timm_encoder, "Timm-ViT"))
    
    # Store results for comparison
    results = {}
    
    for encoder, encoder_name in encoders:
        print(f"\n🔍 Processing {encoder_name} encoder...")
        
        # Extract features
        features, phases, scan_ids = extract_features_from_encoder(
            encoder, data_loader, args.device, encoder_name
        )
        
        # Apply dimensionality reduction methods
        methods = ['TSNE', 'PCA']
        
        for method in methods:
            print(f"\n📊 Applying {method} to {encoder_name} features...")
            
            # Apply dimensionality reduction
            reduced_features, reducer = apply_dimensionality_reduction(
                features, method=method, n_components=2
            )
            
            # Create visualization
            save_path = os.path.join(args.output_dir, f"{encoder_name}_{method}_visualization.png")
            df = plot_feature_visualization(
                reduced_features, phases, scan_ids, method, encoder_name, save_path
            )
            
            # Analyze clustering quality
            silhouette = analyze_clustering_quality(
                reduced_features, phases, method, encoder_name
            )
            
            # Store results
            results[f"{encoder_name}_{method}"] = {
                'features': reduced_features,
                'phases': phases,
                'scan_ids': scan_ids,
                'silhouette_score': silhouette,
                'dataframe': df
            }
    
    # Create comparison plot if we have both encoders
    if len(encoders) == 2:
        print("\n📈 Creating comparison plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Embedding Comparison: MedViT vs Timm-ViT', fontsize=16, fontweight='bold')
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        phase_order = ['Arterial', 'Venous', 'Delayed', 'Non-contrast']
        phase_mapping = create_phase_mapping()
        
        plot_configs = [
            ('MedViT_PCA', 'MedViT + PCA', 0, 0),
            ('MedViT_TSNE', 'MedViT + t-SNE', 0, 1),
            ('Timm-ViT_PCA', 'Timm-ViT + PCA', 1, 0),
            ('Timm-ViT_TSNE', 'Timm-ViT + t-SNE', 1, 1)
        ]
        
        for key, title, row, col in plot_configs:
            if key in results:
                ax = axes[row, col]
                df = results[key]['dataframe']
                
                for i, phase in enumerate(phase_order):
                    phase_data = df[df['phase'] == phase]
                    if len(phase_data) > 0:
                        ax.scatter(
                            phase_data['x'], phase_data['y'],
                            c=colors[i], label=phase,
                            alpha=0.7, s=30, edgecolors='black', linewidth=0.3
                        )
                
                ax.set_title(f"{title}\n(Silhouette: {results[key]['silhouette_score']:.3f})", fontsize=12)
                ax.grid(True, alpha=0.3)
                
                if row == 1:  # Bottom row
                    method_name = 'PCA' if 'PCA' in key else 't-SNE'
                    ax.set_xlabel(f'{method_name} Component 1')
                if col == 0:  # Left column
                    method_name = 'PCA' if 'PCA' in key else 't-SNE'
                    ax.set_ylabel(f'{method_name} Component 2')
        
        # Add legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), title='Contrast Phase')
        
        plt.tight_layout()
        comparison_path = os.path.join(args.output_dir, "encoder_comparison.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {comparison_path}")
        plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("FEATURE VISUALIZATION SUMMARY")
    print("="*60)
    
    for key, result in results.items():
        encoder_name, method = key.split('_', 1)
        print(f"{encoder_name} + {method}:")
        print(f"  Silhouette Score: {result['silhouette_score']:.4f}")
        print(f"  Samples: {len(result['phases'])}")
        print(f"  Unique Phases: {len(np.unique(result['phases']))}")
        print()
    
    print(f"All visualizations saved to: {args.output_dir}")
    print("🎉 Feature visualization complete!")

if __name__ == "__main__":
    main()