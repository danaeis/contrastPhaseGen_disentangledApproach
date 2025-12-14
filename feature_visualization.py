import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import os
import argparse
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from scipy import stats
import warnings
import pickle
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Import project modules
from models import TimmViTEncoder, DinoV3Encoder
from medViT_encoder import create_medvit_encoder
from data import prepare_dataset_from_folders, prepare_data

# ============================================
# DEBUG OPTIMIZATION FEATURES
# ============================================

def save_features_cache(features, phases, scan_ids, cache_path):
    """Save extracted features to avoid re-computation during debugging"""
    cache_data = {
        'features': features,
        'phases': phases,
        'scan_ids': scan_ids,
        'timestamp': datetime.now().isoformat()
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"💾 Features cached to: {cache_path}")

def load_features_cache(cache_path):
    """Load cached features if available"""
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        print(f"📂 Loaded cached features from: {cache_path}")
        print(f"   Cache timestamp: {cache_data['timestamp']}")
        return cache_data['features'], cache_data['phases'], cache_data['scan_ids']
    return None, None, None

def extract_features_from_encoder_optimized(encoder, data_loader, device='cuda', encoder_name='encoder', 
                                           use_cache=True, cache_dir='feature_cache', 
                                           max_samples_debug=None, debug_mode=False):
    """
    OPTIMIZED: Extract features with caching and debug options
    
    Args:
        max_samples_debug: Limit samples for debugging (None for all)
        debug_mode: If True, use smaller batches and show detailed progress
        use_cache: Use cached features if available
    """
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{encoder_name}_features.pkl")
    
    # Try to load from cache first
    if use_cache:
        features, phases, scan_ids = load_features_cache(cache_path)
        if features is not None:
            if max_samples_debug and len(features) > max_samples_debug:
                print(f"🔧 DEBUG MODE: Using first {max_samples_debug} samples")
                return features[:max_samples_debug], phases[:max_samples_debug], scan_ids[:max_samples_debug]
            return features, phases, scan_ids
    
    # Extract features
    encoder.eval()
    encoder.to(device)
    
    all_features = []
    all_phases = []
    all_scan_ids = []
    
    print(f"🔍 Extracting features using {encoder_name}...")
    if debug_mode:
        print(f"🐛 DEBUG MODE: Limited samples, detailed progress")
    
    total_processed = 0
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Processing {encoder_name}")):
            
            # DEBUG: Limit number of samples
            if max_samples_debug and total_processed >= max_samples_debug:
                print(f"🛑 DEBUG: Reached sample limit ({max_samples_debug})")
                break
            
            # Get input volumes and metadata
            input_volumes = batch['input_path'].to(device)
            input_phases = batch['input_phase']
            scan_ids = batch['scan_id']
            
            batch_start = time.time()
            
            # Extract features
            features = encoder(input_volumes)
            
            batch_time = time.time() - batch_start
            
            if debug_mode and batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}: {features.shape} features in {batch_time:.2f}s")
            
            # Convert to numpy and store
            features_np = features.cpu().numpy()
            all_features.append(features_np)
            
            # Store phase labels and scan IDs
            for i in range(len(input_phases)):
                phase_label = input_phases[i].item() if torch.is_tensor(input_phases[i]) else input_phases[i]
                all_phases.append(phase_label)
                all_scan_ids.append(scan_ids[i])
            
            total_processed += len(input_phases)
            
            # Progress update for debug mode
            if debug_mode and batch_idx % 10 == 0:
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                print(f"    Progress: {total_processed} samples, {rate:.1f} samples/sec")
    
    # Concatenate all features
    features = np.vstack(all_features)
    
    total_time = time.time() - start_time
    print(f"⏱️ Feature extraction completed in {total_time:.1f}s")
    print(f"📊 Extracted {features.shape[0]} feature vectors of dimension {features.shape[1]}")
    
    # Save to cache for future use
    if use_cache:
        save_features_cache(features, all_phases, all_scan_ids, cache_path)
    
    return features, all_phases, all_scan_ids

def create_phase_mapping(unique_classes):
    """Create phase mapping based on actual classes in data"""
    
    # Default phase names
    default_phases = {
        0: 'Non-contrast', 
        1: 'Arterial', 
        2: 'Venous', 
        3: 'Delayed', 
        4: 'Hepatobiliary'
    }
    
    # Create mapping only for classes that exist
    phase_mapping = {}
    for class_idx in unique_classes:
        phase_mapping[class_idx] = default_phases.get(class_idx, f'Phase_{class_idx}')
    
    return phase_mapping

def explain_silhouette_score():
    """
    Comprehensive explanation of the Silhouette Score
    """
    explanation = """
    🎯 SILHOUETTE SCORE EXPLAINED:
    
    The Silhouette Score measures how well-separated clusters are in your embedding space.
    It's crucial for validating whether your learned representations are meaningful.
    
    📈 How it works:
    For each sample point:
    1. Calculate 'a': Average distance to other points in the SAME cluster (phase)
    2. Calculate 'b': Average distance to points in the NEAREST other cluster
    3. Silhouette score for point i: s(i) = (b(i) - a(i)) / max(a(i), b(i))
    
    📊 Score interpretation:
    • +1.0: Perfect clustering (point is much closer to own cluster than others)
    • +0.5: Good clustering (point is reasonably well-placed)
    •  0.0: Point is on the border between two clusters
    • -0.5: Poor clustering (point might be in wrong cluster)
    • -1.0: Very poor clustering (point is closer to other clusters)
    
    🔍 What it tells us about your embeddings:
    • HIGH silhouette (>0.5): Your encoder learned meaningful phase representations
    • MEDIUM silhouette (0.2-0.5): Decent separation, some overlap between phases
    • LOW silhouette (<0.2): Phases are poorly separated - encoder may need improvement
    
    ⚠️ Important considerations:
    • Works best with well-defined, separated clusters
    • Can be misleading with complex cluster shapes
    • Compare across different dimensionality reduction methods
    • Higher dimensions may show different patterns than 2D/3D visualizations
    """
    
    print(explanation)
    return explanation

def apply_dimensionality_reduction_flexible(features, phases, method='pca', 
                                          n_components_2d=2, n_components_3d=3, 
                                          random_state=42, **kwargs):
    """
    FIXED: Apply dimensionality reduction with proper component handling
    
    Returns both 2D and 3D representations where possible
    """
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    metrics = {}
    
    if method.lower() == 'pca':
        # PCA can handle both 2D and 3D
        reducer_2d = PCA(n_components=n_components_2d, random_state=random_state)
        reducer_3d = PCA(n_components=n_components_3d, random_state=random_state)
        
        reduced_2d = reducer_2d.fit_transform(features_scaled)
        reduced_3d = reducer_3d.fit_transform(features_scaled)
        
        # Metrics
        metrics['explained_variance_ratio_2d'] = reducer_2d.explained_variance_ratio_
        metrics['explained_variance_ratio_3d'] = reducer_3d.explained_variance_ratio_
        metrics['total_variance_explained_2d'] = np.sum(reducer_2d.explained_variance_ratio_)
        metrics['total_variance_explained_3d'] = np.sum(reducer_3d.explained_variance_ratio_)
        
        return reduced_2d, reduced_3d, (reducer_2d, reducer_3d), metrics
        
    elif method.lower() == 'tsne':
        # FIXED: Use correct parameter name for sklearn
        tsne_params_2d = {
            'n_components': n_components_2d,
            'random_state': random_state,
            'perplexity': min(30, len(features) // 4),
            'max_iter': 1000,  # FIXED: was n_iter
            'learning_rate': 'auto',
            'init': 'pca'
        }
        
        tsne_params_3d = {
            'n_components': n_components_3d,
            'random_state': random_state,
            'perplexity': min(30, len(features) // 4),
            'max_iter': 1000,  # FIXED: was n_iter
            'learning_rate': 'auto',
            'init': 'pca'
        }
        
        # Add any additional kwargs
        tsne_params_2d.update(kwargs)
        tsne_params_3d.update(kwargs)
        
        # Apply preprocessing for high-dimensional data
        if features_scaled.shape[1] > 50:
            pca_preprocess = PCA(n_components=50, random_state=random_state)
            features_scaled = pca_preprocess.fit_transform(features_scaled)
        
        reducer_2d = TSNE(**tsne_params_2d)
        reducer_3d = TSNE(**tsne_params_3d)
        
        print(f"   Running t-SNE 2D...")
        reduced_2d = reducer_2d.fit_transform(features_scaled)
        
        print(f"   Running t-SNE 3D...")
        reduced_3d = reducer_3d.fit_transform(features_scaled)
        
        # t-SNE specific metrics
        metrics['kl_divergence_2d'] = reducer_2d.kl_divergence_ if hasattr(reducer_2d, 'kl_divergence_') else None
        metrics['kl_divergence_3d'] = reducer_3d.kl_divergence_ if hasattr(reducer_3d, 'kl_divergence_') else None
        
        return reduced_2d, reduced_3d, (reducer_2d, reducer_3d), metrics
        
    elif method.lower() == 'lda':
        # FIXED: LDA can only produce n_classes - 1 components maximum
        unique_phases = np.unique(phases)
        max_components = len(unique_phases) - 1
        
        if max_components < 1:
            raise ValueError("LDA requires at least 2 classes")
        
        # Adjust components based on available classes
        actual_2d_components = min(n_components_2d, max_components)
        actual_3d_components = min(n_components_3d, max_components)
        
        print(f"   LDA: {len(unique_phases)} classes → max {max_components} components")
        print(f"   Adjusting to: 2D={actual_2d_components}, 3D={actual_3d_components}")
        
        # For 2D
        reducer_2d = LinearDiscriminantAnalysis(n_components=actual_2d_components)
        reduced_2d = reducer_2d.fit_transform(features_scaled, phases)
        
        # For 3D (if possible)
        if actual_3d_components >= 3:
            reducer_3d = LinearDiscriminantAnalysis(n_components=actual_3d_components)
            reduced_3d = reducer_3d.fit_transform(features_scaled, phases)
        elif actual_3d_components == 2:
            # Pad with zeros for 3D visualization
            reduced_3d = np.zeros((reduced_2d.shape[0], 3))
            reduced_3d[:, :2] = reduced_2d
            reducer_3d = reducer_2d
            print(f"   ⚠️  LDA: Only 2 components available, padding 3rd dimension with zeros")
        else:
            # Only 1 component available
            reduced_3d = np.zeros((reduced_2d.shape[0], 3))
            reduced_3d[:, 0] = reduced_2d[:, 0] if reduced_2d.shape[1] > 0 else 0
            reducer_3d = reducer_2d
            print(f"   ⚠️  LDA: Only 1 component available, padding other dimensions with zeros")
        
        # LDA-specific metrics
        metrics['explained_variance_ratio_2d'] = reducer_2d.explained_variance_ratio_
        metrics['total_variance_explained_2d'] = np.sum(reducer_2d.explained_variance_ratio_)
        
        if actual_3d_components >= 2:
            if hasattr(reducer_3d, 'explained_variance_ratio_'):
                metrics['explained_variance_ratio_3d'] = reducer_3d.explained_variance_ratio_
                metrics['total_variance_explained_3d'] = np.sum(reducer_3d.explained_variance_ratio_)
        
        # Classification accuracy
        cv_scores = cross_val_score(reducer_2d, features_scaled, phases, cv=min(5, len(unique_phases)), scoring='accuracy')
        metrics['cv_accuracy_mean'] = np.mean(cv_scores)
        metrics['cv_accuracy_std'] = np.std(cv_scores)
        
        return reduced_2d, reduced_3d, (reducer_2d, reducer_3d), metrics
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca', 'tsne', or 'lda'")

def compute_comprehensive_metrics(reduced_2d, reduced_3d, phases):
    """Compute metrics for both 2D and 3D representations"""
    metrics = {}
    
    if len(np.unique(phases)) > 1:
        # Silhouette scores
        metrics['silhouette_2d'] = silhouette_score(reduced_2d, phases)
        if reduced_3d is not None and reduced_3d.shape[1] >= 2:
            # Only compute if we have actual variation in 3D
            if np.var(reduced_3d[:, 2]) > 1e-10:  # Check if 3rd dimension has variation
                metrics['silhouette_3d'] = silhouette_score(reduced_3d, phases)
            else:
                metrics['silhouette_3d'] = metrics['silhouette_2d']  # Same as 2D
        
        # Fisher's discriminant ratios
        metrics['fisher_ratio_2d'] = compute_fisher_discriminant_ratio(reduced_2d, phases)
        if reduced_3d is not None:
            metrics['fisher_ratio_3d'] = compute_fisher_discriminant_ratio(reduced_3d, phases)
    
    return metrics

def compute_fisher_discriminant_ratio(reduced_features, phases):
    """Compute Fisher's discriminant ratio"""
    between_var = compute_between_class_variance(reduced_features, phases)
    within_var = compute_within_class_variance(reduced_features, phases)
    return between_var / (within_var + 1e-8)

def compute_between_class_variance(features, labels):
    """Compute between-class variance"""
    overall_mean = np.mean(features, axis=0)
    unique_labels = np.unique(labels)
    
    between_class_var = 0
    total_samples = len(features)
    
    for label in unique_labels:
        class_mask = (labels == label)
        class_samples = np.sum(class_mask)
        class_mean = np.mean(features[class_mask], axis=0)
        
        class_weight = class_samples / total_samples
        between_class_var += class_weight * np.sum((class_mean - overall_mean) ** 2)
    
    return between_class_var

def compute_within_class_variance(features, labels):
    """Compute within-class variance"""
    unique_labels = np.unique(labels)
    within_class_var = 0
    
    for label in unique_labels:
        class_mask = (labels == label)
        class_features = features[class_mask]
        if len(class_features) > 0:
            class_mean = np.mean(class_features, axis=0)
            within_class_var += np.sum((class_features - class_mean) ** 2)
    
    within_class_var /= len(features)
    return within_class_var

def plot_combined_2d_3d_visualization(reduced_2d, reduced_3d, phases, scan_ids, 
                                     method_name, encoder_name, metrics, 
                                     save_path=None, figsize=(20, 12)):
    """
    Create comprehensive side-by-side 2D and 3D visualization with case IDs
    """
    phase_mapping = create_phase_mapping()
    phase_names = [phase_mapping.get(p, f'Phase_{p}') for p in phases]
    
    # Create DataFrame
    df = pd.DataFrame({
        'x_2d': reduced_2d[:, 0],
        'y_2d': reduced_2d[:, 1],
        'x_3d': reduced_3d[:, 0],
        'y_3d': reduced_3d[:, 1],
        'z_3d': reduced_3d[:, 2],
        'phase': phase_names,
        'case_id': scan_ids,
        'phase_num': phases
    })
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)
    
    # NEW: Create color mapping for scan_ids and marker mapping for phases
    unique_scan_ids = sorted(list(set(scan_ids)))
    unique_phases = sorted(list(set(phases)))
    
    # Generate distinct colors for each scan_id
    if len(unique_scan_ids) <= 10:
        # Use distinct colors for small number of cases
        case_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A', 
                      '#98FB98', '#DDA0DD', '#F0E68C', '#FF69B4', '#00CED1']
    else:
        # Use colormap for many cases
        import matplotlib.cm as cm
        case_colors = cm.tab20(np.linspace(0, 1, len(unique_scan_ids)))
    
    case_color_map = {case_id: case_colors[i % len(case_colors)] for i, case_id in enumerate(unique_scan_ids)}
    
    # Define markers for phases
    phase_markers = {0: 'o', 1: 's', 2: '^', 3: 'D'}  # Arterial=circle, Venous=square, Delayed=triangle, Non-contrast=diamond
    phase_order = ['Non-contrast', 'Arterial', 'Venous', 'Delayed' ]
    phase_marker_labels = {0: 'o (Arterial)', 1: 's (Venous)', 2: '^ (Delayed)', 3: 'D (Non-contrast)'}
    

    # # Colors and markers
    # colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    # markers = ['o', 's', '^', 'D']
    # phase_order = ['Arterial', 'Venous', 'Delayed', 'Non-contrast']
    
    # 2D Plot
    ax_2d = fig.add_subplot(gs[0, :2])

    # Plot each point individually with case-specific color and phase-specific marker
    for idx, row in df.iterrows():
        case_color = case_color_map[row['case_id']]
        phase_marker = phase_markers.get(row['phase_num'], 'o')
        
        ax_2d.scatter(
            row['x_2d'], row['y_2d'],
            c=[case_color], marker=phase_marker, 
            alpha=0.7, s=60, edgecolors='black', linewidth=0.5
        )
    
    ax_2d.set_xlabel(f'{method_name} Component 1')
    ax_2d.set_ylabel(f'{method_name} Component 2')
    ax_2d.set_title(f'2D {method_name} - {encoder_name}\nColors=Cases, Shapes=Phases, Silhouette: {metrics.get("silhouette_2d", 0):.3f}')
    ax_2d.grid(True, alpha=0.3)
    
    # Create custom legend for phases (shapes)
    phase_legend_elements = []
    for phase_num in unique_phases:
        phase_name = phase_mapping.get(phase_num, f'Phase_{phase_num}')
        marker = phase_markers.get(phase_num, 'o')
        phase_legend_elements.append(plt.scatter([], [], marker=marker, c='gray', s=60, label=phase_name, edgecolors='black'))
    
    ax_2d.legend(handles=phase_legend_elements, title='Phases (Shapes)', loc='upper right')
    
    # for i, phase in enumerate(phase_order):
    #     phase_data = df[df['phase'] == phase]
    #     if len(phase_data) > 0:
    #         scatter = ax_2d.scatter(
    #             phase_data['x_2d'], phase_data['y_2d'],
    #             c=colors[i], marker=markers[i], label=phase,
    #             alpha=0.7, s=60, edgecolors='black', linewidth=0.5
    #         )
    
    # ax_2d.set_xlabel(f'{method_name} Component 1')
    # ax_2d.set_ylabel(f'{method_name} Component 2')
    # ax_2d.set_title(f'2D {method_name} - {encoder_name}\nSilhouette: {metrics.get("silhouette_2d", 0):.3f}')
    # ax_2d.legend()
    # ax_2d.grid(True, alpha=0.3)
    
    # 3D Plot
    ax_3d = fig.add_subplot(gs[0, 2:], projection='3d')
    # Plot each point individually with case-specific color and phase-specific marker
    for idx, row in df.iterrows():
        case_color = case_color_map[row['case_id']]
        phase_marker = phase_markers.get(row['phase_num'], 'o')
        
        ax_3d.scatter(
            row['x_3d'], row['y_3d'], row['z_3d'],
            c=[case_color], marker=phase_marker,
            alpha=0.7, s=60, edgecolors='black', linewidth=0.5
        )
    
    ax_3d.set_xlabel(f'{method_name} Component 1')
    ax_3d.set_ylabel(f'{method_name} Component 2')
    ax_3d.set_zlabel(f'{method_name} Component 3')
    ax_3d.set_title(f'3D {method_name} - {encoder_name}\nColors=Cases, Shapes=Phases, Silhouette: {metrics.get("silhouette_3d", 0):.3f}')
    
    # Create custom legend for phases (shapes) for 3D plot
    phase_legend_elements_3d = []
    for phase_num in unique_phases:
        phase_name = phase_mapping.get(phase_num, f'Phase_{phase_num}')
        marker = phase_markers.get(phase_num, 'o')
        phase_legend_elements_3d.append(plt.Line2D([0], [0], marker=marker, color='gray', linestyle='None', 
                                                  markersize=8, label=phase_name, markeredgecolor='black'))
    
    ax_3d.legend(handles=phase_legend_elements_3d, title='Phases (Shapes)', loc='upper right')
    
    # for i, phase in enumerate(phase_order):
    #     phase_data = df[df['phase'] == phase]
    #     if len(phase_data) > 0:
    #         ax_3d.scatter(
    #             phase_data['x_3d'], phase_data['y_3d'], phase_data['z_3d'],
    #             c=colors[i], marker=markers[i], label=phase,
    #             alpha=0.7, s=60, edgecolors='black', linewidth=0.5
    #         )
    
    # ax_3d.set_xlabel(f'{method_name} Component 1')
    # ax_3d.set_ylabel(f'{method_name} Component 2')
    # ax_3d.set_zlabel(f'{method_name} Component 3')
    # ax_3d.set_title(f'3D {method_name} - {encoder_name}\nSilhouette: {metrics.get("silhouette_3d", 0):.3f}')
    # ax_3d.legend()
    
    # Case-Phase Distribution
    ax_case = fig.add_subplot(gs[1, :2])
    
    # Create case-phase heatmap
    case_phase_pivot = df.pivot_table(
        index='case_id', columns='phase', 
        values='x_2d', aggfunc='count', fill_value=0
    )
    
    if len(case_phase_pivot) > 20:  # Limit display for readability
        case_phase_pivot = case_phase_pivot.head(20)
    
    sns.heatmap(case_phase_pivot, annot=True, fmt='d', cmap='Blues', ax=ax_case)
    ax_case.set_title('Case-Phase Distribution (Top 20 Cases)')
    ax_case.set_xlabel('Contrast Phase')
    ax_case.set_ylabel('Case ID')
    
    # NEW: Add case color legend
    ax_case_legend = fig.add_subplot(gs[1, 2])
    ax_case_legend.axis('off')
    ax_case_legend.set_title('Case Colors', fontweight='bold', fontsize=12)
    
    # Show case color mapping (limit to prevent clutter)
    display_cases = unique_scan_ids[:15] if len(unique_scan_ids) > 15 else unique_scan_ids
    legend_text = "Case ID → Color:\n\n"
    
    for i, case_id in enumerate(display_cases):
        color = case_color_map[case_id]
        legend_text += f"● {case_id}\n"
        
        # Add colored circle for each case
        circle_y = 0.9 - (i * 0.05)
        if circle_y > 0.1:  # Only show if there's space
            ax_case_legend.scatter(0.05, circle_y, c=[color], s=100, alpha=0.8, edgecolors='black')
            ax_case_legend.text(0.15, circle_y, str(case_id), fontsize=8, va='center')
    
    if len(unique_scan_ids) > 15:
        ax_case_legend.text(0.05, 0.1, f"... and {len(unique_scan_ids) - 15} more cases", 
                           fontsize=8, style='italic')
    
    ax_case_legend.set_xlim(0, 1)
    ax_case_legend.set_ylim(0, 1)

    # Metrics Comparison
    ax_metrics = fig.add_subplot(gs[1, 2:])
    
    metric_names = []
    metric_values_2d = []
    metric_values_3d = []
    
    # Collect comparable metrics
    if 'silhouette_2d' in metrics and 'silhouette_3d' in metrics:
        metric_names.append('Silhouette')
        metric_values_2d.append(metrics['silhouette_2d'])
        metric_values_3d.append(metrics['silhouette_3d'])
    
    if 'fisher_ratio_2d' in metrics and 'fisher_ratio_3d' in metrics:
        metric_names.append('Fisher Ratio')
        metric_values_2d.append(metrics['fisher_ratio_2d'])
        metric_values_3d.append(metrics['fisher_ratio_3d'])
    
    if 'total_variance_explained_2d' in metrics and 'total_variance_explained_3d' in metrics:
        metric_names.append('Variance Explained')
        metric_values_2d.append(metrics['total_variance_explained_2d'])
        metric_values_3d.append(metrics['total_variance_explained_3d'])
    
    if metric_names:
        x = np.arange(len(metric_names))
        width = 0.35
        
        ax_metrics.bar(x - width/2, metric_values_2d, width, label='2D', alpha=0.7)
        ax_metrics.bar(x + width/2, metric_values_3d, width, label='3D', alpha=0.7)
        
        ax_metrics.set_xlabel('Metrics')
        ax_metrics.set_ylabel('Value')
        ax_metrics.set_title('2D vs 3D Metric Comparison')
        ax_metrics.set_xticks(x)
        ax_metrics.set_xticklabels(metric_names)
        ax_metrics.legend()
        ax_metrics.grid(True, alpha=0.3)
    else:
        ax_metrics.text(0.5, 0.5, 'No comparable\nmetrics available', 
                       ha='center', va='center', transform=ax_metrics.transAxes)
        ax_metrics.set_title('Metric Comparison')
    
    # Dataset Statistics
    ax_stats = fig.add_subplot(gs[2, :2])
    ax_stats.axis('off')
    
    unique_cases = len(df['case_id'].unique())
    unique_phases = len(df['phase'].unique())
    total_samples = len(df)
    
    # Case statistics
    case_counts = df['case_id'].value_counts()
    multi_phase_cases = df.groupby('case_id')['phase'].nunique()
    cases_with_multiple_phases = len(multi_phase_cases[multi_phase_cases > 1])
    
    stats_text = f"""Dataset Statistics:
    
Total Samples: {total_samples}
Unique Cases: {unique_cases}  
Unique Phases: {unique_phases}
Multi-phase Cases: {cases_with_multiple_phases}

Top Cases by Sample Count:"""
    
    for case_id, count in case_counts.head(5).items():
        stats_text += f"\n  {case_id}: {count} samples"
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Method-Specific Info
    ax_method = fig.add_subplot(gs[2, 2:])
    ax_method.axis('off')
    
    method_text = f"{method_name} Analysis Summary:\n\n"
    
    # Add method-specific metrics
    if method_name.upper() == 'LDA' and 'cv_accuracy_mean' in metrics:
        method_text += f"Classification Accuracy: {metrics['cv_accuracy_mean']:.3f} ± {metrics['cv_accuracy_std']:.3f}\n"
    
    if 'total_variance_explained_2d' in metrics:
        method_text += f"2D Variance Explained: {metrics['total_variance_explained_2d']:.3f}\n"
    
    if 'total_variance_explained_3d' in metrics:
        method_text += f"3D Variance Explained: {metrics['total_variance_explained_3d']:.3f}\n"
    
    # Add case ID examples with phases
    method_text += f"\nCase-Phase Examples:\n"
    sample_cases = df.groupby('case_id').first().head(3)
    for case_id, row in sample_cases.iterrows():
        method_text += f"  {case_id}: {row['phase']}\n"
    
    ax_method.text(0.05, 0.95, method_text, transform=ax_method.transAxes,
                  verticalalignment='top', fontsize=10,
                  bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle(f'Comprehensive 2D/3D Analysis: {encoder_name} + {method_name}', 
                 fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined 2D/3D plot saved to: {save_path}")
    
    plt.show()
    
    return df

# def create_interactive_combined_plot(reduced_2d, reduced_3d, phases, scan_ids, 
                                    # method_name, encoder_name, metrics, save_path=None):
     
     
def create_interactive_combined_plot(reduced_2d, reduced_3d, phases, scan_ids, 
                                    method_name, encoder_name, metrics, save_path=None):
    """
    Create interactive combined 2D/3D plot with case ID information
    UPDATED: Colors represent scan_ids, symbols represent phases
    """
    phase_mapping = create_phase_mapping()
    phase_names = [phase_mapping.get(p, f'Phase_{p}') for p in phases]
    
    # Create DataFrame with additional info for plotting
    df = pd.DataFrame({
        'x_2d': reduced_2d[:, 0],
        'y_2d': reduced_2d[:, 1],
        'x_3d': reduced_3d[:, 0],
        'y_3d': reduced_3d[:, 1],
        'z_3d': reduced_3d[:, 2],
        'phase': phase_names,
        'case_id': scan_ids,
        'phase_num': phases
    })
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f'2D {method_name} (Colors=Cases, Symbols=Phases)', 
                       f'3D {method_name} (Colors=Cases, Symbols=Phases)'],
        specs=[[{"type": "scatter"}, {"type": "scatter3d"}]]
    )
    
    # NEW: Create distinct colors for each case and symbols for each phase
    unique_scan_ids = sorted(list(set(scan_ids)))
    unique_phases = sorted(list(set(phases)))
    
    # Generate distinct colors for cases
    if len(unique_scan_ids) <= 10:
        case_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A', 
                      '#98FB98', '#DDA0DD', '#F0E68C', '#FF69B4', '#00CED1']
    else:
        import plotly.colors as pc
        case_colors = pc.qualitative.Set3 + pc.qualitative.Pastel + pc.qualitative.Set1
    
    case_color_map = {case_id: case_colors[i % len(case_colors)] for i, case_id in enumerate(unique_scan_ids)}
    
    # Define symbols for phases (Plotly symbols)
    phase_symbols = {
    0: 'circle',   # Non-contrast (was triangle, now cross)
    1: 'square',   # Arterial
    2: 'diamond',  # Venous  
    3: 'cross'     # Delayed
}
    phase_symbol_map = {phase_mapping.get(p, f'Phase_{p}'): phase_symbols.get(p, 'circle') for p in unique_phases}
    
    # Group by case for 2D plot
    for case_id in unique_scan_ids:
        case_data = df[df['case_id'] == case_id]
        if len(case_data) > 0:
            case_color = case_color_map[case_id]
            
            # For each phase in this case
            for phase in case_data['phase'].unique():
                phase_case_data = case_data[case_data['phase'] == phase]
                symbol = phase_symbol_map.get(phase, 'circle')
                
                fig.add_trace(
                    go.Scatter(
                        x=phase_case_data['x_2d'],
                        y=phase_case_data['y_2d'],
                        mode='markers',
                        name=f"{case_id}",
                        legendgroup=f"case_{case_id}",
                        marker=dict(
                            color=case_color, 
                            size=8,
                            symbol=symbol,
                            line=dict(color='black', width=1)
                        ),
                        text=[f"Case: {row['case_id']}<br>Phase: {row['phase']}" for _, row in phase_case_data.iterrows()],
                        hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>',
                        showlegend=(phase == case_data['phase'].iloc[0])  # Show legend only once per case
                    ),
                    row=1, col=1
                )
    
    # Group by case for 3D plot  
    for case_id in unique_scan_ids:
        case_data = df[df['case_id'] == case_id]
        if len(case_data) > 0:
            case_color = case_color_map[case_id]
            
            # For each phase in this case
            for phase in case_data['phase'].unique():
                phase_case_data = case_data[case_data['phase'] == phase]
                symbol = phase_symbol_map.get(phase, 'circle')
                
                fig.add_trace(
                    go.Scatter3d(
                        x=phase_case_data['x_3d'],
                        y=phase_case_data['y_3d'],
                        z=phase_case_data['z_3d'],
                        mode='markers',
                        name=f"{case_id} (3D)",
                        legendgroup=f"case_{case_id}_3d",
                        marker=dict(
                            color=case_color,
                            size=5,
                            symbol=symbol,
                            line=dict(color='black', width=1)
                        ),
                        text=[f"Case: {row['case_id']}<br>Phase: {row['phase']}" for _, row in phase_case_data.iterrows()],
                        hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>',
                        showlegend=False  # Don't duplicate legends for 3D
                    ),
                    row=1, col=2
                )
    
    # Update layout
    fig.update_layout(
        title=f'Interactive 2D/3D Visualization: {encoder_name} + {method_name}<br><sub>Colors = Case IDs, Symbols = Phases (● Non-contrast, ■ Arterial, ▲ Venous, ◆ Delayed)</sub>',
        height=700,
        showlegend=True,
        legend=dict(
            title="Case IDs<br>(Colors)",
            orientation="v",
            x=1.05,
            y=1
        )
    )
    
    # Add annotations with metrics and legend explanation
    metrics_text = f"2D Silhouette: {metrics.get('silhouette_2d', 0):.3f}<br>3D Silhouette: {metrics.get('silhouette_3d', 0):.3f}"
    
    fig.add_annotation(
        text=metrics_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98, xanchor="left", yanchor="top",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1
    )
    
    # Add phase symbol legend
    phase_legend_text = "Phase Symbols:<br>" "(● Non-contrast, ■ Arterial, ▲ Venous, ◆ Delayed)"
    
    fig.add_annotation(
        text=phase_legend_text,
        xref="paper", yref="paper",
        x=0.02, y=0.02, xanchor="left", yanchor="bottom",
        showarrow=False,
        bgcolor="rgba(255,255,200,0.8)",
        bordercolor="gray",
        borderwidth=1
    )
    
    if save_path:
        pyo.plot(fig, filename=save_path, auto_open=False)
        print(f"Interactive combined plot saved to: {save_path}")
    else:
        fig.show()
    
    return fig

def main():
    # First, explain the silhouette score
    explain_silhouette_score()
    
    parser = argparse.ArgumentParser(description="Debug-Optimized Feature Visualization with 2D/3D Analysis")
    parser.add_argument("--data_path", type=str, default="data", help="Path to data directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for feature extraction")
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[128, 128, 128], help="Input volume size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--output_dir", type=str, default="debug_feature_visualizations", help="Output directory")
    parser.add_argument("--latent_dim", type=int, default=256, help="Latent dimension size")
    parser.add_argument("--create_interactive", action="store_true", help="Create interactive plots")
    
    # DEBUG OPTIONS
    parser.add_argument("--debug_mode", action="store_true", help="Enable debug mode with progress details")
    parser.add_argument("--use_cache", action="store_true", default=True, help="Use cached features")
    parser.add_argument("--max_samples_debug", type=int, default=None, help="Limit samples for debugging (e.g., 100)")
    parser.add_argument("--skip_medvit", action="store_true", help="Skip MedViT encoder for faster debugging")
    parser.add_argument("--quick_tsne", action="store_true", help="Use faster t-SNE settings")
    
    # Model arguments
    parser.add_argument('--medvit_size', type=str, default='small', choices=['tiny', 'small', 'base'])
    parser.add_argument('--medvit_pretrained_path', type=str, default='pretrained_medvit_small.pth')
    parser.add_argument('--aggregation_method', type=str, default='lstm', choices=['lstm', 'attention', 'mean', 'max'])
    parser.add_argument('--max_slices', type=int, default=32)
    parser.add_argument('--timm_model_name', type=str, default='vit_small_patch16_224')
    parser.add_argument('--timm_pretrained', action='store_true', help='Use pretrained weights for Timm model')
    
    args = parser.parse_args()
    
    # DEBUG MODE SETUP
    if args.debug_mode:
        print("🐛 DEBUG MODE ENABLED")
        print(f"   Max samples: {args.max_samples_debug or 'All'}")
        print(f"   Use cache: {args.use_cache}")
        print(f"   Skip MedViT: {args.skip_medvit}")
    
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
    
    # DEBUG: Limit dataset size
    if args.max_samples_debug:
        train_data_dicts = train_data_dicts[:args.max_samples_debug * 2]  # Account for batch processing
        print(f"🐛 DEBUG: Limited dataset to ~{len(train_data_dicts)} entries")
    
    print(f"✅ Using {len(train_data_dicts)} samples for visualization")
    
    # Create data loader
    img_size = tuple(args.spatial_size)
    data_loader = prepare_data(train_data_dicts, batch_size=args.batch_size, 
                              augmentation=False, spatial_size=img_size)
    
    # Initialize encoders
    print("🏗️ Initializing encoders...")
    
    encoders = []
    
    # Add MedViT only if not skipped
    if not args.skip_medvit:
        medvit_config = {
            'model_size': args.medvit_size,
            'pretrained_path': args.medvit_pretrained_path if os.path.exists(args.medvit_pretrained_path) else None,
            'latent_dim': args.latent_dim,
            'aggregation_method': args.aggregation_method,
            'slice_sampling': 'uniform',
            'max_slices': args.max_slices
        }
        medvit_encoder = create_medvit_encoder(medvit_config)
        encoders.append((medvit_encoder, "MedViT"))
    else:
        print("⏩ Skipping MedViT encoder for debug speed")
    
    # Add Timm ViT
    try:
        timm_encoder = TimmViTEncoder(
            latent_dim=args.latent_dim,
            model_name=args.timm_model_name,
            pretrained=args.timm_pretrained,
            max_slices=args.max_slices,
            slice_sampling='uniform'
        )
        encoders.append((timm_encoder, "Timm-ViT"))
    except ImportError:
        print("⚠️ Warning: timm library not found. Skipping ViT encoder.")
    
    if not encoders:
        print("❌ No encoders available!")
        return
    
    # Store results
    results = {}
    
    for encoder, encoder_name in encoders:
        print(f"\n🔍 Processing {encoder_name} encoder...")
        
        # Extract features with optimization
        features, phases, scan_ids = extract_features_from_encoder_optimized(
            encoder, data_loader, args.device, encoder_name,
            use_cache=args.use_cache,
            max_samples_debug=args.max_samples_debug,
            debug_mode=args.debug_mode
        )
        
        # Apply dimensionality reduction methods
        methods = ['PCA', 'TSNE', 'LDA']
        
        for method in methods:
            print(f"\n📊 Applying 2D/3D {method} to {encoder_name} features...")
            
            try:
                # Special handling for quick t-SNE in debug mode
                kwargs = {}
                if method.upper() == 'TSNE' and args.quick_tsne:
                    kwargs = {'max_iter': 500, 'perplexity': min(15, len(features) // 8)}
                    print(f"   🚀 Quick t-SNE mode: reduced iterations and perplexity")
                
                # Apply dimensionality reduction
                reduced_2d, reduced_3d, reducers, dr_metrics = apply_dimensionality_reduction_flexible(
                    features, phases, method=method, **kwargs
                )
                
                # Compute comprehensive metrics
                comp_metrics = compute_comprehensive_metrics(reduced_2d, reduced_3d, phases)
                
                # Combine all metrics
                all_metrics = {**dr_metrics, **comp_metrics}
                
                # Create combined visualization
                save_path = os.path.join(args.output_dir, f"{encoder_name}_{method}_combined_2d_3d.png")
                df = plot_combined_2d_3d_visualization(
                    reduced_2d, reduced_3d, phases, scan_ids, method, encoder_name, 
                    all_metrics, save_path
                )
                
                # Create interactive plot if requested
                if args.create_interactive:
                    interactive_path = os.path.join(args.output_dir, f"{encoder_name}_{method}_interactive_combined.html")
                    create_interactive_combined_plot(
                        reduced_2d, reduced_3d, phases, scan_ids, method, encoder_name, 
                        all_metrics, interactive_path
                    )
                
                # Store results
                results[f"{encoder_name}_{method}"] = {
                    'reduced_2d': reduced_2d,
                    'reduced_3d': reduced_3d,
                    'phases': phases,
                    'scan_ids': scan_ids,
                    'metrics': all_metrics,
                    'dataframe': df
                }
                
                # Print results
                print(f"✅ {method} Analysis Results:")
                print(f"   2D Silhouette Score: {all_metrics.get('silhouette_2d', 0):.4f}")
                print(f"   3D Silhouette Score: {all_metrics.get('silhouette_3d', 0):.4f}")
                if 'total_variance_explained_2d' in all_metrics:
                    print(f"   2D Variance Explained: {all_metrics['total_variance_explained_2d']:.4f}")
                if 'total_variance_explained_3d' in all_metrics:
                    print(f"   3D Variance Explained: {all_metrics['total_variance_explained_3d']:.4f}")
                if 'cv_accuracy_mean' in all_metrics:
                    print(f"   LDA Classification Accuracy: {all_metrics['cv_accuracy_mean']:.4f}")
                
            except Exception as e:
                print(f"❌ Error processing {method} for {encoder_name}: {e}")
                import traceback
                if args.debug_mode:
                    print(f"   Debug traceback: {traceback.format_exc()}")
                continue
    
    # Final summary
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE 2D/3D FEATURE VISUALIZATION SUMMARY")
    print("="*80)
    
    print("\n📊 SILHOUETTE SCORE SUMMARY:")
    print("Silhouette scores measure how well-separated your phase clusters are:")
    print("• >0.5: Excellent separation • 0.2-0.5: Good separation • <0.2: Poor separation")
    
    best_2d = None
    best_3d = None
    best_2d_score = -1
    best_3d_score = -1
    
    for key, result in results.items():
        encoder_name, method = key.split('_', 1)
        metrics = result['metrics']
        
        sil_2d = metrics.get('silhouette_2d', 0)
        sil_3d = metrics.get('silhouette_3d', 0)
        
        print(f"\n{encoder_name} + {method}:")
        print(f"  2D Silhouette: {sil_2d:.4f}")
        print(f"  3D Silhouette: {sil_3d:.4f}")
        
        if sil_2d > best_2d_score:
            best_2d_score = sil_2d
            best_2d = key
        
        if sil_3d > best_3d_score:
            best_3d_score = sil_3d
            best_3d = key
        
        # Case information
        unique_cases = len(set(result['scan_ids']))
        print(f"  Unique Cases: {unique_cases}")
        print(f"  Total Samples: {len(result['phases'])}")
    
    print(f"\n🏆 BEST PERFORMING COMBINATIONS:")
    if best_2d:
        print(f"   Best 2D: {best_2d} (Silhouette: {best_2d_score:.4f})")
    if best_3d:
        print(f"   Best 3D: {best_3d} (Silhouette: {best_3d_score:.4f})")
    
    print(f"\n📁 All visualizations saved to: {args.output_dir}")
    
    # Save summary report
    summary_path = os.path.join(args.output_dir, "analysis_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Feature Visualization Analysis Summary\n")
        f.write("="*50 + "\n\n")
        f.write("SILHOUETTE SCORE EXPLANATION:\n")
        f.write(explain_silhouette_score())
        f.write("\n\nRESULTS:\n")
        for key, result in results.items():
            encoder_name, method = key.split('_', 1)
            metrics = result['metrics']
            f.write(f"\n{encoder_name} + {method}:\n")
            f.write(f"  2D Silhouette: {metrics.get('silhouette_2d', 0):.4f}\n")
            f.write(f"  3D Silhouette: {metrics.get('silhouette_3d', 0):.4f}\n")
            f.write(f"  Unique Cases: {len(set(result['scan_ids']))}\n")
    
    print(f"📝 Detailed summary saved to: {summary_path}")
    print("🎉 Analysis complete!")

if __name__ == "__main__":
    main()