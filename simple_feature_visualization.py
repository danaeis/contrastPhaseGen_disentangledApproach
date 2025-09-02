#!/usr/bin/env python3
"""
Simple Feature Visualization for Sequential Training Results

Shows before/after training feature disentanglement with minimal code.
Run this after your sequential training to see if DANN worked.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
from tqdm import tqdm
import os
import argparse


def extract_features_simple(model, data_loader, device, max_batches=20):
    """Extract features from model"""
    print(f"🔍 Extracting features...")
    
    model.eval()
    all_features = []
    all_phases = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Extracting")):
            if batch_idx >= max_batches:
                break
                
            try:
                input_vol = batch["input_path"].to(device)
                input_phase = batch["input_phase"]
                
                # Extract features
                features = model(input_vol)
                
                # Flatten features
                if len(features.shape) > 2:
                    features = features.view(features.size(0), -1)
                
                all_features.append(features.cpu().numpy())
                all_phases.extend(input_phase.numpy())
                
            except Exception as e:
                print(f"⚠️ Batch {batch_idx} error: {e}")
                continue
    
    features = np.vstack(all_features)
    phases = np.array(all_phases)
    
    print(f"✅ Extracted {len(features)} features from {len(np.unique(phases))} phases")
    return features, phases


def analyze_features_simple(features, phases, title="Feature Analysis"):
    """Simple feature analysis with PCA and t-SNE"""
    
    print(f"📊 Analyzing features: {title}")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # PCA
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features_scaled)
    
    # t-SNE (if not too many samples)
    if len(features) <= 1000:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
        tsne_features = tsne.fit_transform(features_scaled)
    else:
        tsne_features = None
        print("⚠️ Too many samples for t-SNE, skipping")
    
    # Calculate metrics
    metrics = {}
    if len(np.unique(phases)) > 1:
        metrics['silhouette_pca'] = silhouette_score(pca_features[:, :2], phases)
        if tsne_features is not None:
            metrics['silhouette_tsne'] = silhouette_score(tsne_features, phases)
        metrics['pca_variance_explained'] = np.sum(pca.explained_variance_ratio_[:3])
    
    return {
        'pca_features': pca_features,
        'tsne_features': tsne_features,
        'metrics': metrics,
        'pca': pca
    }


def plot_feature_analysis(analysis_results, phases, title, save_path=None):
    """Plot feature analysis results"""
    
    phase_colors = {0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1'}
    phase_names = {0: 'Non-contrast', 1: 'Arterial', 2: 'Venous'}
    
    # Create figure
    if analysis_results['tsne_features'] is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes = [axes]
    
    pca_features = analysis_results['pca_features']
    tsne_features = analysis_results['tsne_features']
    metrics = analysis_results['metrics']
    
    # PCA 2D
    ax = axes[0] if len(axes) == 1 else axes[0][0]
    unique_phases = np.unique(phases)
    for phase in unique_phases:
        mask = phases == phase
        ax.scatter(pca_features[mask, 0], pca_features[mask, 1],
                  c=phase_colors.get(phase, 'gray'),
                  label=phase_names.get(phase, f'Phase {phase}'),
                  alpha=0.7, s=20)
    
    ax.set_title(f'PCA 2D - {title}')
    ax.set_xlabel(f'PC1 ({analysis_results["pca"].explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({analysis_results["pca"].explained_variance_ratio_[1]:.2%})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # PCA 3D (if multiple subplots)
    if len(axes) > 1:
        ax = axes[0][1]
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        for phase in unique_phases:
            mask = phases == phase
            ax.scatter(pca_features[mask, 0], pca_features[mask, 1], pca_features[mask, 2],
                      c=phase_colors.get(phase, 'gray'),
                      label=phase_names.get(phase, f'Phase {phase}'),
                      alpha=0.7, s=20)
        
        ax.set_title(f'PCA 3D - {title}')
        ax.legend()
    
    # t-SNE (if available)
    if tsne_features is not None and len(axes) > 1:
        ax = axes[1][0]
        for phase in unique_phases:
            mask = phases == phase
            ax.scatter(tsne_features[mask, 0], tsne_features[mask, 1],
                      c=phase_colors.get(phase, 'gray'),
                      label=phase_names.get(phase, f'Phase {phase}'),
                      alpha=0.7, s=20)
        
        ax.set_title(f't-SNE 2D - {title}')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Metrics summary
        ax = axes[1][1]
        ax.axis('off')
        
        metrics_text = f"📊 METRICS SUMMARY\n\n"
        metrics_text += f"PCA Silhouette: {metrics.get('silhouette_pca', 0):.4f}\n"
        metrics_text += f"t-SNE Silhouette: {metrics.get('silhouette_tsne', 0):.4f}\n"
        metrics_text += f"PCA Variance: {metrics.get('pca_variance_explained', 0):.3f}\n\n"
        
        # Interpretation
        silh_pca = metrics.get('silhouette_pca', 0)
        if silh_pca > 0.5:
            metrics_text += "🟢 EXCELLENT separation\n"
        elif silh_pca > 0.3:
            metrics_text += "🟡 GOOD separation\n"
        elif silh_pca > 0.1:
            metrics_text += "🟠 MODERATE separation\n"
        else:
            metrics_text += "🔴 POOR separation\n"
        
        ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved: {save_path}")
    
    plt.show()
    
    return fig


def compare_before_after(before_features, after_features, phases, output_dir):
    """Compare features before and after training"""
    
    print(f"\n🔄 Comparing Before vs After Training Features")
    
    # Ensure same number of samples
    min_samples = min(len(before_features), len(after_features))
    before_features = before_features[:min_samples]
    after_features = after_features[:min_samples]
    phases = phases[:min_samples]
    
    # Analyze both
    before_analysis = analyze_features_simple(before_features, phases, "Before Training")
    after_analysis = analyze_features_simple(after_features, phases, "After Training")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    phase_colors = {0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1'}
    phase_names = {0: 'Non-contrast', 1: 'Arterial', 2: 'Venous'}
    unique_phases = np.unique(phases)
    
    # Before PCA
    ax = axes[0, 0]
    for phase in unique_phases:
        mask = phases == phase
        ax.scatter(before_analysis['pca_features'][mask, 0], 
                  before_analysis['pca_features'][mask, 1],
                  c=phase_colors.get(phase, 'gray'),
                  label=phase_names.get(phase, f'Phase {phase}'),
                  alpha=0.7, s=20)
    ax.set_title('Before Training - PCA')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # After PCA
    ax = axes[0, 1]
    for phase in unique_phases:
        mask = phases == phase
        ax.scatter(after_analysis['pca_features'][mask, 0],
                  after_analysis['pca_features'][mask, 1],
                  c=phase_colors.get(phase, 'gray'),
                  label=phase_names.get(phase, f'Phase {phase}'),
                  alpha=0.7, s=20)
    ax.set_title('After Training - PCA')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Before t-SNE (if available)
    ax = axes[1, 0]
    if before_analysis['tsne_features'] is not None:
        for phase in unique_phases:
            mask = phases == phase
            ax.scatter(before_analysis['tsne_features'][mask, 0],
                      before_analysis['tsne_features'][mask, 1],
                      c=phase_colors.get(phase, 'gray'),
                      label=phase_names.get(phase, f'Phase {phase}'),
                      alpha=0.7, s=20)
        ax.set_title('Before Training - t-SNE')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 't-SNE not available', ha='center', va='center', transform=ax.transAxes)
    
    # After t-SNE (if available)
    ax = axes[1, 1]
    if after_analysis['tsne_features'] is not None:
        for phase in unique_phases:
            mask = phases == phase
            ax.scatter(after_analysis['tsne_features'][mask, 0],
                      after_analysis['tsne_features'][mask, 1],
                      c=phase_colors.get(phase, 'gray'),
                      label=phase_names.get(phase, f'Phase {phase}'),
                      alpha=0.7, s=20)
        ax.set_title('After Training - t-SNE')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 't-SNE not available', ha='center', va='center', transform=ax.transAxes)
    
    plt.suptitle('Feature Disentanglement: Before vs After Sequential Training', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save comparison
    comparison_path = os.path.join(output_dir, 'before_after_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print comparison summary
    before_metrics = before_analysis['metrics']
    after_metrics = after_analysis['metrics']
    
    print(f"\n📊 COMPARISON SUMMARY:")
    print(f"{'Metric':<20} {'Before':<12} {'After':<12} {'Change':<12}")
    print("-" * 60)
    
    for metric in ['silhouette_pca', 'silhouette_tsne', 'pca_variance_explained']:
        before_val = before_metrics.get(metric, 0)
        after_val = after_metrics.get(metric, 0)
        
        if before_val > 0 and after_val > 0:
            change = ((after_val - before_val) / before_val) * 100
            change_str = f"{change:+.1f}%"
        else:
            change_str = "N/A"
        
        print(f"{metric:<20} {before_val:<12.4f} {after_val:<12.4f} {change_str:<12}")
    
    # DANN success assessment
    print(f"\n🎯 DANN ASSESSMENT:")
    
    before_silh = before_metrics.get('silhouette_pca', 0)
    after_silh = after_metrics.get('silhouette_pca', 0)
    
    if before_silh > 0.4 and after_silh < 0.3:
        print("✅ DANN SUCCESS: Phases were separable, now mixed (phase-invariant features!)")
    elif before_silh > after_silh:
        print("🟡 DANN PARTIAL: Some phase mixing achieved")
    elif after_silh > before_silh + 0.1:
        print("🔴 DANN FAILURE: Phases became MORE separable (not phase-invariant)")
    else:
        print("🟡 DANN UNCLEAR: Minimal change in phase separability")
    
    return before_analysis, after_analysis


def main():
    parser = argparse.ArgumentParser(description="Simple Feature Visualization")
    parser.add_argument("--checkpoint_before", type=str, help="Checkpoint before training")
    parser.add_argument("--checkpoint_after", type=str, required=True, help="Checkpoint after training")
    parser.add_argument("--data_path", type=str, required=True, help="Data path")
    parser.add_argument("--output_dir", type=str, default="feature_analysis", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_batches", type=int, default=20, help="Max batches to analyze")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    print("🔍 Simple Feature Visualization for Sequential Training")
    print(f"📁 Output: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # TODO: Replace with your data loader creation
    """
    from your_data_loading import create_data_loader
    data_loader = create_data_loader(args.data_path, args.batch_size)
    """
    
    # TODO: Replace with your model creation
    """
    from your_models import YourEncoder
    
    # Load trained model
    checkpoint = torch.load(args.checkpoint_after, map_location=args.device)
    trained_encoder = YourEncoder().to(args.device)
    trained_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    
    # Extract features from trained model
    after_features, phases = extract_features_simple(trained_encoder, data_loader, args.device, args.max_batches)
    
    # Analyze trained features
    after_analysis = analyze_features_simple(after_features, phases, "After Sequential Training")
    plot_feature_analysis(after_analysis, phases, "After Sequential Training", 
                         os.path.join(args.output_dir, "after_training.png"))
    
    # Compare with before training if available
    if args.checkpoint_before:
        before_encoder = YourEncoder().to(args.device)
        before_checkpoint = torch.load(args.checkpoint_before, map_location=args.device)
        before_encoder.load_state_dict(before_checkpoint['encoder_state_dict'])
        
        before_features, _ = extract_features_simple(before_encoder, data_loader, args.device, args.max_batches)
        
        compare_before_after(before_features, after_features, phases, args.output_dir)
    """
    
    print("\n💡 TO USE THIS SCRIPT:")
    print("1. Replace the TODO sections with your model and data loading code")
    print("2. Run after your sequential training:")
    print("   python simple_feature_visualization.py \\")
    print("       --checkpoint_after final_optimized_checkpoint.pth \\")
    print("       --data_path /your/data/path \\")
    print("       --output_dir feature_analysis")
    
    print("\n📊 EXPECTED RESULTS:")
    print("✅ If DANN worked: Phases should be mixed/overlapping in after-training plots")
    print("🔴 If DANN failed: Phases remain clearly separated in after-training plots")


if __name__ == "__main__":
    main()