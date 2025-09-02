#!/usr/bin/env python3
"""
Post-Training Feature Visualization & Disentanglement Analysis

Analyzes learned features after sequential training to evaluate:
1. Phase-invariant feature learning (DANN success)
2. Reconstruction quality preservation
3. Feature separability and clustering
4. Before/after training comparison
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, classification_report
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import os
import argparse
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
import pickle
import time
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')


class PostTrainingFeatureAnalyzer:
    """
    Comprehensive feature analysis after sequential training with disentanglement
    """
    
    def __init__(self, device='cuda', output_dir='post_training_analysis'):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Color schemes for visualization
        self.phase_colors = {
            0: '#FF6B6B',  # Non-contrast - Red
            1: '#4ECDC4',  # Arterial - Teal
            2: '#45B7D1',  # Venous - Blue
            3: '#96CEB4'   # Delayed - Green (if applicable)
        }
        
        self.phase_names = {
            0: 'Non-contrast',
            1: 'Arterial', 
            2: 'Venous',
            3: 'Delayed'
        }
        
        self.results = {}
        
    def load_trained_models(self, checkpoint_path: str):
        """Load trained models from checkpoint"""
        print(f"📂 Loading trained models from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract model state dicts
        if 'model_states' in checkpoint:
            model_states = checkpoint['model_states']
        else:
            # Handle different checkpoint formats
            model_states = {
                'encoder': checkpoint.get('encoder_state_dict'),
                'generator': checkpoint.get('generator_state_dict'),
                'discriminator': checkpoint.get('discriminator_state_dict'),
                'phase_detector': checkpoint.get('phase_detector_state_dict')
            }
        
        print("✅ Successfully loaded model states")
        print(f"   Available models: {list(model_states.keys())}")
        
        return model_states, checkpoint.get('final_results', {})
    
    def extract_features_from_trained_encoder(self, encoder, data_loader, 
                                            max_batches=None, use_generator=False, 
                                            generator=None):
        """
        Extract features from trained encoder
        
        Args:
            encoder: Trained encoder model
            data_loader: Data loader for feature extraction
            max_batches: Limit number of batches for analysis
            use_generator: Also extract generated samples
            generator: Trained generator (if use_generator=True)
        """
        print(f"🔍 Extracting features from trained encoder...")
        
        encoder.eval()
        if generator is not None:
            generator.eval()
        
        all_features = []
        all_phases = []
        all_scan_ids = []
        all_reconstructions = []
        all_targets = []
        
        batch_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Extracting Features")):
                if max_batches and batch_count >= max_batches:
                    break
                
                try:
                    input_vol = batch["input_path"].to(self.device)
                    target_vol = batch["target_path"].to(self.device) 
                    input_phase = batch["input_phase"]
                    target_phase = batch["target_phase"]
                    
                    # Get scan IDs if available
                    scan_ids = batch.get("scan_id", [f"scan_{batch_idx}_{i}" for i in range(len(input_vol))])
                    
                    # Extract encoder features
                    features = encoder(input_vol)
                    
                    # Flatten features for analysis
                    if len(features.shape) > 2:
                        features_flat = features.view(features.size(0), -1)
                    else:
                        features_flat = features
                    
                    # Store features and metadata
                    all_features.append(features_flat.cpu().numpy())
                    all_phases.extend(input_phase.numpy())
                    all_scan_ids.extend(scan_ids)
                    
                    # Generate reconstructions if requested
                    if use_generator and generator is not None:
                        # Create phase embeddings
                        phase_emb = self._create_phase_embedding(target_phase, dim=32)
                        generated = generator(features, phase_emb)
                        
                        all_reconstructions.append(generated.cpu().numpy())
                        all_targets.append(target_vol.cpu().numpy())
                    
                    batch_count += 1
                    
                except Exception as e:
                    print(f"⚠️ Error processing batch {batch_idx}: {e}")
                    continue
        
        # Combine all features
        all_features = np.vstack(all_features)
        all_phases = np.array(all_phases)
        
        print(f"✅ Extracted features from {len(all_features)} samples")
        print(f"   Feature shape: {all_features.shape}")
        print(f"   Phase distribution: {np.bincount(all_phases)}")
        
        results = {
            'features': all_features,
            'phases': all_phases,
            'scan_ids': all_scan_ids
        }
        
        if use_generator:
            results['reconstructions'] = np.vstack(all_reconstructions) if all_reconstructions else None
            results['targets'] = np.vstack(all_targets) if all_targets else None
        
        return results
    
    def _create_phase_embedding(self, phase_labels, dim=32):
        """Create simple phase embeddings for generator"""
        batch_size = phase_labels.size(0)
        phase_emb = torch.zeros(batch_size, dim, device=self.device)
        
        for i in range(3):  # 3 phases
            mask = (phase_labels == i)
            if mask.any():
                phase_emb[mask, i*dim//3:(i+1)*dim//3] = 1.0
        
        return phase_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    
    def analyze_feature_disentanglement(self, features, phases, analysis_name="post_training"):
        """
        Comprehensive analysis of feature disentanglement
        """
        print(f"\n📊 Analyzing feature disentanglement: {analysis_name}")
        
        # Apply dimensionality reduction methods
        methods = ['PCA', 'TSNE', 'LDA']
        results = {}
        
        for method in methods:
            print(f"\n🔍 Applying {method} analysis...")
            
            try:
                # Apply dimensionality reduction
                reduced_2d, reduced_3d, reducers, dr_metrics = self._apply_dimensionality_reduction(
                    features, phases, method=method
                )
                
                # Compute comprehensive metrics
                comp_metrics = self._compute_comprehensive_metrics(reduced_2d, reduced_3d, phases)
                
                # Combine all metrics
                all_metrics = {**dr_metrics, **comp_metrics}
                
                # Create visualizations
                self._create_disentanglement_visualizations(
                    reduced_2d, reduced_3d, phases, method, analysis_name, all_metrics
                )
                
                results[method] = {
                    'reduced_2d': reduced_2d,
                    'reduced_3d': reduced_3d,
                    'reducers': reducers,
                    'metrics': all_metrics
                }
                
                # Print key metrics
                print(f"✅ {method} Results:")
                print(f"   2D Silhouette Score: {all_metrics.get('silhouette_2d', 0):.4f}")
                print(f"   3D Silhouette Score: {all_metrics.get('silhouette_3d', 0):.4f}")
                print(f"   Fisher Ratio 2D: {all_metrics.get('fisher_ratio_2d', 0):.4f}")
                
                if method == 'PCA':
                    print(f"   Variance Explained 2D: {all_metrics.get('total_variance_explained_2d', 0):.3f}")
                    print(f"   Variance Explained 3D: {all_metrics.get('total_variance_explained_3d', 0):.3f}")
                
            except Exception as e:
                print(f"⚠️ Error in {method} analysis: {e}")
                continue
        
        return results
    
    def _apply_dimensionality_reduction(self, features, phases, method='PCA', 
                                      n_components_2d=2, n_components_3d=3, 
                                      random_state=42):
        """Apply dimensionality reduction with proper handling"""
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        metrics = {}
        
        if method.upper() == 'PCA':
            reducer_2d = PCA(n_components=n_components_2d, random_state=random_state)
            reducer_3d = PCA(n_components=n_components_3d, random_state=random_state)
            
            reduced_2d = reducer_2d.fit_transform(features_scaled)
            reduced_3d = reducer_3d.fit_transform(features_scaled)
            
            metrics.update({
                'explained_variance_ratio_2d': reducer_2d.explained_variance_ratio_.tolist(),
                'explained_variance_ratio_3d': reducer_3d.explained_variance_ratio_.tolist(),
                'total_variance_explained_2d': np.sum(reducer_2d.explained_variance_ratio_),
                'total_variance_explained_3d': np.sum(reducer_3d.explained_variance_ratio_)
            })
            
        elif method.upper() == 'TSNE':
            # Preprocess with PCA if high-dimensional
            if features_scaled.shape[1] > 50:
                pca_preprocess = PCA(n_components=50, random_state=random_state)
                features_scaled = pca_preprocess.fit_transform(features_scaled)
            
            reducer_2d = TSNE(n_components=2, random_state=random_state, 
                            perplexity=min(30, len(features)//4), max_iter=1000)
            reducer_3d = TSNE(n_components=3, random_state=random_state,
                            perplexity=min(30, len(features)//4), max_iter=1000)
            
            reduced_2d = reducer_2d.fit_transform(features_scaled)
            reduced_3d = reducer_3d.fit_transform(features_scaled)
            
            metrics.update({
                'kl_divergence_2d': getattr(reducer_2d, 'kl_divergence_', None),
                'kl_divergence_3d': getattr(reducer_3d, 'kl_divergence_', None)
            })
            
        elif method.upper() == 'LDA':
            unique_phases = np.unique(phases)
            max_components = len(unique_phases) - 1
            
            actual_2d_components = min(n_components_2d, max_components)
            actual_3d_components = min(n_components_3d, max_components)
            
            reducer_2d = LinearDiscriminantAnalysis(n_components=actual_2d_components)
            reduced_2d = reducer_2d.fit_transform(features_scaled, phases)
            
            if actual_3d_components >= 3:
                reducer_3d = LinearDiscriminantAnalysis(n_components=actual_3d_components)
                reduced_3d = reducer_3d.fit_transform(features_scaled, phases)
            else:
                reduced_3d = reduced_2d
                reducer_3d = reducer_2d
            
            metrics.update({
                'explained_variance_ratio_2d': getattr(reducer_2d, 'explained_variance_ratio_', []).tolist(),
                'n_components_used_2d': actual_2d_components,
                'n_components_used_3d': actual_3d_components
            })
        
        return reduced_2d, reduced_3d, (reducer_2d, reducer_3d), metrics
    
    def _compute_comprehensive_metrics(self, reduced_2d, reduced_3d, phases):
        """Compute comprehensive disentanglement metrics"""
        metrics = {}
        
        if len(np.unique(phases)) > 1:
            # Silhouette scores
            metrics['silhouette_2d'] = silhouette_score(reduced_2d, phases)
            if reduced_3d is not None and reduced_3d.shape[1] >= 2:
                metrics['silhouette_3d'] = silhouette_score(reduced_3d, phases)
            else:
                metrics['silhouette_3d'] = metrics['silhouette_2d']
            
            # Fisher discriminant ratios
            metrics['fisher_ratio_2d'] = self._compute_fisher_ratio(reduced_2d, phases)
            if reduced_3d is not None:
                metrics['fisher_ratio_3d'] = self._compute_fisher_ratio(reduced_3d, phases)
            
            # Inter-cluster vs intra-cluster distances
            metrics['cluster_separation_2d'] = self._compute_cluster_separation(reduced_2d, phases)
            if reduced_3d is not None:
                metrics['cluster_separation_3d'] = self._compute_cluster_separation(reduced_3d, phases)
            
            # Phase confusion analysis (lower is better for DANN)
            metrics['phase_confusion_score'] = self._compute_phase_confusion(reduced_2d, phases)
            
        return metrics
    
    def _compute_fisher_ratio(self, features, labels):
        """Compute Fisher discriminant ratio (between-class vs within-class variance)"""
        between_var = self._compute_between_class_variance(features, labels)
        within_var = self._compute_within_class_variance(features, labels)
        return between_var / (within_var + 1e-8)
    
    def _compute_between_class_variance(self, features, labels):
        """Compute between-class variance"""
        overall_mean = np.mean(features, axis=0)
        unique_labels = np.unique(labels)
        
        between_var = 0
        total_samples = len(features)
        
        for label in unique_labels:
            class_mask = (labels == label)
            class_samples = np.sum(class_mask)
            if class_samples > 0:
                class_mean = np.mean(features[class_mask], axis=0)
                class_weight = class_samples / total_samples
                between_var += class_weight * np.sum((class_mean - overall_mean) ** 2)
        
        return between_var
    
    def _compute_within_class_variance(self, features, labels):
        """Compute within-class variance"""
        unique_labels = np.unique(labels)
        within_var = 0
        
        for label in unique_labels:
            class_mask = (labels == label)
            class_features = features[class_mask]
            if len(class_features) > 0:
                class_mean = np.mean(class_features, axis=0)
                within_var += np.sum((class_features - class_mean) ** 2)
        
        return within_var / len(features)
    
    def _compute_cluster_separation(self, features, labels):
        """Compute ratio of inter-cluster to intra-cluster distances"""
        unique_labels = np.unique(labels)
        
        # Compute centroids
        centroids = {}
        for label in unique_labels:
            class_mask = (labels == label)
            if np.sum(class_mask) > 0:
                centroids[label] = np.mean(features[class_mask], axis=0)
        
        # Inter-cluster distances
        inter_distances = []
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i+1:]:
                if label1 in centroids and label2 in centroids:
                    dist = np.linalg.norm(centroids[label1] - centroids[label2])
                    inter_distances.append(dist)
        
        # Intra-cluster distances
        intra_distances = []
        for label in unique_labels:
            class_mask = (labels == label)
            class_features = features[class_mask]
            if len(class_features) > 1:
                centroid = centroids[label]
                for feature in class_features:
                    dist = np.linalg.norm(feature - centroid)
                    intra_distances.append(dist)
        
        if len(inter_distances) > 0 and len(intra_distances) > 0:
            return np.mean(inter_distances) / (np.mean(intra_distances) + 1e-8)
        return 0.0
    
    def _compute_phase_confusion(self, features, phases):
        """
        Compute phase confusion score - measures how well phases can be distinguished
        Lower score = better phase invariance (good for DANN)
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        try:
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            scores = cross_val_score(clf, features, phases, cv=3)
            return np.mean(scores)  # Higher = more distinguishable phases
        except:
            return 0.5  # Default neutral score
    
    def _create_disentanglement_visualizations(self, reduced_2d, reduced_3d, phases, 
                                             method, analysis_name, metrics):
        """Create comprehensive disentanglement visualizations"""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Create a comprehensive layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. 2D Scatter Plot
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_2d_scatter(ax1, reduced_2d, phases, f"{method} 2D - {analysis_name}")
        
        # 2. 3D Scatter Plot
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        if reduced_3d is not None and reduced_3d.shape[1] >= 3:
            self._plot_3d_scatter(ax2, reduced_3d, phases, f"{method} 3D - {analysis_name}")
        else:
            ax2.text(0.5, 0.5, 0.5, '3D not available', ha='center', va='center')
        
        # 3. Phase Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_phase_distribution(ax3, phases)
        
        # 4. Silhouette Analysis
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_silhouette_analysis(ax4, reduced_2d, phases)
        
        # 5. Feature Distance Matrix
        ax5 = fig.add_subplot(gs[1, 0])
        self._plot_distance_matrix(ax5, reduced_2d, phases)
        
        # 6. Phase Separation Plot
        ax6 = fig.add_subplot(gs[1, 1])
        self._plot_phase_separation(ax6, reduced_2d, phases)
        
        # 7. Feature Density Plot
        ax7 = fig.add_subplot(gs[1, 2])
        self._plot_feature_density(ax7, reduced_2d, phases)
        
        # 8. Confusion Matrix
        ax8 = fig.add_subplot(gs[1, 3])
        self._plot_phase_confusion_matrix(ax8, reduced_2d, phases)
        
        # 9. Metrics Summary (spans bottom row)
        ax9 = fig.add_subplot(gs[2, :])
        self._plot_metrics_summary(ax9, metrics, method, analysis_name)
        
        plt.suptitle(f"Feature Disentanglement Analysis: {method} - {analysis_name}", 
                    fontsize=16, fontweight='bold')
        
        # Save comprehensive analysis
        save_path = self.output_dir / f"{analysis_name}_{method}_comprehensive_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Comprehensive analysis saved: {save_path}")
    
    def _plot_2d_scatter(self, ax, reduced_2d, phases, title):
        """Plot 2D scatter with phase colors"""
        unique_phases = np.unique(phases)
        
        for phase in unique_phases:
            mask = phases == phase
            ax.scatter(reduced_2d[mask, 0], reduced_2d[mask, 1], 
                      c=self.phase_colors.get(phase, 'gray'),
                      label=self.phase_names.get(phase, f'Phase {phase}'),
                      alpha=0.7, s=20)
        
        ax.set_title(title)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_3d_scatter(self, ax, reduced_3d, phases, title):
        """Plot 3D scatter with phase colors"""
        unique_phases = np.unique(phases)
        
        for phase in unique_phases:
            mask = phases == phase
            ax.scatter(reduced_3d[mask, 0], reduced_3d[mask, 1], reduced_3d[mask, 2],
                      c=self.phase_colors.get(phase, 'gray'),
                      label=self.phase_names.get(phase, f'Phase {phase}'),
                      alpha=0.7, s=20)
        
        ax.set_title(title)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.legend()
    
    def _plot_phase_distribution(self, ax, phases):
        """Plot phase distribution"""
        unique_phases, counts = np.unique(phases, return_counts=True)
        
        colors = [self.phase_colors.get(phase, 'gray') for phase in unique_phases]
        labels = [self.phase_names.get(phase, f'Phase {phase}') for phase in unique_phases]
        
        bars = ax.bar(labels, counts, color=colors, alpha=0.7)
        ax.set_title('Phase Distribution')
        ax.set_ylabel('Count')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom')
    
    def _plot_silhouette_analysis(self, ax, features, phases):
        """Plot silhouette analysis"""
        if len(np.unique(phases)) > 1:
            silhouette_avg = silhouette_score(features, phases)
            
            # Create silhouette plot (simplified version)
            from sklearn.metrics import silhouette_samples
            sample_silhouette_values = silhouette_samples(features, phases)
            
            y_lower = 10
            unique_phases = np.unique(phases)
            
            for i, phase in enumerate(unique_phases):
                phase_silhouette_values = sample_silhouette_values[phases == phase]
                phase_silhouette_values.sort()
                
                size_cluster = phase_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster
                
                color = self.phase_colors.get(phase, 'gray')
                ax.fill_betweenx(np.arange(y_lower, y_upper),
                               0, phase_silhouette_values,
                               facecolor=color, edgecolor=color, alpha=0.7)
                
                ax.text(-0.05, y_lower + 0.5 * size_cluster, 
                       self.phase_names.get(phase, f'Phase {phase}'))
                y_lower = y_upper + 10
            
            ax.axvline(x=silhouette_avg, color="red", linestyle="--")
            ax.set_title(f'Silhouette Analysis (avg: {silhouette_avg:.3f})')
            ax.set_xlabel('Silhouette Score')
        else:
            ax.text(0.5, 0.5, 'Single phase\nNo silhouette analysis', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_distance_matrix(self, ax, features, phases):
        """Plot average distance matrix between phases"""
        unique_phases = np.unique(phases)
        n_phases = len(unique_phases)
        
        distance_matrix = np.zeros((n_phases, n_phases))
        
        for i, phase1 in enumerate(unique_phases):
            for j, phase2 in enumerate(unique_phases):
                mask1 = phases == phase1
                mask2 = phases == phase2
                
                features1 = features[mask1]
                features2 = features[mask2]
                
                if len(features1) > 0 and len(features2) > 0:
                    # Compute average distance between phase centroids
                    centroid1 = np.mean(features1, axis=0)
                    centroid2 = np.mean(features2, axis=0)
                    distance_matrix[i, j] = np.linalg.norm(centroid1 - centroid2)
        
        im = ax.imshow(distance_matrix, cmap='viridis', aspect='auto')
        ax.set_title('Inter-Phase Distance Matrix')
        
        # Set ticks and labels
        phase_labels = [self.phase_names.get(p, f'Phase {p}') for p in unique_phases]
        ax.set_xticks(range(n_phases))
        ax.set_yticks(range(n_phases))
        ax.set_xticklabels(phase_labels, rotation=45)
        ax.set_yticklabels(phase_labels)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add distance values
        for i in range(n_phases):
            for j in range(n_phases):
                ax.text(j, i, f'{distance_matrix[i, j]:.2f}',
                       ha='center', va='center', color='white')
    
    def _plot_phase_separation(self, ax, features, phases):
        """Plot phase separation quality"""
        unique_phases = np.unique(phases)
        
        separations = []
        phase_pairs = []
        
        for i, phase1 in enumerate(unique_phases):
            for phase2 in unique_phases[i+1:]:
                mask1 = phases == phase1
                mask2 = phases == phase2
                
                if np.sum(mask1) > 0 and np.sum(mask2) > 0:
                    features1 = features[mask1]
                    features2 = features[mask2]
                    
                    # Calculate separation (distance between centroids / average within-cluster distance)
                    centroid1 = np.mean(features1, axis=0)
                    centroid2 = np.mean(features2, axis=0)
                    
                    inter_dist = np.linalg.norm(centroid1 - centroid2)
                    
                    intra_dist1 = np.mean([np.linalg.norm(f - centroid1) for f in features1])
                    intra_dist2 = np.mean([np.linalg.norm(f - centroid2) for f in features2])
                    avg_intra_dist = (intra_dist1 + intra_dist2) / 2
                    
                    separation = inter_dist / (avg_intra_dist + 1e-8)
                    separations.append(separation)
                    phase_pairs.append(f"{self.phase_names.get(phase1, f'P{phase1}')} vs {self.phase_names.get(phase2, f'P{phase2}')}")
        
        if separations:
            bars = ax.bar(range(len(separations)), separations, 
                         color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(separations)])
            ax.set_title('Phase Separation Quality')
            ax.set_ylabel('Separation Ratio')
            ax.set_xticks(range(len(separations)))
            ax.set_xticklabels(phase_pairs, rotation=45)
            
            # Add value labels
            for bar, sep in zip(bars, separations):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{sep:.2f}', ha='center', va='bottom')
    
    def _plot_feature_density(self, ax, features, phases):
        """Plot feature density for each phase"""
        unique_phases = np.unique(phases)
        
        for phase in unique_phases:
            mask = phases == phase
            phase_features = features[mask]
            
            if len(phase_features) > 0:
                # Plot density of first component
                ax.hist(phase_features[:, 0], bins=20, alpha=0.5, 
                       color=self.phase_colors.get(phase, 'gray'),
                       label=self.phase_names.get(phase, f'Phase {phase}'),
                       density=True)
        
        ax.set_title('Feature Density Distribution (Component 1)')
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_phase_confusion_matrix(self, ax, features, phases):
        """Plot confusion matrix from phase classification"""
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import confusion_matrix
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, phases, test_size=0.3, random_state=42
            )
            
            # Train classifier
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            ax.set_title('Phase Classification\nConfusion Matrix')
            
            unique_phases = np.unique(phases)
            phase_labels = [self.phase_names.get(p, f'Phase {p}') for p in unique_phases]
            
            ax.set_xticks(range(len(unique_phases)))
            ax.set_yticks(range(len(unique_phases)))
            ax.set_xticklabels(phase_labels, rotation=45)
            ax.set_yticklabels(phase_labels)
            
            # Add numbers
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
            
            ax.set_ylabel('True Phase')
            ax.set_xlabel('Predicted Phase')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Classification failed:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_metrics_summary(self, ax, metrics, method, analysis_name):
        """Plot metrics summary"""
        ax.axis('off')
        
        # Create metrics text
        metrics_text = f"📊 {method} Analysis Summary - {analysis_name}\n\n"
        
        # Key metrics
        key_metrics = [
            ('Silhouette 2D', metrics.get('silhouette_2d', 0), '0.4f'),
            ('Silhouette 3D', metrics.get('silhouette_3d', 0), '0.4f'),
            ('Fisher Ratio 2D', metrics.get('fisher_ratio_2d', 0), '0.4f'),
            ('Cluster Separation 2D', metrics.get('cluster_separation_2d', 0), '0.4f'),
            ('Phase Confusion Score', metrics.get('phase_confusion_score', 0), '0.4f'),
        ]
        
        if method == 'PCA':
            key_metrics.extend([
                ('Variance Explained 2D', metrics.get('total_variance_explained_2d', 0), '0.3f'),
                ('Variance Explained 3D', metrics.get('total_variance_explained_3d', 0), '0.3f'),
            ])
        
        # Format metrics
        for i, (name, value, fmt) in enumerate(key_metrics):
            if value is not None:
                metrics_text += f"{name:25s}: {value:{fmt}}\n"
                
                # Add interpretation
                if 'silhouette' in name.lower():
                    if value > 0.5:
                        interp = "🟢 Excellent clustering"
                    elif value > 0.3:
                        interp = "🟡 Good clustering"  
                    elif value > 0.1:
                        interp = "🟠 Moderate clustering"
                    else:
                        interp = "🔴 Poor clustering"
                    metrics_text += f"{' '*27}({interp})\n"
                
                elif 'fisher' in name.lower():
                    if value > 2.0:
                        interp = "🟢 Excellent separation"
                    elif value > 1.0:
                        interp = "🟡 Good separation"
                    else:
                        interp = "🔴 Poor separation"
                    metrics_text += f"{' '*27}({interp})\n"
                
                elif 'confusion' in name.lower():
                    if value < 0.4:
                        interp = "🟢 Excellent DANN (phase-invariant)"
                    elif value < 0.6:
                        interp = "🟡 Good DANN"
                    else:
                        interp = "🔴 Poor DANN (phases distinguishable)"
                    metrics_text += f"{' '*27}({interp})\n"
        
        # Add overall assessment
        silh_2d = metrics.get('silhouette_2d', 0)
        fisher_2d = metrics.get('fisher_ratio_2d', 0)
        confusion = metrics.get('phase_confusion_score', 1)
        
        metrics_text += f"\n📈 OVERALL ASSESSMENT:\n"
        
        if silh_2d > 0.4 and fisher_2d > 1.5:
            metrics_text += "✅ EXCELLENT: Well-separated, meaningful features\n"
        elif silh_2d > 0.3 and fisher_2d > 1.0:
            metrics_text += "🟡 GOOD: Reasonable feature separation\n"
        else:
            metrics_text += "🔴 NEEDS IMPROVEMENT: Poor feature separation\n"
            
        if confusion < 0.4:
            metrics_text += "✅ DANN SUCCESS: Phase-invariant features achieved\n"
        elif confusion < 0.6:
            metrics_text += "🟡 DANN PARTIAL: Some phase invariance\n"
        else:
            metrics_text += "🔴 DANN FAILURE: Phases still distinguishable\n"
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    def compare_before_after_training(self, before_features, after_features, phases, 
                                    before_name="Before Training", after_name="After Training"):
        """Compare features before and after training"""
        print(f"\n🔄 Comparing features: {before_name} vs {after_name}")
        
        # Analyze both feature sets
        before_results = self.analyze_feature_disentanglement(before_features, phases, before_name)
        after_results = self.analyze_feature_disentanglement(after_features, phases, after_name)
        
        # Create comparison visualization
        self._create_comparison_visualization(before_results, after_results, phases, 
                                            before_name, after_name)
        
        # Generate comparison report
        comparison_report = self._generate_comparison_report(before_results, after_results,
                                                          before_name, after_name)
        
        return before_results, after_results, comparison_report
    
    def _create_comparison_visualization(self, before_results, after_results, phases,
                                       before_name, after_name):
        """Create side-by-side comparison visualization"""
        
        methods = ['PCA', 'TSNE']  # Focus on most interpretable methods
        
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        for method_idx, method in enumerate(methods):
            if method not in before_results or method not in after_results:
                continue
                
            # Before training plots
            row_offset = method_idx * 2
            
            # Before 2D
            ax1 = fig.add_subplot(gs[row_offset, 0])
            before_2d = before_results[method]['reduced_2d']
            self._plot_2d_scatter(ax1, before_2d, phases, f"{before_name} - {method} 2D")
            
            # Before 3D
            ax2 = fig.add_subplot(gs[row_offset, 1], projection='3d')
            before_3d = before_results[method]['reduced_3d']
            if before_3d is not None and before_3d.shape[1] >= 3:
                self._plot_3d_scatter(ax2, before_3d, phases, f"{before_name} - {method} 3D")
            
            # After 2D
            ax3 = fig.add_subplot(gs[row_offset, 2])
            after_2d = after_results[method]['reduced_2d']
            self._plot_2d_scatter(ax3, after_2d, phases, f"{after_name} - {method} 2D")
            
            # After 3D  
            ax4 = fig.add_subplot(gs[row_offset, 3], projection='3d')
            after_3d = after_results[method]['reduced_3d']
            if after_3d is not None and after_3d.shape[1] >= 3:
                self._plot_3d_scatter(ax4, after_3d, phases, f"{after_name} - {method} 3D")
            
            # Metrics comparison
            ax5 = fig.add_subplot(gs[row_offset + 1, :])
            self._plot_metrics_comparison(ax5, before_results[method]['metrics'], 
                                        after_results[method]['metrics'], method,
                                        before_name, after_name)
        
        plt.suptitle(f"Feature Disentanglement Comparison: {before_name} vs {after_name}", 
                    fontsize=16, fontweight='bold')
        
        save_path = self.output_dir / f"comparison_{before_name}_{after_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Comparison visualization saved: {save_path}")
    
    def _plot_metrics_comparison(self, ax, before_metrics, after_metrics, method,
                               before_name, after_name):
        """Plot metrics comparison bar chart"""
        
        # Select key metrics to compare
        key_metrics = ['silhouette_2d', 'silhouette_3d', 'fisher_ratio_2d', 
                      'cluster_separation_2d', 'phase_confusion_score']
        
        metric_names = []
        before_values = []
        after_values = []
        
        for metric in key_metrics:
            if metric in before_metrics and metric in after_metrics:
                metric_names.append(metric.replace('_', ' ').title())
                before_values.append(before_metrics[metric] or 0)
                after_values.append(after_metrics[metric] or 0)
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, before_values, width, label=before_name, alpha=0.7)
        bars2 = ax.bar(x + width/2, after_values, width, label=after_name, alpha=0.7)
        
        ax.set_title(f'{method} Metrics Comparison')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    def _generate_comparison_report(self, before_results, after_results, 
                                  before_name, after_name):
        """Generate detailed comparison report"""
        
        report = f"""
# Feature Disentanglement Comparison Report

## Training Comparison: {before_name} vs {after_name}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report compares feature representations before and after sequential training with DANN-based disentanglement.

"""
        
        # Compare key metrics across methods
        methods = set(before_results.keys()) & set(after_results.keys())
        
        for method in methods:
            report += f"\n### {method} Analysis\n\n"
            
            before_metrics = before_results[method]['metrics']
            after_metrics = after_results[method]['metrics']
            
            key_comparisons = [
                ('silhouette_2d', 'Silhouette Score 2D', 'Higher is better'),
                ('fisher_ratio_2d', 'Fisher Discriminant Ratio', 'Higher is better'),
                ('phase_confusion_score', 'Phase Confusion Score', 'Lower is better for DANN')
            ]
            
            for metric_key, metric_name, interpretation in key_comparisons:
                before_val = before_metrics.get(metric_key, 0)
                after_val = after_metrics.get(metric_key, 0)
                
                if before_val and after_val:
                    change = ((after_val - before_val) / before_val) * 100
                    
                    report += f"**{metric_name}**:\n"
                    report += f"- Before: {before_val:.4f}\n"
                    report += f"- After: {after_val:.4f}\n"
                    report += f"- Change: {change:+.2f}%\n"
                    
                    # Interpretation
                    if 'confusion' in metric_key and change < -10:
                        report += f"- ✅ **Excellent improvement**: Phase confusion reduced (DANN success)\n"
                    elif 'confusion' not in metric_key and change > 10:
                        report += f"- ✅ **Significant improvement**: Better feature quality\n"
                    elif abs(change) < 5:
                        report += f"- 🟡 **Stable**: Minimal change\n"
                    else:
                        report += f"- 🔴 **Needs attention**: Unexpected change pattern\n"
                    
                    report += f"- Note: {interpretation}\n\n"
        
        # Overall assessment
        report += "\n## Overall Training Assessment\n\n"
        
        # Calculate average improvements
        improvements = []
        for method in methods:
            before_silh = before_results[method]['metrics'].get('silhouette_2d', 0)
            after_silh = after_results[method]['metrics'].get('silhouette_2d', 0)
            if before_silh > 0:
                improvement = ((after_silh - before_silh) / before_silh) * 100
                improvements.append(improvement)
        
        if improvements:
            avg_improvement = np.mean(improvements)
            
            if avg_improvement > 20:
                report += "🎉 **EXCELLENT TRAINING SUCCESS**: Features show significant improvement in separability and structure.\n\n"
            elif avg_improvement > 10:
                report += "✅ **GOOD TRAINING SUCCESS**: Features show meaningful improvement.\n\n"
            elif avg_improvement > 0:
                report += "🟡 **MODERATE SUCCESS**: Some improvement observed, consider further training.\n\n"
            else:
                report += "🔴 **TRAINING ISSUES**: Features may have degraded, review training process.\n\n"
        
        # DANN-specific assessment
        dann_success = False
        for method in methods:
            before_conf = before_results[method]['metrics'].get('phase_confusion_score', 1)
            after_conf = after_results[method]['metrics'].get('phase_confusion_score', 1)
            
            if after_conf < 0.4:
                dann_success = True
                break
        
        if dann_success:
            report += "🎯 **DANN SUCCESS**: Phase-invariant features successfully learned!\n\n"
        else:
            report += "⚠️ **DANN PARTIAL**: Phase invariance not fully achieved, consider longer Phase 3 training.\n\n"
        
        # Recommendations
        report += "## Recommendations\n\n"
        
        if dann_success and avg_improvement > 10:
            report += "1. ✅ **Ready for deployment** - Training objectives achieved\n"
            report += "2. 📊 **Monitor performance** on test data\n"
            report += "3. 🚀 **Consider inference optimization** for production\n"
        else:
            report += "1. 🔧 **Continue training** - Extend problematic phases\n"
            report += "2. 📈 **Adjust learning rates** - Lower if unstable, higher if slow\n"
            report += "3. 🔍 **Review data quality** - Check preprocessing and normalization\n"
            report += "4. 🎯 **Focus on DANN parameters** - Adjust lambda scheduling\n"
        
        # Save report
        report_path = self.output_dir / f"comparison_report_{before_name}_{after_name}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"📝 Comparison report saved: {report_path}")
        
        return report

    def create_interactive_feature_explorer(self, features, phases, scan_ids, analysis_name):
        """Create interactive feature explorer with Plotly"""
        
        # Apply PCA for interactive visualization
        pca = PCA(n_components=3, random_state=42)
        features_scaled = StandardScaler().fit_transform(features)
        reduced_3d = pca.fit_transform(features_scaled)
        
        # Create DataFrame
        df = pd.DataFrame({
            'PC1': reduced_3d[:, 0],
            'PC2': reduced_3d[:, 1], 
            'PC3': reduced_3d[:, 2],
            'Phase': [self.phase_names.get(p, f'Phase {p}') for p in phases],
            'Phase_ID': phases,
            'Scan_ID': scan_ids,
            'Silhouette': silhouette_samples(reduced_3d, phases) if len(np.unique(phases)) > 1 else [0]*len(phases)
        })
        
        # Create interactive 3D scatter
        fig = px.scatter_3d(
            df, x='PC1', y='PC2', z='PC3',
            color='Phase',
            hover_data=['Scan_ID', 'Silhouette'],
            title=f'Interactive Feature Explorer - {analysis_name}',
            color_discrete_map={
                self.phase_names.get(0, 'Phase 0'): self.phase_colors[0],
                self.phase_names.get(1, 'Phase 1'): self.phase_colors[1], 
                self.phase_names.get(2, 'Phase 2'): self.phase_colors[2]
            }
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)'
            ),
            width=1000,
            height=700
        )
        
        # Save interactive plot
        interactive_path = self.output_dir / f"{analysis_name}_interactive_explorer.html"
        fig.write_html(str(interactive_path))
        
        print(f"🌐 Interactive feature explorer saved: {interactive_path}")
        
        return fig


# Example usage and main function
def main():
    parser = argparse.ArgumentParser(description="Post-Training Feature Visualization & Analysis")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                       help="Path to trained model checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to data for feature extraction")
    parser.add_argument("--output_dir", type=str, default="post_training_analysis",
                       help="Output directory for analysis")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--max_batches", type=int, default=50, 
                       help="Maximum batches for analysis")
    parser.add_argument("--create_interactive", action="store_true",
                       help="Create interactive visualizations")
    parser.add_argument("--compare_pretrained", type=str, default=None,
                       help="Path to pre-trained model for comparison")
    
    args = parser.parse_args()
    
    print("🚀 Starting Post-Training Feature Analysis")
    print(f"📁 Checkpoint: {args.checkpoint_path}")
    print(f"📊 Data: {args.data_path}")
    print(f"💾 Output: {args.output_dir}")
    
    # Initialize analyzer
    analyzer = PostTrainingFeatureAnalyzer(args.device, args.output_dir)
    
    # Load trained models
    model_states, training_results = analyzer.load_trained_models(args.checkpoint_path)
    
    # TODO: Replace with your actual model creation and data loading
    print("⚠️ Please replace the following with your actual implementations:")
    print("1. Model creation from state dictionaries")
    print("2. Data loader creation")
    
    """
    # Example integration (replace with your code):
    
    # Create models from state dicts
    encoder = YourEncoder().to(args.device)
    encoder.load_state_dict(model_states['encoder'])
    
    generator = YourGenerator().to(args.device)
    generator.load_state_dict(model_states['generator'])
    
    # Create data loader
    data_loader = create_your_data_loader(args.data_path, args.batch_size)
    
    # Extract features from trained encoder
    extraction_results = analyzer.extract_features_from_trained_encoder(
        encoder, data_loader, max_batches=args.max_batches,
        use_generator=True, generator=generator
    )
    
    features = extraction_results['features']
    phases = extraction_results['phases']
    scan_ids = extraction_results['scan_ids']
    
    # Analyze feature disentanglement
    analysis_results = analyzer.analyze_feature_disentanglement(
        features, phases, "post_sequential_training"
    )
    
    # Create interactive explorer if requested
    if args.create_interactive:
        analyzer.create_interactive_feature_explorer(
            features, phases, scan_ids, "post_sequential_training"
        )
    
    # Compare with pre-trained model if provided
    if args.compare_pretrained:
        print(f"🔄 Loading pre-trained model for comparison: {args.compare_pretrained}")
        
        # Load pre-trained model
        pretrained_states, _ = analyzer.load_trained_models(args.compare_pretrained)
        
        pretrained_encoder = YourEncoder().to(args.device)
        pretrained_encoder.load_state_dict(pretrained_states['encoder'])
        
        # Extract features from pre-trained model
        pretrained_results = analyzer.extract_features_from_trained_encoder(
            pretrained_encoder, data_loader, max_batches=args.max_batches
        )
        
        # Compare before and after training
        analyzer.compare_before_after_training(
            pretrained_results['features'], features, phases,
            "Pre-trained", "Post Sequential Training"
        )
    
    print("✅ Post-training feature analysis completed!")
    print(f"📁 Results saved to: {args.output_dir}")
    """


if __name__ == "__main__":
    main()