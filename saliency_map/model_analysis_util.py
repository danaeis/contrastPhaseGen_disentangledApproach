#!/usr/bin/env python3
"""
Model analysis and comparison utilities for contrast phase classification
Provides tools for analyzing model performance, comparing approaches, and generating insights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, cohen_kappa_score
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle
import json
import os
from pathlib import Path
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class ModelPerformanceAnalyzer:
    """
    Comprehensive model performance analysis for contrast phase classification
    """
    
    def __init__(self, models_dict, phase_mapping=None):
        """
        Args:
            models_dict: Dictionary of {model_name: model} pairs
            phase_mapping: Mapping from class indices to phase names
        """
        self.models = models_dict
        self.phase_mapping = phase_mapping or {
            0: 'Non-contrast', 1: 'Arterial', 2: 'Venous', 3: 'Delayed', 4: 'Hepatobiliary'
        }
        self.results = {}
        
    def evaluate_models(self, test_loader, device='cuda'):
        """
        Evaluate all models on test data
        
        Args:
            test_loader: Test data loader
            device: Device for evaluation
        
        Returns:
            results: Dictionary with evaluation results for each model
        """
        print("Evaluating models on test data...")
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            
            model.eval()
            model = model.to(device)
            
            all_predictions = []
            all_probabilities = []
            all_labels = []
            all_features = []
            
            with torch.no_grad():
                for batch_data in test_loader:
                    # Get data
                    if isinstance(batch_data, dict):
                        images = batch_data['image'].to(device)
                        labels = batch_data['label'].to(device)
                    else:
                        images, labels = batch_data
                        images = images.to(device)
                        labels = labels.to(device)
                    
                    # Ensure labels are correct format
                    if labels.dim() > 1:
                        labels = labels.squeeze()
                    labels = labels.long()
                    
                    # Get predictions
                    logits = model(images)
                    probabilities = F.softmax(logits, dim=1)
                    predictions = torch.argmax(logits, dim=1)
                    
                    # Extract features from encoder
                    features = model.encoder(images)
                    
                    # Store results
                    all_predictions.extend(predictions.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_features.extend(features.cpu().numpy())
            
            # Convert to numpy arrays
            predictions = np.array(all_predictions)
            probabilities = np.array(all_probabilities)
            labels = np.array(all_labels)
            features = np.array(all_features)
            
            # Calculate metrics
            metrics = self._calculate_comprehensive_metrics(labels, predictions, probabilities)
            
            # Store results
            self.results[model_name] = {
                'predictions': predictions,
                'probabilities': probabilities,
                'labels': labels,
                'features': features,
                'metrics': metrics
            }
            
            print(f"  {model_name} Accuracy: {metrics['accuracy']:.4f}")
            print(f"  {model_name} F1-Score: {metrics['macro_f1']:.4f}")
        
        return self.results
    
    def _calculate_comprehensive_metrics(self, labels, predictions, probabilities):
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = np.mean(predictions == labels)
        
        # Per-class and macro metrics
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        metrics['per_class_precision'] = precision.tolist()
        metrics['per_class_recall'] = recall.tolist()
        metrics['per_class_f1'] = f1.tolist()
        metrics['per_class_support'] = support.tolist()
        
        # Macro averages
        metrics['macro_precision'] = np.mean(precision)
        metrics['macro_recall'] = np.mean(recall)
        metrics['macro_f1'] = np.mean(f1)
        
        # Weighted averages
        metrics['weighted_precision'] = np.average(precision, weights=support)
        metrics['weighted_recall'] = np.average(recall, weights=support)
        metrics['weighted_f1'] = np.average(f1, weights=support)
        
        # Cohen's Kappa
        metrics['cohen_kappa'] = cohen_kappa_score(labels, predictions)
        
        # ROC AUC (multi-class)
        try:
            metrics['roc_auc_ovr'] = roc_auc_score(labels, probabilities, multi_class='ovr')
            metrics['roc_auc_ovo'] = roc_auc_score(labels, probabilities, multi_class='ovo')
        except ValueError:
            metrics['roc_auc_ovr'] = None
            metrics['roc_auc_ovo'] = None
        
        # Average Precision Score
        try:
            # Convert to one-vs-rest format
            n_classes = len(np.unique(labels))
            y_true_binary = np.eye(n_classes)[labels]
            avg_precision_scores = []
            for i in range(n_classes):
                ap = average_precision_score(y_true_binary[:, i], probabilities[:, i])
                avg_precision_scores.append(ap)
            metrics['average_precision_scores'] = avg_precision_scores
            metrics['mean_average_precision'] = np.mean(avg_precision_scores)
        except:
            metrics['average_precision_scores'] = None
            metrics['mean_average_precision'] = None
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(labels, predictions).tolist()
        
        return metrics
    
    def create_comparison_report(self, save_dir=None):
        """
        Create comprehensive comparison report
        
        Args:
            save_dir: Directory to save report and plots
        
        Returns:
            report: Comparison report dictionary
        """
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_models() first.")
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        print("Creating model comparison report...")
        
        # Extract metrics for comparison
        model_names = list(self.results.keys())
        comparison_metrics = self._extract_comparison_metrics()
        
        # Create comparison plots
        if save_dir:
            self._plot_accuracy_comparison(comparison_metrics, save_dir / 'accuracy_comparison.png')
            self._plot_per_class_performance(comparison_metrics, save_dir / 'per_class_performance.png')
            self._plot_confusion_matrices(save_dir / 'confusion_matrices.png')
            self._plot_roc_curves(save_dir / 'roc_curves.png')
            self._plot_feature_analysis(save_dir / 'feature_analysis.png')
        
        # Statistical significance tests
        significance_tests = self._perform_significance_tests()
        
        # Feature analysis
        feature_analysis = self._analyze_features()
        
        # Create report
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_evaluated': model_names,
            'comparison_metrics': comparison_metrics,
            'significance_tests': significance_tests,
            'feature_analysis': feature_analysis,
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        if save_dir:
            report_path = save_dir / 'model_comparison_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Also create human-readable report
            self._create_readable_report(report, save_dir / 'model_comparison_report.txt')
        
        return report
    
    def _extract_comparison_metrics(self):
        """Extract metrics for comparison"""
        metrics_dict = {}
        
        for model_name, results in self.results.items():
            metrics = results['metrics']
            metrics_dict[model_name] = {
                'accuracy': metrics['accuracy'],
                'macro_f1': metrics['macro_f1'],
                'macro_precision': metrics['macro_precision'],
                'macro_recall': metrics['macro_recall'],
                'cohen_kappa': metrics['cohen_kappa'],
                'roc_auc_ovr': metrics.get('roc_auc_ovr'),
                'mean_ap': metrics.get('mean_average_precision'),
                'per_class_f1': metrics['per_class_f1']
            }
        
        return metrics_dict
    
    def _plot_accuracy_comparison(self, comparison_metrics, save_path):
        """Plot accuracy comparison across models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        model_names = list(comparison_metrics.keys())
        
        # Overall accuracy
        accuracies = [comparison_metrics[name]['accuracy'] for name in model_names]
        axes[0, 0].bar(model_names, accuracies, alpha=0.7)
        axes[0, 0].set_title('Overall Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        for i, acc in enumerate(accuracies):
            axes[0, 0].text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
        
        # F1 scores
        f1_scores = [comparison_metrics[name]['macro_f1'] for name in model_names]
        axes[0, 1].bar(model_names, f1_scores, alpha=0.7, color='orange')
        axes[0, 1].set_title('Macro F1-Score')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        for i, f1 in enumerate(f1_scores):
            axes[0, 1].text(i, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom')
        
        # Cohen's Kappa
        kappa_scores = [comparison_metrics[name]['cohen_kappa'] for name in model_names]
        axes[1, 0].bar(model_names, kappa_scores, alpha=0.7, color='green')
        axes[1, 0].set_title("Cohen's Kappa")
        axes[1, 0].set_ylabel('Kappa')
        axes[1, 0].tick_params(axis='x', rotation=45)
        for i, kappa in enumerate(kappa_scores):
            axes[1, 0].text(i, kappa + 0.01, f'{kappa:.3f}', ha='center', va='bottom')
        
        # ROC AUC
        roc_aucs = [comparison_metrics[name]['roc_auc_ovr'] for name in model_names if comparison_metrics[name]['roc_auc_ovr']]
        if roc_aucs:
            valid_names = [name for name in model_names if comparison_metrics[name]['roc_auc_ovr']]
            axes[1, 1].bar(valid_names, roc_aucs, alpha=0.7, color='red')
            axes[1, 1].set_title('ROC AUC (OvR)')
            axes[1, 1].set_ylabel('AUC')
            axes[1, 1].tick_params(axis='x', rotation=45)
            for i, auc in enumerate(roc_aucs):
                axes[1, 1].text(i, auc + 0.01, f'{auc:.3f}', ha='center', va='bottom')
        else:
            axes[1, 1].text(0.5, 0.5, 'ROC AUC\nNot Available', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('ROC AUC (OvR)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_performance(self, comparison_metrics, save_path):
        """Plot per-class performance comparison"""
        n_classes = len(list(comparison_metrics.values())[0]['per_class_f1'])
        model_names = list(comparison_metrics.keys())
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create grouped bar plot
        x = np.arange(n_classes)
        width = 0.8 / len(model_names)
        
        for i, model_name in enumerate(model_names):
            f1_scores = comparison_metrics[model_name]['per_class_f1']
            ax.bar(x + i * width, f1_scores, width, label=model_name, alpha=0.7)
        
        # Customize plot
        ax.set_xlabel('Contrast Phase')
        ax.set_ylabel('F1-Score')
        ax.set_title('Per-Class F1-Score Comparison')
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels([self.phase_mapping.get(i, f'Phase_{i}') for i in range(n_classes)])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self, save_path):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(self.results.items()):
            cm = np.array(results['metrics']['confusion_matrix'])
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot
            im = axes[i].imshow(cm_normalized, interpolation='nearest', cmap='Blues')
            axes[i].set_title(f'{model_name}\nAccuracy: {results["metrics"]["accuracy"]:.3f}')
            
            # Add text annotations
            thresh = cm_normalized.max() / 2.
            for row in range(cm.shape[0]):
                for col in range(cm.shape[1]):
                    axes[i].text(col, row, f'{cm[row, col]}\n({cm_normalized[row, col]:.2f})',
                               ha="center", va="center",
                               color="white" if cm_normalized[row, col] > thresh else "black")
            
            # Set labels
            phase_names = [self.phase_mapping.get(j, f'P{j}') for j in range(cm.shape[0])]
            axes[i].set_xticks(range(cm.shape[1]))
            axes[i].set_yticks(range(cm.shape[0]))
            axes[i].set_xticklabels(phase_names, rotation=45)
            axes[i].set_yticklabels(phase_names)
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, save_path):
        """Plot ROC curves for all models"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot for each class (One-vs-Rest)
        n_classes = len(self.phase_mapping)
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.results)))
        
        for model_idx, (model_name, results) in enumerate(self.results.items()):
            labels = results['labels']
            probabilities = results['probabilities']
            
            # Convert to one-vs-rest format
            y_true_binary = np.eye(n_classes)[labels]
            
            # Plot ROC for each class
            for class_idx in range(n_classes):
                if np.sum(y_true_binary[:, class_idx]) > 0:  # Only if class exists in test set
                    fpr, tpr, _ = roc_curve(y_true_binary[:, class_idx], probabilities[:, class_idx])
                    auc_score = roc_auc_score(y_true_binary[:, class_idx], probabilities[:, class_idx])
                    
                    if model_idx == 0:  # Only label classes once
                        label = f'{self.phase_mapping.get(class_idx, f"Class {class_idx}")}'
                    else:
                        label = None
                    
                    axes[0].plot(fpr, tpr, color=colors[model_idx], alpha=0.7, 
                               linestyle='-' if class_idx == 0 else '--', label=label)
        
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curves by Class')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot average ROC for each model
        for model_name, results in self.results.items():
            if results['metrics'].get('roc_auc_ovr'):
                auc = results['metrics']['roc_auc_ovr']
                # For visualization, plot a simple curve (this is approximate)
                fpr = np.linspace(0, 1, 100)
                tpr = np.power(fpr, 1 / (2 * auc))  # Approximate curve
                axes[1].plot(fpr, tpr, label=f'{model_name} (AUC: {auc:.3f})')
        
        axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('Average ROC Curves by Model')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_analysis(self, save_path):
        """Plot feature analysis using dimensionality reduction"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Collect all features and labels
        all_features = []
        all_labels = []
        all_model_ids = []
        
        for model_idx, (model_name, results) in enumerate(self.results.items()):
            all_features.append(results['features'])
            all_labels.append(results['labels'])
            all_model_ids.extend([model_idx] * len(results['labels']))
        
        # Combine features (use first model for main analysis)
        main_features = all_features[0]
        main_labels = all_labels[0]
        
        # PCA
        if main_features.shape[1] > 2:
            pca = PCA(n_components=2)
            pca_features = pca.fit_transform(main_features)
            
            # Plot PCA colored by true labels
            scatter = axes[0, 0].scatter(pca_features[:, 0], pca_features[:, 1], 
                                       c=main_labels, cmap='tab10', alpha=0.7)
            axes[0, 0].set_title(f'PCA - True Labels\nExplained Variance: {pca.explained_variance_ratio_.sum():.3f}')
            axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
            axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=axes[0, 0])
            cbar.set_label('True Phase')
        
        # t-SNE (if enough samples)
        if len(main_features) > 30:
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(main_features)//4))
                tsne_features = tsne.fit_transform(main_features)
                
                scatter = axes[0, 1].scatter(tsne_features[:, 0], tsne_features[:, 1], 
                                           c=main_labels, cmap='tab10', alpha=0.7)
                axes[0, 1].set_title('t-SNE - True Labels')
                axes[0, 1].set_xlabel('t-SNE 1')
                axes[0, 1].set_ylabel('t-SNE 2')
                
                cbar = plt.colorbar(scatter, ax=axes[0, 1])
                cbar.set_label('True Phase')
            except:
                axes[0, 1].text(0.5, 0.5, 't-SNE\nFailed', ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # Feature importance (if available)
        try:
            feature_std = np.std(main_features, axis=0)
            top_features = np.argsort(feature_std)[-20:]  # Top 20 most variable features
            
            axes[1, 0].bar(range(len(top_features)), feature_std[top_features])
            axes[1, 0].set_title('Feature Importance (Top 20)')
            axes[1, 0].set_xlabel('Feature Index')
            axes[1, 0].set_ylabel('Standard Deviation')
        except:
            axes[1, 0].text(0.5, 0.5, 'Feature Importance\nNot Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Model comparison in feature space (if multiple models)
        if len(self.results) > 1:
            # Use PCA features from first model
            model_names = list(self.results.keys())
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            
            for i, model_name in enumerate(model_names[:5]):  # Limit to 5 models
                results = self.results[model_name]
                if len(results['features']) == len(pca_features):
                    # Color by prediction correctness
                    correct = results['predictions'] == results['labels']
                    axes[1, 1].scatter(pca_features[correct, 0], pca_features[correct, 1], 
                                     c=colors[i], alpha=0.7, marker='o', s=20, 
                                     label=f'{model_name} Correct')
                    axes[1, 1].scatter(pca_features[~correct, 0], pca_features[~correct, 1], 
                                     c=colors[i], alpha=0.7, marker='x', s=20, 
                                     label=f'{model_name} Wrong')
            
            axes[1, 1].set_title('Model Predictions in Feature Space')
            axes[1, 1].set_xlabel('PC1')
            axes[1, 1].set_ylabel('PC2')
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            axes[1, 1].text(0.5, 0.5, 'Model Comparison\nRequires Multiple Models', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _perform_significance_tests(self):
        """Perform statistical significance tests between models"""
        if len(self.results) < 2:
            return "Significance tests require at least 2 models"
        
        significance_tests = {}
        model_names = list(self.results.keys())
        
        # McNemar's test for paired predictions
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]
                
                pred1 = self.results[model1]['predictions']
                pred2 = self.results[model2]['predictions']
                labels = self.results[model1]['labels']
                
                # Create contingency table
                model1_correct = pred1 == labels
                model2_correct = pred2 == labels
                
                # McNemar's test contingency table
                both_correct = np.sum(model1_correct & model2_correct)
                model1_only = np.sum(model1_correct & ~model2_correct)
                model2_only = np.sum(~model1_correct & model2_correct)
                both_wrong = np.sum(~model1_correct & ~model2_correct)
                
                # McNemar's test statistic
                if model1_only + model2_only > 0:
                    mcnemar_stat = (abs(model1_only - model2_only) - 1)**2 / (model1_only + model2_only)
                    p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
                else:
                    mcnemar_stat = 0
                    p_value = 1.0
                
                significance_tests[f'{model1}_vs_{model2}'] = {
                    'mcnemar_statistic': float(mcnemar_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'contingency_table': {
                        'both_correct': int(both_correct),
                        'model1_only_correct': int(model1_only),
                        'model2_only_correct': int(model2_only),
                        'both_wrong': int(both_wrong)
                    }
                }
        
        return significance_tests
    
    def _analyze_features(self):
        """Analyze feature distributions and separability"""
        feature_analysis = {}
        
        for model_name, results in self.results.items():
            features = results['features']
            labels = results['labels']
            
            # Feature statistics
            feature_analysis[model_name] = {
                'feature_dim': features.shape[1],
                'feature_mean': float(np.mean(features)),
                'feature_std': float(np.std(features)),
                'feature_range': [float(np.min(features)), float(np.max(features))]
            }
            
            # Class separability (Fisher's criterion for each feature)
            if len(np.unique(labels)) > 1:
                fisher_scores = []
                for feature_idx in range(features.shape[1]):
                    feature_vals = features[:, feature_idx]
                    
                    # Calculate between-class and within-class variance
                    overall_mean = np.mean(feature_vals)
                    between_class_var = 0
                    within_class_var = 0
                    
                    for class_label in np.unique(labels):
                        class_mask = labels == class_label
                        class_features = feature_vals[class_mask]
                        class_mean = np.mean(class_features)
                        class_size = len(class_features)
                        
                        between_class_var += class_size * (class_mean - overall_mean)**2
                        within_class_var += np.sum((class_features - class_mean)**2)
                    
                    # Fisher's criterion
                    if within_class_var > 0:
                        fisher_score = between_class_var / within_class_var
                    else:
                        fisher_score = 0
                    
                    fisher_scores.append(fisher_score)
                
                feature_analysis[model_name]['mean_fisher_score'] = float(np.mean(fisher_scores))
                feature_analysis[model_name]['max_fisher_score'] = float(np.max(fisher_scores))
        
        return feature_analysis
    
    def _generate_recommendations(self):
        """Generate recommendations based on analysis"""
        if not self.results:
            return []
        
        recommendations = []
        
        # Find best performing model
        best_model = max(self.results.keys(), 
                        key=lambda x: self.results[x]['metrics']['accuracy'])
        best_accuracy = self.results[best_model]['metrics']['accuracy']
        
        recommendations.append(f"Best performing model: {best_model} (Accuracy: {best_accuracy:.4f})")
        
        # Check for overfitting/underfitting
        for model_name, results in self.results.items():
            accuracy = results['metrics']['accuracy']
            if accuracy < 0.6:
                recommendations.append(f"{model_name}: Low accuracy suggests underfitting. Consider larger model or more training.")
            elif accuracy > 0.95:
                recommendations.append(f"{model_name}: Very high accuracy might indicate overfitting. Validate on independent dataset.")
        
        # Feature analysis recommendations
        feature_analyses = self._analyze_features()
        for model_name, analysis in feature_analyses.items():
            if analysis.get('mean_fisher_score', 0) < 1.0:
                recommendations.append(f"{model_name}: Low feature separability. Consider feature engineering or different encoder.")
        
        # Class imbalance check
        first_model = list(self.results.keys())[0]
        labels = self.results[first_model]['labels']
        class_counts = np.bincount(labels)
        if np.max(class_counts) / np.min(class_counts) > 3:
            recommendations.append("Class imbalance detected. Consider class weighting or data augmentation.")
        
        return recommendations
    
    def _create_readable_report(self, report, save_path):
        """Create human-readable text report"""
        with open(save_path, 'w') as f:
            f.write("MODEL COMPARISON REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {report['timestamp']}\n")
            f.write(f"Models evaluated: {', '.join(report['models_evaluated'])}\n\n")
            
            # Performance comparison
            f.write("PERFORMANCE COMPARISON:\n")
            f.write("-" * 30 + "\n")
            for model_name, metrics in report['comparison_metrics'].items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"  Macro F1: {metrics['macro_f1']:.4f}\n")
                f.write(f"  Cohen's Kappa: {metrics['cohen_kappa']:.4f}\n")
                if metrics['roc_auc_ovr']:
                    f.write(f"  ROC AUC: {metrics['roc_auc_ovr']:.4f}\n")
            
            # Recommendations
            f.write("\nRECOMMENDATIONS:\n")
            f.write("-" * 30 + "\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
            
            # Significance tests
            if isinstance(report['significance_tests'], dict):
                f.write("\nSTATISTICAL SIGNIFICANCE:\n")
                f.write("-" * 30 + "\n")
                for comparison, test_result in report['significance_tests'].items():
                    f.write(f"{comparison}: ")
                    if test_result['significant']:
                        f.write(f"Significant difference (p={test_result['p_value']:.4f})\n")
                    else:
                        f.write(f"No significant difference (p={test_result['p_value']:.4f})\n")


def compare_with_lda_baseline(mlp_results, lda_results_path=None):
    """
    Compare MLP results with LDA baseline
    
    Args:
        mlp_results: Results from MLP models
        lda_results_path: Path to saved LDA results
    
    Returns:
        comparison: Comparison results
    """
    comparison = {
        'mlp_results': mlp_results,
        'comparison_summary': {}
    }
    
    if lda_results_path and os.path.exists(lda_results_path):
        # Load LDA results
        with open(lda_results_path, 'rb') as f:
            lda_results = pickle.load(f)
        
        comparison['lda_results'] = lda_results
        
        # Compare performance
        for mlp_model_name, mlp_result in mlp_results.items():
            mlp_accuracy = mlp_result['metrics']['accuracy']
            
            # Find corresponding LDA result (simplified matching)
            lda_accuracy = 0
            for lda_model_name, lda_result in lda_results.items():
                if any(encoder_name in mlp_model_name for encoder_name in ['MedViT', 'TimmViT', 'DinoV3']):
                    if any(encoder_name in lda_model_name for encoder_name in ['MedViT', 'TimmViT', 'DinoV3']):
                        lda_accuracy = lda_result.get('final_val_acc', 0)
                        break
            
            improvement = mlp_accuracy - lda_accuracy
            comparison['comparison_summary'][mlp_model_name] = {
                'mlp_accuracy': mlp_accuracy,
                'lda_accuracy': lda_accuracy,
                'improvement': improvement,
                'relative_improvement': improvement / lda_accuracy if lda_accuracy > 0 else float('inf')
            }
    
    return comparison


# Example usage
def example_model_analysis():
    """Example of how to use the model analysis tools"""
    print("Model Analysis Example")
    print("=" * 50)
    
    # This would typically be called after training your models
    # models_dict = {
    #     'MedViT': trained_medvit_model,
    #     'TimmViT': trained_timm_model,
    #     'DinoV3': trained_dino_model
    # }
    # 
    # analyzer = ModelPerformanceAnalyzer(models_dict)
    # results = analyzer.evaluate_models(test_loader)
    # report = analyzer.create_comparison_report(save_dir='analysis_results')
    
    print("This is an example of how to use ModelPerformanceAnalyzer")
    print("1. Create analyzer with your trained models")
    print("2. Evaluate on test data")
    print("3. Generate comprehensive comparison report")
    print("4. Review recommendations and statistical tests")


if __name__ == "__main__":
    example_model_analysis()