import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import argparse
import pickle
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import your existing modules
from models import TimmViTEncoder
from dino_encoder import DinoV3Encoder
from medViT_encoder import create_medvit_encoder
from data import prepare_dataset_from_folders, prepare_data
from feature_visualization import (
    extract_features_from_encoder_optimized, 
    create_phase_mapping,
    apply_dimensionality_reduction_flexible,
    compute_comprehensive_metrics
)

class ContrastPhaseClassifier:
    """
    Main class for training and testing contrast phase classification using encoders + LDA
    """
    def __init__(self, encoder, encoder_name, lda_params=None, use_scaler=True):
        self.encoder = encoder
        self.encoder_name = encoder_name
        self.lda_params = lda_params or {'n_components': None}  # Will be set automatically
        self.use_scaler = use_scaler
        
        # Don't initialize LDA yet - we need to know the number of classes first
        self.lda = None
        self.scaler = StandardScaler() if use_scaler else None
        self.phase_mapping = create_phase_mapping()
        
        self.is_fitted = False
        self.training_history = {}
        
    def extract_features_batch(self, data_loader, device='cuda', use_cache=False, cache_dir='cache'):
        """Extract features from data loader"""
        print(f"Extracting features using {self.encoder_name}...")
        
        features, phases, scan_ids = extract_features_from_encoder_optimized(
            self.encoder, data_loader, device, self.encoder_name,
            use_cache=use_cache, cache_dir=cache_dir
        )
        
        return features, np.array(phases), scan_ids
    
    def fit(self, train_features, train_labels, validation_split=0.2, random_state=42):
        """
        Fit the LDA classifier on extracted features
        
        Args:
            train_features: Extracted features from encoder
            train_labels: Phase labels
            validation_split: Fraction for validation
            random_state: Random seed
        """
        print(f"Training LDA classifier...")
        print(f"Training data shape: {train_features.shape}")
        print(f"Unique phases: {np.unique(train_labels)}")
        
        # Determine optimal number of LDA components
        unique_phases = np.unique(train_labels)
        n_classes = len(unique_phases)
        max_components = min(train_features.shape[1], n_classes - 1)
        
        # Set LDA components based on the constraint
        if self.lda_params.get('n_components') is None:
            optimal_components = max_components
        else:
            requested_components = self.lda_params['n_components']
            optimal_components = min(requested_components, max_components)
        
        print(f"LDA constraint: max_components = min({train_features.shape[1]}, {n_classes}-1) = {max_components}")
        print(f"Using {optimal_components} LDA components")
        
        # Update LDA parameters and initialize LDA
        self.lda_params['n_components'] = optimal_components
        self.lda = LinearDiscriminantAnalysis(**self.lda_params)
        
        # Split into train/validation
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                train_features, train_labels, 
                test_size=validation_split, 
                stratify=train_labels,
                random_state=random_state
            )
        else:
            X_train, y_train = train_features, train_labels
            X_val, y_val = None, None
        
        # Scale features if requested
        if self.scaler is not None:
            print("Scaling features...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
        
        # Fit LDA
        print("Fitting LDA...")
        self.lda.fit(X_train_scaled, y_train)
        
        # Training metrics
        train_pred = self.lda.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        self.training_history = {
            'train_accuracy': train_accuracy,
            'train_samples': len(X_train),
            'n_features': X_train.shape[1],
            'n_components': len(self.lda.explained_variance_ratio_),
            'max_possible_components': max_components,
            'n_classes': n_classes,
            'explained_variance_ratio': self.lda.explained_variance_ratio_,
            'total_variance_explained': np.sum(self.lda.explained_variance_ratio_)
        }
        
        # Validation metrics
        if X_val is not None:
            val_pred = self.lda.predict(X_val_scaled)
            val_accuracy = accuracy_score(y_val, val_pred)
            self.training_history['val_accuracy'] = val_accuracy
            self.training_history['val_samples'] = len(X_val)
            
            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")
        else:
            print(f"Training Accuracy: {train_accuracy:.4f}")
        
        # Cross-validation
        print("Performing cross-validation...")
        try:
            cv_folds = min(5, len(np.unique(y_train)))  # Don't exceed number of classes
            cv_scores = cross_val_score(
                self.lda, X_train_scaled, y_train, 
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
                scoring='accuracy'
            )
            
            self.training_history['cv_mean'] = np.mean(cv_scores)
            self.training_history['cv_std'] = np.std(cv_scores)
            
            print(f"Cross-validation ({cv_folds}-fold): {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        except Exception as e:
            print(f"Cross-validation failed: {e}")
            self.training_history['cv_mean'] = train_accuracy
            self.training_history['cv_std'] = 0.0
        
        print(f"LDA Components Used: {len(self.lda.explained_variance_ratio_)} / {max_components} possible")
        print(f"Variance Explained: {np.sum(self.lda.explained_variance_ratio_):.4f}")
        
        # Display component breakdown
        if len(self.lda.explained_variance_ratio_) > 1:
            print("Component breakdown:")
            for i, ratio in enumerate(self.lda.explained_variance_ratio_):
                print(f"  Component {i+1}: {ratio:.4f} ({ratio*100:.1f}%)")
        
        self.is_fitted = True
        return self.training_history
    
    def predict(self, test_features):
        """Predict phases for test features"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale features if scaler was used
        if self.scaler is not None:
            test_features_scaled = self.scaler.transform(test_features)
        else:
            test_features_scaled = test_features
        
        # Predict
        predictions = self.lda.predict(test_features_scaled)
        probabilities = self.lda.predict_proba(test_features_scaled)
        
        return predictions, probabilities
    
    def evaluate(self, test_features, test_labels, detailed=True):
        """
        Comprehensive evaluation on test data
        
        Args:
            test_features: Test features
            test_labels: True test labels
            detailed: Whether to include detailed metrics
        """
        print(f"Evaluating {self.encoder_name} + LDA...")
        
        # Predict
        predictions, probabilities = self.predict(test_features)
        
        # Basic metrics
        accuracy = accuracy_score(test_labels, predictions)
        
        # Classification report
        phase_names = [self.phase_mapping.get(i, f'Phase_{i}') for i in sorted(np.unique(test_labels))]
        report = classification_report(
            test_labels, predictions, 
            target_names=phase_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities,
            'test_samples': len(test_labels)
        }
        
        if detailed:
            # Per-class metrics
            precision, recall, f1, support = precision_recall_fscore_support(test_labels, predictions)
            results['per_class_metrics'] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support
            }
            
            # Multi-class ROC AUC if possible
            try:
                if len(np.unique(test_labels)) > 2:
                    roc_auc = roc_auc_score(test_labels, probabilities, multi_class='ovr')
                    results['roc_auc'] = roc_auc
            except ValueError:
                pass  # Skip if ROC AUC can't be computed
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Samples: {len(test_labels)}")
        
        return results
    
    def plot_training_summary(self, save_path=None):
        """Plot training summary"""
        if not self.is_fitted:
            print("Model not fitted yet")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training vs Validation Accuracy
        ax1 = axes[0, 0]
        accuracies = [self.training_history['train_accuracy']]
        labels = ['Training']
        
        if 'val_accuracy' in self.training_history:
            accuracies.append(self.training_history['val_accuracy'])
            labels.append('Validation')
        
        bars = ax1.bar(labels, accuracies, alpha=0.7, color=['blue', 'orange'])
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'{self.encoder_name} + LDA: Training Performance')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Cross-validation scores
        ax2 = axes[0, 1]
        cv_mean = self.training_history['cv_mean']
        cv_std = self.training_history['cv_std']
        
        ax2.bar(['CV Accuracy'], [cv_mean], yerr=[cv_std], 
               alpha=0.7, color='green', capsize=10)
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Cross-Validation Results')
        ax2.set_ylim(0, 1)
        ax2.text(0, cv_mean + cv_std + 0.02, f'{cv_mean:.3f}±{cv_std:.3f}', 
                ha='center', va='bottom')
        
        # Explained variance ratio
        ax3 = axes[1, 0]
        components = range(1, len(self.training_history['explained_variance_ratio']) + 1)
        ax3.bar(components, self.training_history['explained_variance_ratio'], alpha=0.7)
        ax3.set_xlabel('LDA Component')
        ax3.set_ylabel('Explained Variance Ratio')
        ax3.set_title('LDA Components Explained Variance')
        
        # Cumulative explained variance
        ax4 = axes[1, 1]
        cumulative_var = np.cumsum(self.training_history['explained_variance_ratio'])
        ax4.plot(components, cumulative_var, 'o-', linewidth=2, markersize=6)
        ax4.set_xlabel('Number of Components')
        ax4.set_ylabel('Cumulative Explained Variance')
        ax4.set_title('Cumulative Explained Variance')
        ax4.grid(True, alpha=0.3)
        
        # Add total variance text
        total_var = self.training_history['total_variance_explained']
        ax4.text(0.05, 0.95, f'Total: {total_var:.3f}', 
                transform=ax4.transAxes, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.suptitle(f'Training Summary: {self.encoder_name}', fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training summary saved to: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, test_results, save_path=None):
        """Plot confusion matrix with phase names"""
        cm = test_results['confusion_matrix']
        phase_names = [self.phase_mapping.get(i, f'Phase_{i}') 
                      for i in range(len(cm))]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=phase_names, yticklabels=phase_names)
        plt.title(f'Confusion Matrix: {self.encoder_name} + LDA')
        plt.ylabel('True Phase')
        plt.xlabel('Predicted Phase')
        
        # Add accuracy text
        accuracy = test_results['accuracy']
        plt.text(len(cm)/2, len(cm) + 0.1, f'Accuracy: {accuracy:.3f}', 
                ha='center', fontsize=14, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def plot_lda_projection(self, features, labels, save_path=None, sample_size=None):
        """Plot LDA projection with phase separation"""
        if not self.is_fitted:
            print("Model not fitted yet")
            return
        
        # Scale features if scaler was used
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features
        
        # Transform to LDA space
        lda_features = self.lda.transform(features_scaled)
        
        # Sample for visualization if too many points
        if sample_size and len(features) > sample_size:
            indices = np.random.choice(len(features), sample_size, replace=False)
            lda_features = lda_features[indices]
            labels = labels[indices]
        
        # Create plots based on number of LDA components
        n_components = lda_features.shape[1]
        
        if n_components == 1:
            # 1D plot
            plt.figure(figsize=(12, 6))
            for phase in np.unique(labels):
                mask = labels == phase
                phase_name = self.phase_mapping.get(phase, f'Phase_{phase}')
                plt.scatter(lda_features[mask, 0], np.random.normal(0, 0.1, np.sum(mask)), 
                           label=phase_name, alpha=0.7)
            plt.xlabel('LDA Component 1')
            plt.ylabel('Random Jitter')
            plt.title(f'LDA Projection (1D): {self.encoder_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        elif n_components >= 2:
            # 2D or 3D plot
            if n_components == 2:
                plt.figure(figsize=(12, 8))
                for phase in np.unique(labels):
                    mask = labels == phase
                    phase_name = self.phase_mapping.get(phase, f'Phase_{phase}')
                    plt.scatter(lda_features[mask, 0], lda_features[mask, 1], 
                               label=phase_name, alpha=0.7, s=50)
                plt.xlabel('LDA Component 1')
                plt.ylabel('LDA Component 2')
                plt.title(f'LDA Projection (2D): {self.encoder_name}')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            else:  # n_components >= 3
                fig = plt.figure(figsize=(15, 5))
                
                # 2D plot
                ax1 = fig.add_subplot(121)
                for phase in np.unique(labels):
                    mask = labels == phase
                    phase_name = self.phase_mapping.get(phase, f'Phase_{phase}')
                    ax1.scatter(lda_features[mask, 0], lda_features[mask, 1], 
                               label=phase_name, alpha=0.7, s=50)
                ax1.set_xlabel('LDA Component 1')
                ax1.set_ylabel('LDA Component 2')
                ax1.set_title('LDA Projection (2D)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 3D plot
                ax2 = fig.add_subplot(122, projection='3d')
                colors = ['red', 'blue', 'green', 'orange', 'purple']
                for i, phase in enumerate(np.unique(labels)):
                    mask = labels == phase
                    phase_name = self.phase_mapping.get(phase, f'Phase_{phase}')
                    ax2.scatter(lda_features[mask, 0], lda_features[mask, 1], lda_features[mask, 2],
                               label=phase_name, alpha=0.7, s=50, c=colors[i % len(colors)])
                ax2.set_xlabel('LDA Component 1')
                ax2.set_ylabel('LDA Component 2')
                ax2.set_zlabel('LDA Component 3')
                ax2.set_title('LDA Projection (3D)')
                ax2.legend()
                
                plt.suptitle(f'LDA Projections: {self.encoder_name}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"LDA projection saved to: {save_path}")
        
        plt.show()
    
    def save_model(self, save_path):
        """Save trained model components"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'encoder_name': self.encoder_name,
            'lda_params': self.lda_params,
            'use_scaler': self.use_scaler,
            'lda': self.lda,
            'scaler': self.scaler,
            'training_history': self.training_history,
            'phase_mapping': self.phase_mapping,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {save_path}")
    
    @classmethod
    def load_model(cls, load_path, encoder):
        """Load trained model components"""
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create classifier instance
        classifier = cls(
            encoder=encoder,
            encoder_name=model_data['encoder_name'],
            lda_params=model_data['lda_params'],
            use_scaler=model_data['use_scaler']
        )
        
        # Restore trained components
        classifier.lda = model_data['lda']
        classifier.scaler = model_data['scaler']
        classifier.training_history = model_data['training_history']
        classifier.phase_mapping = model_data['phase_mapping']
        classifier.is_fitted = True
        
        print(f"Model loaded from: {load_path}")
        print(f"Model timestamp: {model_data['timestamp']}")
        
        return classifier

def create_encoder(encoder_type, config):
    """Create encoder based on type and configuration"""
    if encoder_type.lower() == 'medvit':
        return create_medvit_encoder(config)
    elif encoder_type.lower() == 'timm_vit':
        return TimmViTEncoder(**config)
    elif encoder_type.lower() == 'dinov3':
        return DinoV3Encoder(**config)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

def run_training_experiment(data_path, labels_csv, encoder_configs, args):
    """
    Run complete training experiment with multiple encoders
    
    Args:
        data_path: Path to image data
        labels_csv: Path to labels CSV file
        encoder_configs: Dictionary of encoder configurations
        args: Command line arguments
    """
    print("=" * 80)
    print("CONTRAST PHASE CLASSIFICATION TRAINING EXPERIMENT")
    print("=" * 80)
    
    # Prepare dataset
    print("Preparing dataset...")
    train_data_dicts, val_data_dicts = prepare_dataset_from_folders(
        data_path, labels_csv, validation_split=0.2, skip_prep=True
    )
    
    # Limit dataset size for debugging
    if args.max_samples_debug:
        train_data_dicts = train_data_dicts[:args.max_samples_debug]
        val_data_dicts = val_data_dicts[:args.max_samples_debug//4] if val_data_dicts else []
        print(f"DEBUG: Limited to {len(train_data_dicts)} train, {len(val_data_dicts)} val samples")
    
    # Create data loaders
    img_size = tuple(args.spatial_size)
    train_loader = prepare_data(train_data_dicts, batch_size=args.batch_size, 
                               augmentation=False, spatial_size=img_size)
    
    if val_data_dicts:
        val_loader = prepare_data(val_data_dicts, batch_size=args.batch_size, 
                                 augmentation=False, spatial_size=img_size)
    else:
        val_loader = None
    
    print(f"Training samples: {len(train_data_dicts)}")
    print(f"Validation samples: {len(val_data_dicts) if val_data_dicts else 0}")
    
    # Results storage
    all_results = {}
    
    # Train each encoder
    for encoder_name, config in encoder_configs.items():
        print(f"\n{'='*60}")
        print(f"Training {encoder_name} encoder")
        print(f"{'='*60}")
        
        try:
            # Create encoder
            encoder = create_encoder(config['type'], config['params'])
            
            # Create classifier
            lda_params = config.get('lda_params', {'n_components': 3})
            classifier = ContrastPhaseClassifier(
                encoder=encoder,
                encoder_name=encoder_name,
                lda_params=lda_params,
                use_scaler=True
            )
            
            # Extract training features
            train_features, train_labels, train_scan_ids = classifier.extract_features_batch(
                train_loader, device=args.device, use_cache=args.use_cache, 
                cache_dir=os.path.join(args.output_dir, 'feature_cache')
            )
            
            print(f"Training features shape: {train_features.shape}")
            print(f"Phase distribution: {np.bincount(train_labels)}")
            
            # Train classifier
            training_history = classifier.fit(
                train_features, train_labels, 
                validation_split=0.2, random_state=42
            )
            
            # Test on validation set if available
            if val_loader is not None:
                print("Evaluating on validation set...")
                val_features, val_labels, val_scan_ids = classifier.extract_features_batch(
                    val_loader, device=args.device, use_cache=args.use_cache,
                    cache_dir=os.path.join(args.output_dir, 'feature_cache')
                )
                
                test_results = classifier.evaluate(val_features, val_labels, detailed=True)
            else:
                # Use a portion of training data for testing (not ideal but for demo)
                test_size = min(0.3, 200 / len(train_features))  # Max 30% or 200 samples
                _, test_features, _, test_labels = train_test_split(
                    train_features, train_labels, test_size=test_size, 
                    stratify=train_labels, random_state=42
                )
                test_results = classifier.evaluate(test_features, test_labels, detailed=True)
            
            # Create visualizations
            output_subdir = os.path.join(args.output_dir, encoder_name)
            os.makedirs(output_subdir, exist_ok=True)
            
            # Training summary plot
            training_plot_path = os.path.join(output_subdir, f"{encoder_name}_training_summary.png")
            classifier.plot_training_summary(save_path=training_plot_path)
            
            # Confusion matrix
            cm_path = os.path.join(output_subdir, f"{encoder_name}_confusion_matrix.png")
            classifier.plot_confusion_matrix(test_results, save_path=cm_path)
            
            # LDA projection
            lda_proj_path = os.path.join(output_subdir, f"{encoder_name}_lda_projection.png")
            classifier.plot_lda_projection(
                train_features, train_labels, 
                save_path=lda_proj_path, sample_size=500
            )
            
            # Save model
            model_path = os.path.join(output_subdir, f"{encoder_name}_trained_model.pkl")
            classifier.save_model(model_path)
            
            # Store results
            all_results[encoder_name] = {
                'training_history': training_history,
                'test_results': test_results,
                'classifier': classifier,
                'config': config
            }
            
            print(f"✅ {encoder_name} training completed successfully!")
            print(f"   Test Accuracy: {test_results['accuracy']:.4f}")
            
        except Exception as e:
            print(f"❌ Error training {encoder_name}: {e}")
            if args.debug_mode:
                import traceback
                print(traceback.format_exc())
            continue
    
    # Create comparison plots
    create_comparison_plots(all_results, args.output_dir)
    
    # Save experiment summary
    save_experiment_summary(all_results, args.output_dir, args)
    
    return all_results

def create_comparison_plots(all_results, output_dir):
    """Create comparison plots across all encoders"""
    if not all_results:
        print("No results to compare")
        return
    
    print("Creating comparison plots...")
    
    # Extract metrics for comparison
    encoder_names = list(all_results.keys())
    train_accuracies = [results['training_history']['train_accuracy'] for results in all_results.values()]
    test_accuracies = [results['test_results']['accuracy'] for results in all_results.values()]
    cv_means = [results['training_history']['cv_mean'] for results in all_results.values()]
    cv_stds = [results['training_history']['cv_std'] for results in all_results.values()]
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    ax1 = axes[0, 0]
    x = np.arange(len(encoder_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, train_accuracies, width, label='Training', alpha=0.7)
    bars2 = ax1.bar(x + width/2, test_accuracies, width, label='Test', alpha=0.7)
    
    ax1.set_xlabel('Encoder')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training vs Test Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(encoder_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Cross-validation comparison
    ax2 = axes[0, 1]
    bars = ax2.bar(encoder_names, cv_means, yerr=cv_stds, alpha=0.7, capsize=5)
    ax2.set_xlabel('Encoder')
    ax2.set_ylabel('CV Accuracy')
    ax2.set_title('Cross-Validation Accuracy Comparison')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    for i, (bar, mean, std) in enumerate(zip(bars, cv_means, cv_stds)):
        ax2.text(bar.get_x() + bar.get_width()/2., mean + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Variance explained comparison
    ax3 = axes[1, 0]
    var_explained = [results['training_history']['total_variance_explained'] for results in all_results.values()]
    bars = ax3.bar(encoder_names, var_explained, alpha=0.7)
    ax3.set_xlabel('Encoder')
    ax3.set_ylabel('Total Variance Explained')
    ax3.set_title('LDA Variance Explained Comparison')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    for bar, var in zip(bars, var_explained):
        ax3.text(bar.get_x() + bar.get_width()/2., var + 0.005,
                f'{var:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Feature dimensions comparison
    ax4 = axes[1, 1]
    n_features = [results['training_history']['n_features'] for results in all_results.values()]
    n_components = [results['training_history']['n_components'] for results in all_results.values()]
    
    x = np.arange(len(encoder_names))
    ax4.bar(x - 0.2, n_features, 0.4, label='Input Features', alpha=0.7)
    ax4.bar(x + 0.2, n_components, 0.4, label='LDA Components', alpha=0.7)
    ax4.set_xlabel('Encoder')
    ax4.set_ylabel('Number of Dimensions')
    ax4.set_title('Feature Dimensions Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(encoder_names, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    plt.suptitle('Encoder Performance Comparison', fontsize=16, y=1.02)
    
    comparison_path = os.path.join(output_dir, 'encoder_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {comparison_path}")
    plt.show()

def save_experiment_summary(all_results, output_dir, args):
    """Save comprehensive experiment summary"""
    summary_path = os.path.join(output_dir, 'experiment_summary.json')
    
    # Prepare summary data
    summary_data = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'data_path': args.data_path,
            'batch_size': args.batch_size,
            'spatial_size': args.spatial_size,
            'device': args.device,
            'max_samples_debug': args.max_samples_debug
        },
        'results': {}
    }
    
    # Add results for each encoder
    for encoder_name, results in all_results.items():
        training_hist = results['training_history']
        test_results = results['test_results']
        
        summary_data['results'][encoder_name] = {
            'training': {
                'train_accuracy': float(training_hist['train_accuracy']),
                'val_accuracy': float(training_hist.get('val_accuracy', 0)),
                'cv_mean': float(training_hist['cv_mean']),
                'cv_std': float(training_hist['cv_std']),
                'train_samples': int(training_hist['train_samples']),
                'n_features': int(training_hist['n_features']),
                'n_components': int(training_hist['n_components']),
                'total_variance_explained': float(training_hist['total_variance_explained'])
            },
            'testing': {
                'accuracy': float(test_results['accuracy']),
                'test_samples': int(test_results['test_samples']),
                'roc_auc': float(test_results.get('roc_auc', 0)) if 'roc_auc' in test_results else None
            },
            'config': results['config']
        }
    
    # Save summary
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Also save human-readable summary
    txt_summary_path = os.path.join(output_dir, 'experiment_summary.txt')
    with open(txt_summary_path, 'w') as f:
        f.write("CONTRAST PHASE CLASSIFICATION EXPERIMENT SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Experiment Date: {summary_data['experiment_info']['timestamp']}\n")
        f.write(f"Data Path: {summary_data['experiment_info']['data_path']}\n")
        f.write(f"Device: {summary_data['experiment_info']['device']}\n\n")
        
        f.write("RESULTS SUMMARY:\n")
        f.write("-" * 30 + "\n")
        
        # Find best performing encoder
        best_encoder = max(all_results.keys(), 
                          key=lambda x: all_results[x]['test_results']['accuracy'])
        best_accuracy = all_results[best_encoder]['test_results']['accuracy']
        
        f.write(f"Best Performing Encoder: {best_encoder}\n")
        f.write(f"Best Test Accuracy: {best_accuracy:.4f}\n\n")
        
        # Detailed results for each encoder
        for encoder_name, results in all_results.items():
            f.write(f"\n{encoder_name}:\n")
            f.write(f"  Training Accuracy: {results['training_history']['train_accuracy']:.4f}\n")
            f.write(f"  Test Accuracy: {results['test_results']['accuracy']:.4f}\n")
            f.write(f"  CV Score: {results['training_history']['cv_mean']:.4f} ± {results['training_history']['cv_std']:.4f}\n")
            f.write(f"  Variance Explained: {results['training_history']['total_variance_explained']:.4f}\n")
            f.write(f"  Feature Dimensions: {results['training_history']['n_features']}\n")
            f.write(f"  LDA Components: {results['training_history']['n_components']}\n")
    
    print(f"Experiment summary saved to: {summary_path}")
    print(f"Text summary saved to: {txt_summary_path}")

def test_trained_model(model_path, encoder, test_data_path, labels_csv, args):
    """
    Test a previously trained model on new data
    
    Args:
        model_path: Path to saved model
        encoder: Encoder instance (must match the saved model)
        test_data_path: Path to test data
        labels_csv: Path to test labels CSV
        args: Command line arguments
    """
    print("=" * 60)
    print("TESTING TRAINED MODEL")
    print("=" * 60)
    
    # Load trained model
    classifier = ContrastPhaseClassifier.load_model(model_path, encoder)
    
    # Prepare test dataset
    print("Preparing test dataset...")
    test_data_dicts, _ = prepare_dataset_from_folders(
        test_data_path, labels_csv, validation_split=0.0, skip_prep=True
    )
    
    if args.max_samples_debug:
        test_data_dicts = test_data_dicts[:args.max_samples_debug]
        print(f"DEBUG: Limited test set to {len(test_data_dicts)} samples")
    
    # Create data loader
    img_size = tuple(args.spatial_size)
    test_loader = prepare_data(test_data_dicts, batch_size=args.batch_size, 
                              augmentation=False, spatial_size=img_size)
    
    print(f"Test samples: {len(test_data_dicts)}")
    
    # Extract test features
    test_features, test_labels, test_scan_ids = classifier.extract_features_batch(
        test_loader, device=args.device, use_cache=args.use_cache,
        cache_dir=os.path.join(args.output_dir, 'feature_cache')
    )
    
    print(f"Test features shape: {test_features.shape}")
    print(f"Test phase distribution: {np.bincount(test_labels)}")
    
    # Evaluate model
    test_results = classifier.evaluate(test_features, test_labels, detailed=True)
    
    # Create test-specific visualizations
    test_output_dir = os.path.join(args.output_dir, 'test_results')
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Confusion matrix
    cm_path = os.path.join(test_output_dir, f"{classifier.encoder_name}_test_confusion_matrix.png")
    classifier.plot_confusion_matrix(test_results, save_path=cm_path)
    
    # LDA projection of test data
    lda_proj_path = os.path.join(test_output_dir, f"{classifier.encoder_name}_test_lda_projection.png")
    classifier.plot_lda_projection(test_features, test_labels, save_path=lda_proj_path)
    
    # Print detailed results
    print("\nTEST RESULTS:")
    print("-" * 30)
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test Samples: {test_results['test_samples']}")
    
    if 'roc_auc' in test_results:
        print(f"ROC AUC: {test_results['roc_auc']:.4f}")
    
    # Per-class results
    report = test_results['classification_report']
    phase_mapping = classifier.phase_mapping
    
    print("\nPer-class Results:")
    for i, phase_name in phase_mapping.items():
        if str(i) in report:
            metrics = report[str(i)]
            print(f"  {phase_name}:")
            print(f"    Precision: {metrics['precision']:.3f}")
            print(f"    Recall: {metrics['recall']:.3f}")
            print(f"    F1-score: {metrics['f1-score']:.3f}")
            print(f"    Support: {metrics['support']}")
    
    # Save test results
    test_results_path = os.path.join(test_output_dir, 'test_results.json')
    test_results_serializable = {
        'accuracy': float(test_results['accuracy']),
        'test_samples': int(test_results['test_samples']),
        'classification_report': test_results['classification_report'],
        'confusion_matrix': test_results['confusion_matrix'].tolist(),
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path
    }
    
    if 'roc_auc' in test_results:
        test_results_serializable['roc_auc'] = float(test_results['roc_auc'])
    
    with open(test_results_path, 'w') as f:
        json.dump(test_results_serializable, f, indent=2)
    
    print(f"\nTest results saved to: {test_results_path}")
    
    return test_results

def main():
    """Main function for training and testing contrast phase classification"""
    parser = argparse.ArgumentParser(description="Contrast Phase Classification Training and Testing")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="data", help="Path to training data directory")
    parser.add_argument("--labels_csv", type=str, default=None, help="Path to labels CSV file")
    parser.add_argument("--test_data_path", type=str, default=None, help="Path to test data directory")
    parser.add_argument("--test_labels_csv", type=str, default=None, help="Path to test labels CSV file")
    
    # Model arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[128, 128, 128], help="Input volume size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--latent_dim", type=int, default=256, help="Latent dimension size")
    
    # Encoder-specific arguments
    parser.add_argument('--medvit_size', type=str, default='small', choices=['tiny', 'small', 'base'])
    parser.add_argument('--medvit_pretrained_path', type=str, default='pretrained_medvit_small.pth')
    parser.add_argument('--aggregation_method', type=str, default='lstm', choices=['lstm', 'attention', 'mean', 'max'])
    parser.add_argument('--max_slices', type=int, default=32)
    parser.add_argument('--timm_model_name', type=str, default='vit_small_patch16_224')
    parser.add_argument('--timm_pretrained', action='store_true', help='Use pretrained weights for Timm model')
    parser.add_argument("--dino_v3_size", type=str, default="small", choices=["small", "base", "large"])
    parser.add_argument("--dino_v3_pretrained", action="store_true", default=True, help="Use pretrained DINO v3 weights")

    # Experiment arguments
    parser.add_argument("--output_dir", type=str, default="contrast_phase_results", help="Output directory")
    parser.add_argument("--mode", type=str, choices=['train', 'test', 'both'], default='both', help="Mode: train, test, or both")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model for testing")
    
    # Debug arguments
    parser.add_argument("--debug_mode", action="store_true", help="Enable debug mode")
    parser.add_argument("--use_cache", action="store_true", default=True, help="Use cached features")
    parser.add_argument("--max_samples_debug", type=int, default=None, help="Limit samples for debugging")
    
    args = parser.parse_args()
    
    # Set up paths
    labels_csv = args.labels_csv or os.path.join(args.data_path, "labels.csv")
    test_labels_csv = args.test_labels_csv or labels_csv
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure encoders
    encoder_configs = {
        'Dino_v3': {
            'type': 'dinov3',
            'params': {
                'latent_dim': args.latent_dim,
                'model_size': getattr(args, 'dino_v3_size', 'small'),
                'pretrained': getattr(args, 'dino_v3_pretrained', True),
                'max_slices': getattr(args, 'max_slices', 32),
                'slice_sampling': 'uniform'
            },
            'lda_params': {'n_components': None}
        },
        'MedViT': {
            'type': 'medvit',
            'params': {
                'model_size': args.medvit_size,
                'pretrained_path': args.medvit_pretrained_path if os.path.exists(args.medvit_pretrained_path) else None,
                'latent_dim': args.latent_dim,
                'aggregation_method': args.aggregation_method,
                'slice_sampling': 'uniform',
                'max_slices': args.max_slices
            },
            'lda_params': {'n_components': None}  # Will be determined automatically
        },
        'TimmViT': {
            'type': 'timm_vit',
            'params': {
                'latent_dim': args.latent_dim,
                'model_name': args.timm_model_name,
                'pretrained': args.timm_pretrained,
                'max_slices': args.max_slices,
                'slice_sampling': 'uniform'
            },
            'lda_params': {'n_components': None}  # Will be determined automatically
        }
        
    }
    
    # Training mode
    if args.mode in ['train', 'both']:
        print("Starting training phase...")
        
        # Check if data exists
        if not os.path.exists(labels_csv):
            print(f"Error: Labels CSV not found at {labels_csv}")
            return
        
        # Run training experiment
        training_results = run_training_experiment(
            args.data_path, labels_csv, encoder_configs, args
        )
        
        if not training_results:
            print("No training results obtained")
            return
    
    # Testing mode
    if args.mode in ['test', 'both']:
        print("Starting testing phase...")
        
        test_data_path = args.test_data_path or args.data_path
        
        if args.model_path:
            # Test specific model
            print(f"Testing model: {args.model_path}")
            # Note: You'll need to provide the correct encoder that matches the saved model
            # This is a limitation - in practice, you'd save encoder config with the model
            print("Note: Please ensure the encoder matches the saved model")
            
        elif args.mode == 'both' and training_results:
            # Test all trained models
            print("Testing all trained models...")
            
            for encoder_name, results in training_results.items():
                model_path = os.path.join(args.output_dir, encoder_name, f"{encoder_name}_trained_model.pkl")
                
                if os.path.exists(model_path):
                    print(f"\nTesting {encoder_name} model...")
                    try:
                        # Create encoder (this is a simplification - in practice you'd need proper config handling)
                        config = encoder_configs[encoder_name]
                        encoder = create_encoder(config['type'], config['params'])
                        
                        test_results = test_trained_model(
                            model_path, encoder, test_data_path, test_labels_csv, args
                        )
                        
                        # Store test results in training results
                        training_results[encoder_name]['final_test_results'] = test_results
                        
                    except Exception as e:
                        print(f"Error testing {encoder_name}: {e}")
                        if args.debug_mode:
                            import traceback
                            print(traceback.format_exc())
        else:
            print("No models to test. Run training first or provide --model_path")
    
    print("\n" + "=" * 80)
    print("CONTRAST PHASE CLASSIFICATION EXPERIMENT COMPLETED")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()