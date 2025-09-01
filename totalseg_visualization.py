import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import argparse
from tqdm import tqdm

from typing import Dict, List, Tuple, Optional, Any
import warnings
from pathlib import Path

# Import project modules
from data import prepare_dataset_from_folders, prepare_data

# Import TotalSegmentator encoder with fallback
try:
    from monai_totalseg_encoder import create_monai_totalsegmentator_encoder
    TOTALSEG_AVAILABLE = True
    print("✅ TotalSegmentator encoder available for anatomical analysis")
except ImportError:
    TOTALSEG_AVAILABLE = False
    print("❌ TotalSegmentator encoder not available")
    exit(1)


class TotalSegmentatorAnalyzer:
    """
    Comprehensive analyzer for TotalSegmentator outputs with anatomical insights
    """
    
    def __init__(self, encoder, device='cuda'):
        """
        Args:
            encoder: TotalSegmentator encoder instance
            device: Device to run analysis on
        """
        self.encoder = encoder
        self.device = device
        
        # Standard TotalSegmentator organ mapping (104 classes)
        self.organ_mapping = self._create_organ_mapping()
        self.phase_mapping = {
            0: 'Arterial',
            1: 'Venous', 
            2: 'Delayed',
            3: 'Non-contrast'
        }
        
    def _create_organ_mapping(self):
        """
        Create comprehensive organ mapping for TotalSegmentator
        Based on the 104-class TotalSegmentator v2 model
        """
        return {
            # Major organs
            'liver': [1],
            'spleen': [2], 
            'pancreas': [3],
            'right_kidney': [4],
            'left_kidney': [5],
            'stomach': [6],
            'gallbladder': [7],
            'esophagus': [8],
            'thyroid': [9],
            'prostate': [10],  # Male only
            'uterus': [11],    # Female only
            
            # Cardiovascular system
            'heart': [12, 13, 14, 15],  # Multiple heart chambers
            'aorta': [16, 17, 18, 19],  # Ascending, descending, arch, etc.
            'pulmonary_artery': [20, 21],
            'vena_cava': [22, 23],  # Superior and inferior
            'portal_vein': [24],
            'hepatic_vein': [25],
            
            # Respiratory system  
            'lung_right': [26, 27, 28],  # Right upper, middle, lower lobe
            'lung_left': [29, 30],       # Left upper, lower lobe
            'trachea': [31],
            
            # Digestive system
            'duodenum': [32],
            'small_bowel': [33],
            'colon': [34, 35, 36, 37],  # Different segments
            
            # Musculoskeletal system (vertebrae)
            'vertebrae': list(range(38, 62)),  # C1-L5 vertebrae
            'ribs': list(range(62, 86)),       # 24 ribs
            'sternum': [86],
            'clavicula': [87, 88],  # Left and right
            
            # Other structures
            'brain': [89],
            'skull': [90],
            'mandible': [91],
            'spinal_cord': [92],
            'urinary_bladder': [93],
            'adrenal_glands': [94, 95],  # Left and right
            
            # Additional structures (up to 104 classes)
            'other_organs': list(range(96, 105))
        }
    
    def extract_comprehensive_data(self, data_loader, max_samples=None):
        """
        Extract comprehensive data including features, segmentations, and anatomical analyses
        
        Args:
            data_loader: DataLoader with CT volumes
            max_samples: Maximum number of samples to process
            
        Returns:
            Dictionary with all extracted data
        """
        self.encoder.eval()
        
        all_features = []
        all_segmentations = []
        all_anatomical_analyses = []
        all_phases = []
        all_scan_ids = []
        all_volumes = []
        
        print("🔍 Extracting comprehensive TotalSegmentator data...")
        
        sample_count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing volumes")):
                if max_samples and sample_count >= max_samples:
                    break
                    
                input_volumes = batch['input_path'].to(self.device)
                input_phases = batch['input_phase']
                scan_ids = batch['scan_id']
                
                batch_size = input_volumes.shape[0]
                
                try:
                    # Extract latent features
                    features = self.encoder(input_volumes)
                    all_features.append(features.cpu().numpy())
                    
                    # Extract segmentation masks
                    segmentations = self.encoder.get_segmentation(input_volumes)
                    all_segmentations.append(segmentations.cpu().numpy())
                    
                    # Analyze anatomical features for each volume
                    for i in range(batch_size):
                        volume = input_volumes[i:i+1]
                        analysis = self.encoder.analyze_anatomical_features(volume)
                        all_anatomical_analyses.append(analysis)
                        
                        # Store metadata
                        phase_label = input_phases[i].item() if torch.is_tensor(input_phases[i]) else input_phases[i]
                        all_phases.append(phase_label)
                        all_scan_ids.append(scan_ids[i])
                        all_volumes.append(volume.cpu().numpy())
                        
                        sample_count += 1
                        if max_samples and sample_count >= max_samples:
                            break
                            
                except Exception as e:
                    print(f"⚠️ Error processing batch {batch_idx}: {e}")
                    continue
        
        # Concatenate arrays
        features = np.vstack(all_features) if all_features else np.array([])
        segmentations = np.concatenate(all_segmentations, axis=0) if all_segmentations else np.array([])
        volumes = np.concatenate(all_volumes, axis=0) if all_volumes else np.array([])
        
        return {
            'features': features,
            'segmentations': segmentations,
            'anatomical_analyses': all_anatomical_analyses,
            'phases': all_phases,
            'scan_ids': all_scan_ids,
            'volumes': volumes
        }
    
    def analyze_organ_volumes_by_phase(self, data):
        """
        Analyze organ volumes across different contrast phases
        
        Args:
            data: Data dictionary from extract_comprehensive_data
            
        Returns:
            Analysis results dictionary
        """
        print("\n🏥 Analyzing organ volumes by contrast phase...")
        
        segmentations = data['segmentations']
        phases = data['phases']
        
        # Organize data by phase
        phase_organ_volumes = {}
        
        for phase_id in np.unique(phases):
            phase_name = self.phase_mapping.get(phase_id, f'Phase_{phase_id}')
            phase_mask = np.array(phases) == phase_id
            phase_segmentations = segmentations[phase_mask]
            
            organ_volumes = {}
            for organ_name, organ_ids in self.organ_mapping.items():
                total_volume = 0
                sample_count = 0
                
                for seg in phase_segmentations:
                    volume_for_organ = 0
                    for organ_id in organ_ids:
                        if organ_id < seg.shape[1]:  # Check if class exists
                            volume_for_organ += np.sum(seg[0, organ_id] > 0.5)  # Threshold at 0.5
                    
                    if volume_for_organ > 0:  # Only count if organ is detected
                        total_volume += volume_for_organ
                        sample_count += 1
                
                # Average volume for this organ in this phase
                organ_volumes[organ_name] = {
                    'mean_volume': total_volume / sample_count if sample_count > 0 else 0,
                    'detection_rate': sample_count / len(phase_segmentations) if len(phase_segmentations) > 0 else 0,
                    'total_samples': len(phase_segmentations)
                }
            
            phase_organ_volumes[phase_name] = organ_volumes
        
        return phase_organ_volumes
    
    def create_organ_detection_heatmap(self, organ_analysis, save_path=None):
        """
        Create heatmap showing organ detection rates across phases
        
        Args:
            organ_analysis: Results from analyze_organ_volumes_by_phase
            save_path: Path to save the plot
        """
        print("📊 Creating organ detection heatmap...")
        
        # Prepare data for heatmap
        phases = list(organ_analysis.keys())
        organs = list(self.organ_mapping.keys())
        
        # Filter to organs that are commonly detected
        common_organs = ['liver', 'spleen', 'pancreas', 'right_kidney', 'left_kidney', 
                        'stomach', 'heart', 'lung_right', 'lung_left', 'aorta']
        
        detection_matrix = []
        for organ in common_organs:
            row = []
            for phase in phases:
                detection_rate = organ_analysis[phase][organ]['detection_rate']
                row.append(detection_rate)
            detection_matrix.append(row)
        
        detection_matrix = np.array(detection_matrix)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(detection_matrix, 
                   xticklabels=phases,
                   yticklabels=common_organs,
                   annot=True, 
                   fmt='.2f',
                   cmap='YlOrRd',
                   vmin=0, 
                   vmax=1,
                   cbar_kws={'label': 'Detection Rate'})
        
        plt.title('Organ Detection Rates Across Contrast Phases\nTotalSegmentator Analysis', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Contrast Phase')
        plt.ylabel('Anatomical Organ')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Organ detection heatmap saved to: {save_path}")
        
        plt.show()
        
        return detection_matrix
    
    def create_phase_specific_anatomical_maps(self, data, save_dir=None, num_examples=3):
        """
        Create anatomical overlay visualizations for each phase
        
        Args:
            data: Data dictionary from extract_comprehensive_data
            save_dir: Directory to save visualizations
            num_examples: Number of examples per phase to visualize
        """
        print("🎨 Creating phase-specific anatomical maps...")
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        volumes = data['volumes']
        segmentations = data['segmentations']
        phases = data['phases']
        scan_ids = data['scan_ids']
        
        # Create visualizations for each phase
        for phase_id in np.unique(phases):
            phase_name = self.phase_mapping.get(phase_id, f'Phase_{phase_id}')
            phase_indices = [i for i, p in enumerate(phases) if p == phase_id]
            
            # Select examples for this phase
            selected_indices = phase_indices[:num_examples]
            
            for idx, data_idx in enumerate(selected_indices):
                volume = volumes[data_idx, 0]  # Remove channel dimension
                segmentation = segmentations[data_idx]
                scan_id = scan_ids[data_idx]
                
                # Create multi-slice visualization
                self._create_anatomical_overlay_visualization(
                    volume, segmentation, phase_name, scan_id, 
                    save_path=os.path.join(save_dir, f"{phase_name}_{scan_id}_anatomical_overlay.png") if save_dir else None
                )
    
    def _create_anatomical_overlay_visualization(self, volume, segmentation, phase_name, scan_id, save_path=None):
        """
        Create anatomical overlay visualization for a single volume
        
        Args:
            volume: 3D CT volume
            segmentation: 3D segmentation mask
            phase_name: Name of contrast phase
            scan_id: Scan identifier
            save_path: Path to save visualization
        """
        # Select representative slices
        depth = volume.shape[0]
        slice_indices = [depth//4, depth//2, 3*depth//4]  # 25%, 50%, 75%
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Anatomical Segmentation Overlay - {phase_name} Phase\nScan: {scan_id}', 
                    fontsize=14, fontweight='bold')
        
        # Color map for different organs
        colors = plt.cm.Set3(np.linspace(0, 1, 12))
        organ_colors = {
            'liver': colors[0],
            'heart': colors[1], 
            'lung_right': colors[2],
            'lung_left': colors[3],
            'kidney_right': colors[4],
            'kidney_left': colors[5],
            'spleen': colors[6],
            'pancreas': colors[7],
            'stomach': colors[8],
            'aorta': colors[9]
        }
        
        for i, slice_idx in enumerate(slice_indices):
            # Original CT slice
            axes[0, i].imshow(volume[slice_idx], cmap='gray', vmin=-200, vmax=200)
            axes[0, i].set_title(f'CT Slice {slice_idx}')
            axes[0, i].axis('off')
            
            # Overlay segmentation
            axes[1, i].imshow(volume[slice_idx], cmap='gray', vmin=-200, vmax=200)
            
            # Add organ overlays with transparency
            overlay = np.zeros((*volume[slice_idx].shape, 3))
            
            for organ_name, organ_ids in self.organ_mapping.items():
                if organ_name in organ_colors:
                    color = organ_colors[organ_name][:3]  # RGB only
                    
                    for organ_id in organ_ids:
                        if organ_id < segmentation.shape[0]:
                            mask = segmentation[organ_id, slice_idx] > 0.5
                            if np.any(mask):
                                for c in range(3):
                                    overlay[mask, c] = color[c]
            
            axes[1, i].imshow(overlay, alpha=0.4)
            axes[1, i].set_title(f'Anatomical Overlay {slice_idx}')
            axes[1, i].axis('off')
        
        # Add legend
        legend_elements = []
        for organ_name, color in organ_colors.items():
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                            markerfacecolor=color, markersize=8, label=organ_name))
        
        fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.15, 0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def analyze_contrast_enhancement_patterns(self, data):
        """
        Analyze how contrast enhancement affects different anatomical regions
        
        Args:
            data: Data dictionary from extract_comprehensive_data
            
        Returns:
            Enhancement analysis results
        """
        print("\n💉 Analyzing contrast enhancement patterns...")
        
        volumes = data['volumes']
        segmentations = data['segmentations']
        phases = data['phases']
        
        # Group data by scan_id to track the same anatomy across phases
        scan_phase_data = {}
        
        for i, (phase, scan_id) in enumerate(zip(phases, data['scan_ids'])):
            if scan_id not in scan_phase_data:
                scan_phase_data[scan_id] = {}
            
            scan_phase_data[scan_id][phase] = {
                'volume': volumes[i],
                'segmentation': segmentations[i],
                'index': i
            }
        
        # Analyze enhancement for scans that have multiple phases
        enhancement_analysis = {}
        
        for scan_id, phase_data in scan_phase_data.items():
            if len(phase_data) < 2:  # Need at least 2 phases to compare
                continue
            
            # Find non-contrast and contrast phases
            has_noncontrast = 3 in phase_data  # Non-contrast is phase 3
            contrast_phases = [p for p in phase_data.keys() if p != 3]
            
            if not has_noncontrast or not contrast_phases:
                continue
            
            noncontrast_volume = phase_data[3]['volume'][0]  # Remove batch dimension
            noncontrast_seg = phase_data[3]['segmentation']
            
            scan_analysis = {}
            
            for contrast_phase in contrast_phases:
                contrast_volume = phase_data[contrast_phase]['volume'][0]
                phase_name = self.phase_mapping[contrast_phase]
                
                # Calculate enhancement for each organ
                organ_enhancements = {}
                
                for organ_name, organ_ids in self.organ_mapping.items():
                    organ_mask = np.zeros_like(noncontrast_volume, dtype=bool)
                    
                    for organ_id in organ_ids:
                        if organ_id < noncontrast_seg.shape[0]:
                            organ_mask |= (noncontrast_seg[organ_id] > 0.5)
                    
                    if np.any(organ_mask):
                        # Calculate mean intensity in organ region
                        noncontrast_intensity = np.mean(noncontrast_volume[organ_mask])
                        contrast_intensity = np.mean(contrast_volume[organ_mask])
                        
                        enhancement = contrast_intensity - noncontrast_intensity
                        relative_enhancement = enhancement / (np.abs(noncontrast_intensity) + 1e-6)
                        
                        organ_enhancements[organ_name] = {
                            'absolute_enhancement': enhancement,
                            'relative_enhancement': relative_enhancement,
                            'noncontrast_intensity': noncontrast_intensity,
                            'contrast_intensity': contrast_intensity
                        }
                
                scan_analysis[phase_name] = organ_enhancements
            
            enhancement_analysis[scan_id] = scan_analysis
        
        return enhancement_analysis
    
    def create_enhancement_analysis_plots(self, enhancement_analysis, save_path=None):
        """
        Create plots showing contrast enhancement patterns
        
        Args:
            enhancement_analysis: Results from analyze_contrast_enhancement_patterns
            save_path: Path to save the plot
        """
        print("📈 Creating contrast enhancement analysis plots...")
        
        if not enhancement_analysis:
            print("⚠️ No enhancement data available for plotting")
            return
        
        # Aggregate enhancement data across all scans
        phase_organ_enhancements = {}
        
        for scan_data in enhancement_analysis.values():
            for phase_name, organ_data in scan_data.items():
                if phase_name not in phase_organ_enhancements:
                    phase_organ_enhancements[phase_name] = {}
                
                for organ_name, enhancement_data in organ_data.items():
                    if organ_name not in phase_organ_enhancements[phase_name]:
                        phase_organ_enhancements[phase_name][organ_name] = []
                    
                    phase_organ_enhancements[phase_name][organ_name].append(
                        enhancement_data['relative_enhancement']
                    )
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Enhancement patterns for key organs
        key_organs = ['liver', 'spleen', 'pancreas', 'heart', 'aorta', 'kidney_right']
        phases = list(phase_organ_enhancements.keys())
        
        x = np.arange(len(key_organs))
        width = 0.25
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, phase_name in enumerate(phases):
            enhancements = []
            for organ in key_organs:
                if organ in phase_organ_enhancements[phase_name]:
                    values = phase_organ_enhancements[phase_name][organ]
                    enhancements.append(np.mean(values) if values else 0)
                else:
                    enhancements.append(0)
            
            ax1.bar(x + i * width, enhancements, width, label=phase_name, 
                   color=colors[i % len(colors)], alpha=0.8)
        
        ax1.set_xlabel('Anatomical Organs')
        ax1.set_ylabel('Mean Relative Enhancement')
        ax1.set_title('Contrast Enhancement by Organ and Phase')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(key_organs, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 2: Enhancement distribution for liver (most enhanced organ)
        if 'liver' in phase_organ_enhancements.get(list(phases)[0], {}):
            liver_data = []
            liver_labels = []
            
            for phase_name in phases:
                if 'liver' in phase_organ_enhancements[phase_name]:
                    liver_enhancements = phase_organ_enhancements[phase_name]['liver']
                    if liver_enhancements:
                        liver_data.append(liver_enhancements)
                        liver_labels.append(phase_name)
            
            if liver_data:
                bp = ax2.boxplot(liver_data, labels=liver_labels, patch_artist=True)
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax2.set_ylabel('Relative Enhancement')
                ax2.set_title('Liver Enhancement Distribution')
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Enhancement analysis plot saved to: {save_path}")
        
        plt.show()
        
        return phase_organ_enhancements
    
    def generate_comprehensive_report(self, data, save_path=None):
        """
        Generate comprehensive anatomical analysis report
        
        Args:
            data: Data dictionary from extract_comprehensive_data
            save_path: Path to save the report
        """
        print("\n📋 Generating comprehensive anatomical analysis report...")
        
        # Run all analyses
        organ_analysis = self.analyze_organ_volumes_by_phase(data)
        enhancement_analysis = self.analyze_contrast_enhancement_patterns(data)
        
        # Generate report content
        report = []
        report.append("=" * 80)
        report.append("TOTALSEGMENTATOR ANATOMICAL ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Dataset overview
        report.append("DATASET OVERVIEW")
        report.append("-" * 40)
        report.append(f"Total samples: {len(data['phases'])}")
        report.append(f"Unique scans: {len(set(data['scan_ids']))}")
        report.append(f"Feature dimension: {data['features'].shape[1] if len(data['features'].shape) > 1 else 'N/A'}")
        report.append("")
        
        # Phase distribution
        phase_counts = pd.Series(data['phases']).value_counts().sort_index()
        report.append("PHASE DISTRIBUTION")
        report.append("-" * 40)
        for phase_num, count in phase_counts.items():
            phase_name = self.phase_mapping.get(phase_num, f'Phase_{phase_num}')
            percentage = (count / len(data['phases'])) * 100
            report.append(f"{phase_name}: {count} samples ({percentage:.1f}%)")
        report.append("")
        
        # Organ detection analysis
        report.append("ORGAN DETECTION ANALYSIS")
        report.append("-" * 40)
        
        key_organs = ['liver', 'spleen', 'pancreas', 'heart', 'lung_right', 'lung_left']
        
        for organ in key_organs:
            report.append(f"\n{organ.upper().replace('_', ' ')}:")
            
            for phase_name, organ_data in organ_analysis.items():
                if organ in organ_data:
                    detection_rate = organ_data[organ]['detection_rate']
                    mean_volume = organ_data[organ]['mean_volume']
                    report.append(f"  {phase_name}: {detection_rate:.1%} detection rate, "
                                f"avg volume: {mean_volume:.0f} voxels")
        
        report.append("")
        
        # Enhancement analysis (if available)
        if enhancement_analysis:
            report.append("CONTRAST ENHANCEMENT ANALYSIS")
            report.append("-" * 40)
            
            # Calculate average enhancements across scans
            avg_enhancements = {}
            for scan_data in enhancement_analysis.values():
                for phase_name, organ_data in scan_data.items():
                    if phase_name not in avg_enhancements:
                        avg_enhancements[phase_name] = {}
                    
                    for organ_name, enhancement_data in organ_data.items():
                        if organ_name not in avg_enhancements[phase_name]:
                            avg_enhancements[phase_name][organ_name] = []
                        
                        avg_enhancements[phase_name][organ_name].append(
                            enhancement_data['relative_enhancement']
                        )
            
            # Report key findings
            for organ in key_organs:
                if any(organ in phase_data for phase_data in avg_enhancements.values()):
                    report.append(f"\n{organ.upper().replace('_', ' ')} ENHANCEMENT:")
                    
                    for phase_name, organ_data in avg_enhancements.items():
                        if organ in organ_data and organ_data[organ]:
                            mean_enhancement = np.mean(organ_data[organ])
                            std_enhancement = np.std(organ_data[organ])
                            report.append(f"  {phase_name}: {mean_enhancement:+.2f} ± {std_enhancement:.2f}")
        
        report.append("")
        report.append("ANALYSIS COMPLETE")
        report.append("=" * 80)
        
        # Join and print/save report
        report_text = "\n".join(report)
        print(report_text)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"\n📄 Comprehensive report saved to: {save_path}")
        
        return report_text


def main():
    parser = argparse.ArgumentParser(description="TotalSegmentator Anatomical Analysis and Visualization")
    parser.add_argument("--data_path", type=str, default="data", help="Path to data directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (keep low for memory)")
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[96, 96, 96], help="Input volume size D H W")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--output_dir", type=str, default="totalseg_analysis", help="Output directory")
    parser.add_argument("--latent_dim", type=int, default=256, help="Latent dimension size")
    parser.add_argument("--max_samples", type=int, default=50, help="Maximum samples to analyze")
    
    # TotalSegmentator specific arguments
    parser.add_argument('--totalseg_roi_size', type=int, nargs=3, default=[64, 64, 64], help='ROI size')
    parser.add_argument('--totalseg_enhanced', action='store_true', help='Use enhanced features')
    
    args = parser.parse_args()
    
    if not TOTALSEG_AVAILABLE:
        print("❌ TotalSegmentator not available. Please install required dependencies.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("📁 Preparing dataset...")
    
    # Path to labels.csv
    labels_csv = os.path.join(args.data_path, "labels.csv")
    
    if not os.path.exists(labels_csv):
        print(f"❌ Error: labels.csv not found at {labels_csv}")
        return
    
    # Prepare dataset
    train_data_dicts, val_data_dicts = prepare_dataset_from_folders(
        args.data_path,
        labels_csv,
        validation_split=0.2,
        skip_prep=True
    )
    
    # Use validation data and limit samples
    viz_data = val_data_dicts[:args.max_samples] if len(val_data_dicts) > args.max_samples else val_data_dicts
    print(f"✅ Using {len(viz_data)} samples for analysis")
    
    # Create data loader
    img_size = tuple(args.spatial_size)
    data_loader = prepare_data(viz_data, batch_size=args.batch_size, 
                              augmentation=False, spatial_size=img_size)
    
    # Initialize TotalSegmentator encoder
    print("🏗️ Initializing TotalSegmentator encoder...")
    
    totalseg_config = {
        'latent_dim': args.latent_dim,
        'device': args.device,
        'roi_size': tuple(args.totalseg_roi_size),
        'use_enhanced_features': args.totalseg_enhanced,
        'use_pretrained': True,
        'sw_batch_size': 1,
        'overlap': 0.25,
        'img_size': img_size
    }
    
    try:
        encoder = create_monai_totalsegmentator_encoder(totalseg_config)
        analyzer = TotalSegmentatorAnalyzer(encoder, args.device)
        
        # Extract comprehensive data
        print("\n🔍 Extracting comprehensive anatomical data...")
        data = analyzer.extract_comprehensive_data(data_loader, max_samples=args.max_samples)
        
        if len(data['phases']) == 0:
            print("❌ No data was successfully extracted")
            return
        
        # Perform analyses and create visualizations
        
        # 1. Organ volume analysis
        print("\n📊 Creating organ detection heatmap...")
        organ_analysis = analyzer.analyze_organ_volumes_by_phase(data)
        heatmap_path = os.path.join(args.output_dir, "organ_detection_heatmap.png")
        analyzer.create_organ_detection_heatmap(organ_analysis, heatmap_path)
        
        # 2. Anatomical overlay visualizations
        print("\n🎨 Creating anatomical overlay visualizations...")
        overlay_dir = os.path.join(args.output_dir, "anatomical_overlays")
        analyzer.create_phase_specific_anatomical_maps(data, overlay_dir, num_examples=2)
        
        # 3. Contrast enhancement analysis
        print("\n💉 Analyzing contrast enhancement patterns...")
        enhancement_analysis = analyzer.analyze_contrast_enhancement_patterns(data)
        
        if enhancement_analysis:
            enhancement_path = os.path.join(args.output_dir, "contrast_enhancement_analysis.png")
            analyzer.create_enhancement_analysis_plots(enhancement_analysis, enhancement_path)
        else:
            print("⚠️ Insufficient data for enhancement analysis (need multiple phases per scan)")
        
        # 4. Generate comprehensive report
        print("\n📋 Generating comprehensive report...")
        report_path = os.path.join(args.output_dir, "anatomical_analysis_report.txt")
        analyzer.generate_comprehensive_report(data, report_path)
        
        # Cleanup
        if hasattr(encoder, 'cleanup'):
            encoder.cleanup()
        
        print(f"\n🎉 TotalSegmentator anatomical analysis complete!")
        print(f"📂 All results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()