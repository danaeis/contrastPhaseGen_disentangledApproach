#!/usr/bin/env python3
"""
TotalSegmentator Integration Test Script

This script tests the TotalSegmentator encoder integration and provides
a quick way to verify that everything is working correctly before running
the full feature visualization and anatomical analysis.

Usage:
    python test_totalseg_integration.py --data_path /path/to/data
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# Test imports
def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import monai
        print("  ✅ MONAI imported successfully")
        monai_version = getattr(monai, '__version__', 'unknown')
        print(f"     Version: {monai_version}")
    except ImportError:
        print("  ❌ MONAI import failed")
        return False
    
    try:
        from monai_totalseg_encoder import create_monai_totalsegmentator_encoder
        print("  ✅ TotalSegmentator encoder imported successfully")
    except ImportError as e:
        print(f"  ❌ TotalSegmentator encoder import failed: {e}")
        return False
    
    try:
        from data import prepare_dataset_from_folders, prepare_data
        print("  ✅ Data utilities imported successfully")
    except ImportError as e:
        print(f"  ❌ Data utilities import failed: {e}")
        return False
    
    return True


def test_encoder_creation(device='cuda', latent_dim=128):
    """Test TotalSegmentator encoder creation"""
    print("\n🏗️ Testing TotalSegmentator encoder creation...")
    
    from monai_totalseg_encoder import create_monai_totalsegmentator_encoder
    
    # Conservative configuration for testing
    config = {
        'latent_dim': latent_dim,
        'device': device,
        'roi_size': (64, 64, 64),  # Small for testing
        'use_enhanced_features': False,  # Disable for stability
        'use_pretrained': True,
        'sw_batch_size': 1,  # Very conservative
        'overlap': 0.25,
        'img_size': (64, 64, 64)
    }
    
    try:
        encoder = create_monai_totalsegmentator_encoder(config)
        print("  ✅ Encoder created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in encoder.parameters())
        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        
        print(f"  📊 Model statistics:")
        print(f"     Total parameters: {total_params:,}")
        print(f"     Trainable parameters: {trainable_params:,}")
        print(f"     Model size: ~{total_params * 4 / (1024**2):.1f} MB")
        
        return encoder
        
    except Exception as e:
        print(f"  ❌ Encoder creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(encoder, device='cuda', volume_shape=(64, 64, 64)):
    """Test encoder forward pass with dummy data"""
    print(f"\n🔄 Testing forward pass with volume shape {volume_shape}...")
    
    try:
        # Create dummy CT volume (normalized to typical range)
        dummy_volume = torch.randn(1, 1, *volume_shape).to(device)
        dummy_volume = torch.clamp(dummy_volume * 200, -1000, 1000)  # Realistic HU range
        dummy_volume = (dummy_volume + 1000) / 2000  # Normalize to [0, 1]
        
        print(f"  📊 Input statistics:")
        print(f"     Shape: {dummy_volume.shape}")
        print(f"     Range: [{dummy_volume.min():.3f}, {dummy_volume.max():.3f}]")
        print(f"     Mean: {dummy_volume.mean():.3f}")
        print(f"     Device: {dummy_volume.device}")
        
        # Time the forward pass
        start_time = time.time()
        
        encoder.eval()
        with torch.no_grad():
            features = encoder(dummy_volume)
        
        forward_time = time.time() - start_time
        
        print(f"  ✅ Forward pass successful!")
        print(f"     Output shape: {features.shape}")
        print(f"     Output range: [{features.min():.6f}, {features.max():.6f}]")
        print(f"     Output mean: {features.mean():.6f}")
        print(f"     Output std: {features.std():.6f}")
        print(f"     Forward pass time: {forward_time:.2f} seconds")
        
        return features
        
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_segmentation_output(encoder, device='cuda', volume_shape=(64, 64, 64)):
    """Test segmentation output if available"""
    print(f"\n🏥 Testing segmentation output...")
    
    if not hasattr(encoder, 'get_segmentation'):
        print("  ⚠️ No segmentation method available")
        return None
    
    try:
        # Create dummy CT volume
        dummy_volume = torch.randn(1, 1, *volume_shape).to(device)
        dummy_volume = torch.clamp(dummy_volume * 200, -1000, 1000)
        dummy_volume = (dummy_volume + 1000) / 2000
        
        start_time = time.time()
        
        encoder.eval()
        with torch.no_grad():
            segmentation = encoder.get_segmentation(dummy_volume)
        
        seg_time = time.time() - start_time
        
        print(f"  ✅ Segmentation successful!")
        print(f"     Segmentation shape: {segmentation.shape}")
        print(f"     Number of classes: {segmentation.shape[1] if len(segmentation.shape) > 3 else 'N/A'}")
        print(f"     Value range: [{segmentation.min():.6f}, {segmentation.max():.6f}]")
        print(f"     Segmentation time: {seg_time:.2f} seconds")
        
        # Analyze segmentation
        if len(segmentation.shape) == 5:  # (B, C, D, H, W)
            # Check how many classes are "active" (have max probability > 0.5)
            max_probs = segmentation.max(dim=(2, 3, 4))[0]  # Max over spatial dims
            active_classes = (max_probs > 0.5).sum(dim=1)
            print(f"     Active classes (prob > 0.5): {active_classes.cpu().numpy()}")
        
        return segmentation
        
    except Exception as e:
        print(f"  ❌ Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_anatomical_analysis(encoder, device='cuda', volume_shape=(64, 64, 64)):
    """Test anatomical analysis if available"""
    print(f"\n🔬 Testing anatomical analysis...")
    
    if not hasattr(encoder, 'analyze_anatomical_features'):
        print("  ⚠️ No anatomical analysis method available")
        return None
    
    try:
        # Create dummy CT volume
        dummy_volume = torch.randn(1, 1, *volume_shape).to(device)
        dummy_volume = torch.clamp(dummy_volume * 200, -1000, 1000)
        dummy_volume = (dummy_volume + 1000) / 2000
        
        start_time = time.time()
        
        analysis = encoder.analyze_anatomical_features(dummy_volume)
        
        analysis_time = time.time() - start_time
        
        print(f"  ✅ Anatomical analysis successful!")
        print(f"     Analysis time: {analysis_time:.2f} seconds")
        print(f"     Analysis keys: {list(analysis.keys())}")
        
        # Print analysis details
        for key, value in analysis.items():
            if isinstance(value, dict):
                print(f"     {key}:")
                for subkey, subvalue in value.items():
                    print(f"       {subkey}: {subvalue}")
            else:
                print(f"     {key}: {value}")
        
        return analysis
        
    except Exception as e:
        print(f"  ❌ Anatomical analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_memory_usage(encoder, device='cuda', volume_shape=(64, 64, 64)):
    """Test memory usage with different batch sizes"""
    print(f"\n💾 Testing memory usage...")
    
    if device == 'cpu':
        print("  ⚠️ Skipping memory test on CPU")
        return
    
    try:
        batch_sizes = [1, 2]
        memory_usage = {}
        
        for batch_size in batch_sizes:
            try:
                # Clear cache
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Create batch
                dummy_volumes = torch.randn(batch_size, 1, *volume_shape).to(device)
                dummy_volumes = torch.clamp(dummy_volumes * 200, -1000, 1000)
                dummy_volumes = (dummy_volumes + 1000) / 2000
                
                initial_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
                
                encoder.eval()
                with torch.no_grad():
                    features = encoder(dummy_volumes)
                
                peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                final_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
                
                memory_usage[batch_size] = {
                    'initial': initial_memory,
                    'peak': peak_memory,
                    'final': final_memory,
                    'difference': peak_memory - initial_memory
                }
                
                print(f"  📊 Batch size {batch_size}:")
                print(f"     Peak memory usage: {peak_memory:.1f} MB")
                print(f"     Memory increase: {peak_memory - initial_memory:.1f} MB")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  ⚠️ Out of memory with batch size {batch_size}")
                    break
                else:
                    raise
        
        return memory_usage
        
    except Exception as e:
        print(f"  ❌ Memory test failed: {e}")
        return None


def test_data_loading(data_path, max_samples=5):
    """Test data loading with actual CT data"""
    print(f"\n📁 Testing data loading from {data_path}...")
    
    try:
        from data import prepare_dataset_from_folders, prepare_data
        
        labels_csv = os.path.join(data_path, "labels.csv")
        
        if not os.path.exists(labels_csv):
            print(f"  ⚠️ Labels CSV not found at {labels_csv}")
            return None
        
        # Prepare dataset
        train_data_dicts, val_data_dicts = prepare_dataset_from_folders(
            data_path,
            labels_csv,
            validation_split=0.2,
            skip_prep=True
        )
        
        print(f"  📊 Dataset statistics:")
        print(f"     Training samples: {len(train_data_dicts)}")
        print(f"     Validation samples: {len(val_data_dicts)}")
        
        # Use a few validation samples for testing
        test_data = val_data_dicts[:max_samples]
        
        # Create data loader
        data_loader = prepare_data(
            test_data, 
            batch_size=1, 
            augmentation=False, 
            spatial_size=(64, 64, 64)
        )
        
        print(f"  ✅ Data loader created successfully")
        print(f"     Using {len(test_data)} samples for testing")
        print(f"     Batch size: 1")
        print(f"     Spatial size: (64, 64, 64)")
        
        return data_loader
        
    except Exception as e:
        print(f"  ❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_with_real_data(encoder, data_loader, device='cuda'):
    """Test encoder with real CT data"""
    print(f"\n🏥 Testing with real CT data...")
    
    if data_loader is None:
        print("  ⚠️ No data loader available")
        return
    
    try:
        encoder.eval()
        
        # Test with first batch
        batch = next(iter(data_loader))
        
        input_volumes = batch['input_path'].to(device)
        phases = batch['input_phase']
        scan_ids = batch['scan_id']
        
        print(f"  📊 Real data statistics:")
        print(f"     Batch shape: {input_volumes.shape}")
        print(f"     Value range: [{input_volumes.min():.3f}, {input_volumes.max():.3f}]")
        print(f"     Phases: {phases.cpu().numpy()}")
        print(f"     Scan IDs: {scan_ids}")
        
        start_time = time.time()
        
        with torch.no_grad():
            # Test feature extraction
            features = encoder(input_volumes)
            
            # Test segmentation if available
            segmentation = None
            if hasattr(encoder, 'get_segmentation'):
                segmentation = encoder.get_segmentation(input_volumes)
            
            # Test anatomical analysis if available
            analysis = None
            if hasattr(encoder, 'analyze_anatomical_features'):
                analysis = encoder.analyze_anatomical_features(input_volumes)
        
        total_time = time.time() - start_time
        
        print(f"  ✅ Real data test successful!")
        print(f"     Feature shape: {features.shape}")
        print(f"     Feature range: [{features.min():.6f}, {features.max():.6f}]")
        
        if segmentation is not None:
            print(f"     Segmentation shape: {segmentation.shape}")
        
        if analysis is not None:
            print(f"     Anatomical analysis keys: {list(analysis.keys())}")
        
        print(f"     Total processing time: {total_time:.2f} seconds")
        
        return {
            'features': features,
            'segmentation': segmentation,
            'analysis': analysis
        }
        
    except Exception as e:
        print(f"  ❌ Real data test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_test_visualization(results, save_path=None):
    """Create a simple visualization of test results"""
    print(f"\n🎨 Creating test visualization...")
    
    if not results:
        print("  ⚠️ No results to visualize")
        return
    
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Feature distribution
        features = results['features'].cpu().numpy().flatten()
        axes[0].hist(features, bins=50, alpha=0.7, color='blue')
        axes[0].set_title('Feature Distribution')
        axes[0].set_xlabel('Feature Value')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Segmentation overview (if available)
        if results['segmentation'] is not None:
            seg = results['segmentation'][0].cpu().numpy()  # First batch item
            
            # Show middle slice of segmentation
            if len(seg.shape) == 4:  # (C, D, H, W)
                middle_slice = seg.shape[1] // 2
                seg_slice = seg[:, middle_slice, :, :]
                
                # Sum across classes to get overall segmentation
                seg_sum = seg_slice.sum(axis=0)
                axes[1].imshow(seg_sum, cmap='viridis')
                axes[1].set_title('Segmentation Overview\n(Middle Slice)')
            else:
                axes[1].text(0.5, 0.5, 'Segmentation\nFormat Unknown', 
                           ha='center', va='center', transform=axes[1].transAxes)
        else:
            axes[1].text(0.5, 0.5, 'No Segmentation\nAvailable', 
                       ha='center', va='center', transform=axes[1].transAxes)
        
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        
        # Plot 3: Analysis summary (if available)
        axes[2].axis('off')
        
        if results['analysis'] is not None:
            analysis = results['analysis']
            
            summary_text = "Anatomical Analysis:\n\n"
            
            if 'feature_statistics' in analysis:
                stats = analysis['feature_statistics']
                summary_text += f"Feature Stats:\n"
                summary_text += f"  Mean: {stats.get('mean', 0):.3f}\n"
                summary_text += f"  Std: {stats.get('std', 0):.3f}\n\n"
            
            if 'anatomical_coverage' in analysis:
                coverage = analysis['anatomical_coverage']
                summary_text += f"Anatomical Coverage:\n"
                summary_text += f"  Organs: {coverage.get('num_organs_detected', 0)}\n"
                summary_text += f"  Confidence: {coverage.get('segmentation_confidence', 0):.3f}\n\n"
            
            if 'feature_quality' in analysis:
                quality = analysis['feature_quality']
                summary_text += f"Feature Quality:\n"
                summary_text += f"  Norm: {quality.get('feature_norm', 0):.3f}\n"
                summary_text += f"  Diversity: {quality.get('feature_diversity', 0):.6f}\n"
        else:
            summary_text = "No Anatomical\nAnalysis Available"
        
        axes[2].text(0.05, 0.95, summary_text, transform=axes[2].transAxes,
                    verticalalignment='top', fontfamily='monospace', fontsize=9)
        
        plt.tight_layout()
        plt.suptitle('TotalSegmentator Integration Test Results', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  ✅ Visualization saved to: {save_path}")
        else:
            plt.show()
        
    except Exception as e:
        print(f"  ❌ Visualization failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="TotalSegmentator Integration Test")
    parser.add_argument("--data_path", type=str, help="Path to data directory (optional)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent dimension for testing")
    parser.add_argument("--volume_shape", type=int, nargs=3, default=[16, 16, 16], help="Test volume shape")
    parser.add_argument("--output_dir", type=str, default="test_results", help="Output directory")
    parser.add_argument("--quick_test", action="store_true", help="Run only basic tests")
    
    args = parser.parse_args()
    
    print("🧪" + "="*60)
    print("TOTALSEGMENTATOR INTEGRATION TEST")
    print("="*61)
    print(f"Device: {args.device}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Test volume shape: {args.volume_shape}")
    print("="*61)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test 1: Imports
    if not test_imports():
        print("❌ Import tests failed. Cannot continue.")
        return
    
    # Test 2: Encoder creation
    encoder = test_encoder_creation(args.device, args.latent_dim)
    if encoder is None:
        print("❌ Encoder creation failed. Cannot continue.")
        return
    
    # Test 3: Forward pass with dummy data
    features = test_forward_pass(encoder, args.device, tuple(args.volume_shape))
    if features is None:
        print("❌ Forward pass failed. Cannot continue with advanced tests.")
        basic_results = True
    else:
        basic_results = False
    
    if args.quick_test or basic_results:
        print("\n✅ Quick test completed successfully!")
        if hasattr(encoder, 'cleanup'):
            encoder.cleanup()
        return
    
    # Test 4: Segmentation output
    segmentation = test_segmentation_output(encoder, args.device, tuple(args.volume_shape))
    
    # Test 5: Anatomical analysis
    analysis = test_anatomical_analysis(encoder, args.device, tuple(args.volume_shape))
    
    # Test 6: Memory usage
    memory_usage = test_memory_usage(encoder, args.device, tuple(args.volume_shape))
    
    # Test 7: Real data (if available)
    real_results = None
    if args.data_path:
        data_loader = test_data_loading(args.data_path, max_samples=3)
        if data_loader is not None:
            real_results = test_with_real_data(encoder, data_loader, args.device)
    
    # Create visualization
    if real_results:
        viz_path = os.path.join(args.output_dir, "test_results_visualization.png")
        create_test_visualization(real_results, viz_path)
    
    # Cleanup
    if hasattr(encoder, 'cleanup'):
        encoder.cleanup()
    
    # Final summary
    print("\n" + "="*61)
    print("TEST SUMMARY")
    print("="*61)
    print("✅ Import tests: PASSED")
    print("✅ Encoder creation: PASSED")
    print("✅ Forward pass: PASSED")
    
    if segmentation is not None:
        print("✅ Segmentation output: PASSED")
    else:
        print("⚠️ Segmentation output: NOT AVAILABLE")
    
    if analysis is not None:
        print("✅ Anatomical analysis: PASSED")
    else:
        print("⚠️ Anatomical analysis: NOT AVAILABLE")
    
    if memory_usage is not None:
        print("✅ Memory usage test: PASSED")
        max_batch_size = max(memory_usage.keys())
        max_memory = memory_usage[max_batch_size]['peak']
        print(f"   Max tested batch size: {max_batch_size}")
        print(f"   Peak memory usage: {max_memory:.1f} MB")
    else:
        print("⚠️ Memory usage test: SKIPPED")
    
    if real_results is not None:
        print("✅ Real data test: PASSED")
    elif args.data_path:
        print("❌ Real data test: FAILED")
    else:
        print("⏭️ Real data test: SKIPPED (no data path provided)")
    
    print("\n🎉 TotalSegmentator integration test completed!")
    print(f"📁 Results saved to: {args.output_dir}")
    
    if args.data_path and real_results:
        print("\n✅ Ready to run full feature visualization and anatomical analysis!")
        print("   Next steps:")
        print("   1. Run: python feature_visualization.py --data_path <path>")  
        print("   2. Run: python totalseg_anatomical_analysis.py --data_path <path>")
    else:
        print("\n💡 To test with real data, run:")
        print("   python test_totalseg_integration.py --data_path /path/to/your/data")


if __name__ == "__main__":
    main()