#!/usr/bin/env python3
"""
Comprehensive test script for all encoder implementations
Run this before training to ensure all encoders work properly
"""

import torch
import torch.nn as nn
import sys
import traceback
from typing import Dict, Any

def test_encoder(encoder_name: str, encoder_class, config: Dict[str, Any], input_shape=(1, 1, 64, 128, 128)):
    """
    Test a single encoder implementation
    
    Args:
        encoder_name: Name of the encoder
        encoder_class: Encoder class to test
        config: Configuration dictionary
        input_shape: Input tensor shape
    
    Returns:
        dict: Test results
    """
    
    print(f"\n{'='*60}")
    print(f"TESTING {encoder_name.upper()} ENCODER")
    print(f"{'='*60}")
    
    results = {
        'name': encoder_name,
        'success': False,
        'output_shape': None,
        'latent_dim': None,
        'error': None,
        'memory_usage': None
    }
    
    try:
        # Create encoder
        print(f"Creating encoder with config: {config}")
        encoder = encoder_class(**config)
        
        # Count parameters
        total_params = sum(p.numel() for p in encoder.parameters())
        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
        
        # Create test input
        dummy_input = torch.randn(*input_shape)
        print(f"Input shape: {dummy_input.shape}")
        
        # Test forward pass
        encoder.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                encoder = encoder.cuda()
                dummy_input = dummy_input.cuda()
                
                # Measure memory
                torch.cuda.reset_peak_memory_stats()
                
            output = encoder(dummy_input)
            
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                results['memory_usage'] = f"{memory_used:.1f} MB"
                torch.cuda.empty_cache()
        
        print(f"Output shape: {output.shape}")
        print(f"Expected latent dimension: {config.get('latent_dim', 'unknown')}")
        print(f"Actual output dimension: {output.shape[-1]}")
        
        if torch.cuda.is_available():
            print(f"GPU memory usage: {results['memory_usage']}")
        
        # Verify output shape
        expected_latent_dim = config.get('latent_dim', None)
        if expected_latent_dim and output.shape[-1] != expected_latent_dim:
            print(f"⚠️  WARNING: Expected latent_dim {expected_latent_dim}, got {output.shape[-1]}")
        
        # Check for NaN or inf
        if torch.isnan(output).any():
            print("❌ ERROR: Output contains NaN values")
        elif torch.isinf(output).any():
            print("❌ ERROR: Output contains Inf values")
        else:
            print("✅ Output values are valid")
        
        results['success'] = True
        results['output_shape'] = tuple(output.shape)
        results['latent_dim'] = output.shape[-1]
        
        print(f"✅ {encoder_name} encoder test PASSED")
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        results['error'] = error_msg
        print(f"❌ {encoder_name} encoder test FAILED")
        print(f"Error: {error_msg}")
        print("Traceback:")
        traceback.print_exc()
    
    return results


def test_all_encoders():
    """Test all available encoder implementations"""
    
    print("COMPREHENSIVE ENCODER TESTING")
    print("=" * 80)
    
    # Common test configuration
    base_config = {
        'latent_dim': 256,
        'in_channels': 1
    }
    
    test_input_shape = (2, 1, 32, 128, 128)  # Small volume for testing
    
    # Test configurations for each encoder
    encoder_tests = []
    
    # 1. Simple 3D CNN
    try:
        from models import Simple3DCNNEncoder
        encoder_tests.append({
            'name': 'simple_cnn',
            'class': Simple3DCNNEncoder,
            'config': {
                **base_config,
                'img_size': (128, 128, 128)
            }
        })
    except ImportError as e:
        print(f"❌ Cannot import Simple3DCNNEncoder: {e}")
    
    # 2. timm ViT
    try:
        from models import TimmViTEncoder
        encoder_tests.append({
            'name': 'timm_vit',
            'class': TimmViTEncoder,
            'config': {
                'latent_dim': 256,
                'model_name': 'vit_tiny_patch16_224',  # Use tiny for faster testing
                'pretrained': False  # Avoid download during testing
            }
        })
    except ImportError as e:
        print(f"❌ Cannot import TimmViTEncoder: {e}")
    
    # 3. ResNet3D
    try:
        from models import ResNet3DEncoder
        encoder_tests.append({
            'name': 'resnet3d',
            'class': ResNet3DEncoder,
            'config': base_config
        })
    except ImportError as e:
        print(f"❌ Cannot import ResNet3DEncoder: {e}")
    
    # 4. Hybrid
    try:
        from models import LightweightHybridEncoder
        encoder_tests.append({
            'name': 'hybrid',
            'class': LightweightHybridEncoder,
            'config': base_config
        })
    except ImportError as e:
        print(f"❌ Cannot import LightweightHybridEncoder: {e}")
    
    # 5. MedViT
    try:
        from arc_medViT_encoder import create_medvit_encoder
        
        # Special handling for MedViT factory function
        def medvit_wrapper(**config):
            return create_medvit_encoder(config)
        
        encoder_tests.append({
            'name': 'medvit',
            'class': medvit_wrapper,
            'config': {
                'model_size': 'small',
                'pretrained_path': None,  # No pretrained for testing
                'latent_dim': 256,
                'aggregation_method': 'mean',  # Faster than LSTM
                'slice_sampling': 'uniform',
                'max_slices': 8  # Small number for testing
            }
        })
    except ImportError as e:
        print(f"❌ Cannot import MedViT encoder: {e}")
    
    # Run tests
    results = []
    for test_config in encoder_tests:
        result = test_encoder(
            test_config['name'],
            test_config['class'],
            test_config['config'],
            test_input_shape
        )
        results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    if successful:
        print(f"\n📊 WORKING ENCODERS:")
        print("-" * 40)
        for result in successful:
            print(f"  {result['name']:<12} | Output: {result['output_shape']} | Latent: {result['latent_dim']}")
            if result['memory_usage']:
                print(f"               | Memory: {result['memory_usage']}")
    
    if failed:
        print(f"\n💥 FAILED ENCODERS:")
        print("-" * 40)
        for result in failed:
            print(f"  {result['name']:<12} | Error: {result['error']}")
    
    return results


def test_generator_compatibility(encoder_results):
    """Test generator compatibility with encoder outputs"""
    
    print(f"\n{'='*60}")
    print("TESTING GENERATOR COMPATIBILITY")
    print(f"{'='*60}")
    
    try:
        from models import Generator
        
        for result in encoder_results:
            if not result['success']:
                continue
                
            encoder_latent_dim = result['latent_dim']
            phase_dim = 32
            # total_dim = encoder_latent_dim + phase_dim
            
            try:
                # Test generator creation
                generator = Generator(
                    latent_dim=encoder_latent_dim,
                    phase_dim=phase_dim,
                    output_shape=(64, 64, 64, 1)  # Small for testing
                )
                
                # Test forward pass
                dummy_z = torch.randn(1, encoder_latent_dim)
                dummy_phase = torch.randn(1, phase_dim)
                
                with torch.no_grad():
                    output = generator(dummy_z, dummy_phase)
                
                print(f"✅ {result['name']:<12} | Generator compatible | Output: {output.shape}")
                
            except Exception as e:
                print(f"❌ {result['name']:<12} | Generator ERROR: {e}")
                
    except ImportError:
        print("❌ Cannot import Generator for compatibility testing")


def create_fixed_training_schedule():
    """Provide fixed training schedule based on test results"""
    
    schedule = """
    RECOMMENDED TRAINING SCHEDULE (Based on Test Results):
    =====================================================
    
    Phase 1: Extended Phase Detector Pre-training (Epochs 1-30)
    ───────────────────────────────────────────────────────────
    • Train ONLY phase detector
    • No gradient reversal (lambda = 0)
    • Higher learning rate: 1e-3
    • Target: Phase accuracy > 70%
    
    Phase 2: Generator Training (Epochs 31-60)
    ──────────────────────────────────────────
    • Train encoder + generator + discriminator
    • Phase detector frozen
    • Learning rates: encoder 1e-4, generator 1e-4, discriminator 1e-4
    • Target: Generator loss < 0.3
    
    Phase 3: Gentle Disentanglement (Epochs 61-90)
    ──────────────────────────────────────────────
    • Add phase detector with light gradient reversal
    • Lambda: 0.0 → 0.3 gradually
    • Monitor phase accuracy (may drop temporarily)
    • Target: Maintain generation quality
    
    Phase 4: Full Disentanglement (Epochs 91-150)
    ─────────────────────────────────────────────
    • Full gradient reversal
    • Lambda: 0.3 → 1.0
    • Balance all losses
    • Target: High generation quality + good disentanglement
    
    Key Changes from Current Training:
    • 3x longer phase detector pre-training
    • Separated generator and disentanglement phases
    • Gradual lambda introduction
    • Higher initial learning rates
    """
    
    return schedule


if __name__ == "__main__":
    # Run comprehensive tests
    results = test_all_encoders()
    
    # Test generator compatibility
    test_generator_compatibility(results)
    
    # Show recommended training schedule
    # print(create_fixed_training_schedule())
    
    # Final recommendation
    successful_encoders = [r['name'] for r in results if r['success']]
    
    if successful_encoders:
        print(f"\n🎯 RECOMMENDATION:")
        print(f"Use one of these working encoders: {', '.join(successful_encoders)}")
        print(f"Start with 'simple_cnn' for fastest training/debugging")
    else:
        print(f"\n❌ No encoders working. Check your environment and dependencies.")