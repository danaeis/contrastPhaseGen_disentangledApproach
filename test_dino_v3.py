# test_dino_v3.py - Quick test script for DINO v3 encoder

import torch
import sys
import os

# Add your project directory to path if needed
# sys.path.append('/path/to/your/project')

def test_dino_imports():
    """Test what DINO models are actually available"""
    print("🔍 Testing available DINO/ViT models...")
    
    try:
        import torchvision
        print(f"✅ Torchvision version: {torchvision.__version__}")
        
        # Test direct DINO v3 imports
        try:
            from torchvision.models import dinov3_small
            print("✅ dinov3_small available")
        except ImportError:
            print("❌ dinov3_small not available")
            
        # Test regular ViT models
        try:
            from torchvision.models import vit_b_16, vit_l_16
            print("✅ Standard ViT models available")
        except ImportError:
            print("❌ Standard ViT models not available")
            
        # Test torch hub
        try:
            import torch
            model = torch.hub.load('facebookresearch/dino', 'dino_vits16', pretrained=False)
            print("✅ DINO models available via torch.hub")
        except Exception as e:
            print(f"❌ Torch hub DINO not available: {e}")
            
    except Exception as e:
        print(f"❌ Error testing imports: {e}")

def test_dino_encoder_minimal():
    """Minimal test of DINO encoder"""
    print("\n🧪 Testing DINO v3 Encoder...")
    
    try:
        # Copy the DinoV3Encoder class here for testing
        from dino_v3_complete import DinoV3Encoder  # This won't work, so we'll inline it
        
        # Create small test input
        test_input = torch.randn(1, 1, 16, 32, 32)  # Very small for quick test
        print(f"📊 Test input shape: {test_input.shape}")
        
        # Test small model
        encoder = DinoV3Encoder(
            latent_dim=128, 
            model_size='small', 
            pretrained=False,  # Disable pretrained for faster testing
            max_slices=8
        )
        
        with torch.no_grad():
            output = encoder(test_input)
            print(f"✅ Success! Output shape: {output.shape}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dino_imports()
    # test_dino_encoder_minimal()  # Uncomment when ready to test