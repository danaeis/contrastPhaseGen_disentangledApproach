import torch

def debug_your_checkpoint():
    """Debug the specific checkpoint you're using"""
    
    checkpoint_path = 'pretrained_medvit_small.pth'
    
    print("🔍 DEBUGGING YOUR MEDVIT CHECKPOINT")
    print("=" * 50)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Checkpoint loaded successfully")
        
        # Check main structure
        print(f"\n📦 Main checkpoint keys:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                print(f"  {key}: dict with {len(checkpoint[key])} items")
            elif hasattr(checkpoint[key], 'shape'):
                print(f"  {key}: tensor {checkpoint[key].shape}")
            else:
                print(f"  {key}: {type(checkpoint[key])}")
        
        # Get state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"\n📋 Using 'state_dict' key")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print(f"📋 Using 'model' key")
        else:
            state_dict = checkpoint
            print(f"📋 Using checkpoint directly as state_dict")
        
        print(f"📊 State dict contains {len(state_dict)} parameters")
        
        # Analyze key patterns
        print(f"\n🔍 Key patterns analysis:")
        key_patterns = {}
        
        for key in state_dict.keys():
            parts = key.split('.')
            pattern = parts[0] if parts else 'root'
            
            if pattern not in key_patterns:
                key_patterns[pattern] = []
            key_patterns[pattern].append(key)
        
        for pattern, keys in sorted(key_patterns.items()):
            print(f"  {pattern}: {len(keys)} keys")
            # Show first few examples
            for i, key in enumerate(keys[:3]):
                tensor_shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'no shape'
                print(f"    {key}: {tensor_shape}")
            if len(keys) > 3:
                print(f"    ... and {len(keys) - 3} more")
        
        # Check for common architectural components
        print(f"\n🏗️  Architecture analysis:")
        has_stem = any('stem' in key for key in state_dict.keys())
        has_features = any('features' in key for key in state_dict.keys())
        has_blocks = any('block' in key for key in state_dict.keys())
        has_layers = any('layer' in key for key in state_dict.keys())
        has_patch_embed = any('patch_embed' in key for key in state_dict.keys())
        has_attention = any('attn' in key or 'mhsa' in key or 'mhca' in key for key in state_dict.keys())
        has_norm = any(key == 'norm.weight' for key in state_dict.keys())
        has_head = any('head' in key or 'classifier' in key for key in state_dict.keys())
        
        print(f"  Stem layers: {'✅' if has_stem else '❌'}")
        print(f"  Feature layers: {'✅' if has_features else '❌'}")
        print(f"  Transformer blocks: {'✅' if has_blocks else '❌'}")
        print(f"  Regular layers: {'✅' if has_layers else '❌'}")
        print(f"  Patch embedding: {'✅' if has_patch_embed else '❌'}")
        print(f"  Attention layers: {'✅' if has_attention else '❌'}")
        print(f"  Final norm: {'✅' if has_norm else '❌'}")
        print(f"  Classification head: {'✅' if has_head else '❌'}")
        
        # Guess model type
        print(f"\n🤔 Model type guess:")
        if has_stem and has_features and has_patch_embed and has_attention:
            print("  This looks like a MedViT model ✅")
        elif has_features and has_attention:
            print("  This looks like a Vision Transformer variant")
        elif has_layers and has_attention:
            print("  This looks like a standard Transformer")
        else:
            print("  Unknown architecture - might be incompatible ❌")
        
        # Check final output dimension
        head_keys = [key for key in state_dict.keys() if 'head' in key or 'classifier' in key or 'proj_head' in key]
        if head_keys:
            for key in head_keys[:3]:  # Show first few
                if 'weight' in key:
                    shape = state_dict[key].shape
                    print(f"  Output dimension from {key}: {shape[0]} classes")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if not (has_stem and has_features):
            print("  ❌ This checkpoint doesn't match expected MedViT structure")
            print("  🔄 Try finding the correct MedViT checkpoint or train from scratch")
        else:
            print("  ✅ Checkpoint structure looks compatible with MedViT")
            print("  🔧 Use the fixed loading code I provided above")
        
        # Show sample keys for manual inspection
        print(f"\n📝 First 10 keys for manual inspection:")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'no shape'
            print(f"  {i+1:2d}. {key}: {shape}")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print(f"💡 Possible issues:")
        print(f"   - File is corrupted")
        print(f"   - Wrong file format")
        print(f"   - File doesn't exist")
        print(f"   - Insufficient memory")

if __name__ == "__main__":
    debug_your_checkpoint()