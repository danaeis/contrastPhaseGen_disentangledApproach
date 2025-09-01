


def download_medvit_checkpoint(model_size='small', force_download=False):
    """Download official MedViT checkpoint"""
    import urllib.request
    import os
    
    urls = {
        'tiny': 'https://github.com/Omid-Nejad/MedViT/releases/download/v1.0/medvit_tiny.pth',
        'small': 'https://github.com/Omid-Nejad/MedViT/releases/download/v1.0/medvit_small.pth', 
        'base': 'https://github.com/Omid-Nejad/MedViT/releases/download/v1.0/medvit_base.pth'
    }
    
    if model_size not in urls:
        print(f"❌ Unknown model size: {model_size}")
        return False
    
    filename = f"medvit_{model_size}_official.pth"
    url = urls[model_size]
    
    if os.path.exists(filename) and not force_download:
        print(f"✅ {filename} already exists")
        return filename
    
    try:
        print(f"📥 Downloading {filename} from official source...")
        urllib.request.urlretrieve(url, filename)
        print(f"✅ Downloaded {filename}")
        return filename
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False