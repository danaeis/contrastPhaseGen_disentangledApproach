import monai
from monai.data import Dataset, DataLoader
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Resized,
    ToTensord,
    RandRotate90d,
    RandFlipd,
    RandShiftIntensityd
)

def prepare_data(data_dicts, batch_size=2, augmentation=True):
    """Prepare MONAI dataset and DataLoader with optional augmentation."""
    # Basic transforms
    transforms = [
        LoadImaged(keys=["input_volume", "target_volume"]),
        EnsureChannelFirstd(keys=["input_volume", "target_volume"]),
        ScaleIntensityRanged(keys=["input_volume", "target_volume"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0),
        Resized(keys=["input_volume", "target_volume"], spatial_size=(128, 128, 128)),
        ToTensord(keys=["input_volume", "target_volume", "input_phase", "target_phase"])
    ]
    
    # Add augmentation if requested
    if augmentation:
        aug_transforms = [
            RandRotate90d(keys=["input_volume", "target_volume"], prob=0.2, spatial_axes=(0, 1)),
            RandFlipd(keys=["input_volume", "target_volume"], prob=0.2, spatial_axis=0),
            RandShiftIntensityd(keys=["input_volume", "target_volume"], prob=0.2, offsets=0.1)
        ]
        transforms.extend(aug_transforms)
    
    transform = monai.transforms.Compose(transforms)
    dataset = Dataset(data_dicts, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Optional: Registration function using SimpleITK
def register_volumes(fixed_path, moving_path, output_path=None):
    """Register moving volume to fixed volume using SimpleITK."""
    try:
        import SimpleITK as sitk
        
        # Load volumes
        fixed = sitk.ReadImage(fixed_path)
        moving = sitk.ReadImage(moving_path)
        
        # Initialize transform
        transform = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler3DTransform(), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        
        # Resample
        registered = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0)
        
        # Save if output path is provided
        if output_path:
            sitk.WriteImage(registered, output_path)
            
        return registered
    except ImportError:
        print("SimpleITK not installed. Please install with: pip install SimpleITK")
        return None

import os
import pandas as pd

def prepare_dataset_from_folders(data_root, labels_csv, validation_split=0.2, seed=42, apply_registration=False):
    # First, validate dataset
    report, pairs_df = validate_dataset(data_root, labels_csv)
    print("Validation Report:")
    print(report)
    
    if 'error' in report:
        raise ValueError(report['error'])
    
    if apply_registration:
        # Apply registration to each scan
        for scan_id in pairs_df['scan_id'].unique():
            scan_dir = os.path.join(data_root, scan_id)
            output_dir = os.path.join(data_root, f"{scan_id}_registered")
            register_case_series(scan_dir, output_dir)
            # Update paths in pairs_df to registered versions
            pairs_df.loc[pairs_df['scan_id'] == scan_id, 'input_path'] = pairs_df['input_path'].apply(
                lambda p: p.replace(scan_dir, output_dir).replace('.nii.gz', '_registered.nii.gz')
            )
            pairs_df.loc[pairs_df['scan_id'] == scan_id, 'target_path'] = pairs_df['target_path'].apply(
                lambda p: p.replace(scan_dir, output_dir).replace('.nii.gz', '_registered.nii.gz')
            )
    
    # Create data_dicts from pairs_df
    data_dicts = []
    for _, row in pairs_df.iterrows():
        data_dicts.append({
            "input_volume": row['input_path'],
            "target_volume": row['target_path'],
            "input_phase": row['input_phase'],
            "target_phase": row['target_phase'],
            "scan_id": row['scan_id']
        })
    
    # Split into train/val
    import numpy as np
    np.random.seed(seed)
    np.random.shuffle(data_dicts)
    split_idx = int(len(data_dicts) * (1 - validation_split))
    train_data_dicts = data_dicts[:split_idx]
    val_data_dicts = data_dicts[split_idx:]
    
    return train_data_dicts, val_data_dicts

import SimpleITK as sitk

def register_case_series(scan_dir, output_dir, non_contrast_file='non_contrast.nii.gz'):
    """Perform rigid registration on series of a case using non-contrast as atlas.
    
    Args:
        scan_dir: Directory of the scan_id
        output_dir: Directory to save registered volumes
        non_contrast_file: Filename of non-contrast volume
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fixed_path = os.path.join(scan_dir, non_contrast_file)
    if not os.path.exists(fixed_path):
        print(f"Non-contrast file not found: {fixed_path}")
        return
    
    fixed = sitk.ReadImage(fixed_path)
    
    nifti_files = [f for f in os.listdir(scan_dir) if f.endswith('.nii.gz') and f != non_contrast_file]
    
    for moving_file in nifti_files:
        moving_path = os.path.join(scan_dir, moving_file)
        moving = sitk.ReadImage(moving_path)
        
        # Initialize rigid transform
        transform = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler3DTransform(), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        
        # Resample
        registered = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0)
        
        # Save
        output_path = os.path.join(output_dir, f"registered_{moving_file}")
        sitk.WriteImage(registered, output_path)
        print(f"Registered {moving_file} to {output_path}")
    
    # Copy non-contrast as is
    import shutil
    shutil.copy(fixed_path, os.path.join(output_dir, non_contrast_file))