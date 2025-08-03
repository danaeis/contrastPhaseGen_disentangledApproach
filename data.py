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

def prepare_dataset_from_folders(data_root, labels_csv, validation_split=0.2, seed=42):
    """Prepare dataset from folders with scan_ids containing NIfTI files and a labels.csv file.
    
    Args:
        data_root: Root directory containing scan_id folders
        labels_csv: Path to CSV file with phase labels
        validation_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        train_data_dicts: List of dictionaries for training
        val_data_dicts: List of dictionaries for validation
    """
    # Load labels
    labels_df = pd.read_csv(labels_csv)
    
    # Organize volumes by scan_id and phase
    volumes_by_scan_id = {}
    
    # Map phase names to integers
    phase_to_int = {
        'arterial': 0,
        'venous': 1,
        'delayed': 2,
        'non-contrast': 3
    }
    
    # Process each scan_id folder
    scan_ids = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    
    for scan_id in scan_ids:
        scan_dir = os.path.join(data_root, scan_id)
        nifti_files = [f for f in os.listdir(scan_dir) if f.endswith('.nii.gz')]
        
        # Get phase labels for this scan_id from the CSV
        scan_labels = labels_df[labels_df['scan_id'] == scan_id]
        
        if len(scan_labels) == 0:
            print(f"Warning: No labels found for scan_id {scan_id}")
            continue
        
        volumes_by_scan_id[scan_id] = {}
        
        for nifti_file in nifti_files:
            file_path = os.path.join(scan_dir, nifti_file)
            
            # Extract phase from filename or use label from CSV
            # Assuming filename contains phase information or can be matched with CSV
            for _, row in scan_labels.iterrows():
                if row['filename'] in nifti_file:
                    phase = row['phase']
                    phase_int = phase_to_int.get(phase.lower(), None)
                    
                    if phase_int is not None:
                        volumes_by_scan_id[scan_id][phase_int] = file_path
                    else:
                        print(f"Warning: Unknown phase '{phase}' for file {nifti_file}")
    
    # Create input-target pairs for training
    data_dicts = []
    
    for scan_id, phases in volumes_by_scan_id.items():
        # Only use scans that have at least 2 phases
        if len(phases) < 2:
            continue
        
        # Create all possible pairs of phases
        phase_ids = list(phases.keys())
        for i, input_phase in enumerate(phase_ids):
            for target_phase in phase_ids[i+1:]:  # Only use phases after the current one
                data_dicts.append({
                    "input_volume": volumes_by_scan_id[scan_id][input_phase],
                    "target_volume": volumes_by_scan_id[scan_id][target_phase],
                    "input_phase": input_phase,
                    "target_phase": target_phase,
                    "scan_id": scan_id
                })
                
                # Also add the reverse direction
                data_dicts.append({
                    "input_volume": volumes_by_scan_id[scan_id][target_phase],
                    "target_volume": volumes_by_scan_id[scan_id][input_phase],
                    "input_phase": target_phase,
                    "target_phase": input_phase,
                    "scan_id": scan_id
                })
    
    # Split into training and validation sets
    import numpy as np
    np.random.seed(seed)
    np.random.shuffle(data_dicts)
    
    split_idx = int(len(data_dicts) * (1 - validation_split))
    train_data_dicts = data_dicts[:split_idx]
    val_data_dicts = data_dicts[split_idx:]
    
    print(f"Created {len(train_data_dicts)} training samples and {len(val_data_dicts)} validation samples")
    
    return train_data_dicts, val_data_dicts