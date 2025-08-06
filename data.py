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
import csv
import nibabel as nib
import pandas as pd
from collections import defaultdict

def prepare_dataset_from_folders(data_root, labels_csv, validation_split=0.2, seed=42, apply_registration=False):
    # First, validate dataset
    report, pairs_df = validate_dataset(data_root, labels_csv)
    print("Validation Report:")
    print(report)
    
    if 'error' in report:
        raise ValueError(report['error'])
    
    if apply_registration:
        # Apply registration to each scan
        for StudyInstanceUID in pairs_df['StudyInstanceUID'].unique():
            scan_dir = os.path.join(data_root, StudyInstanceUID)
            output_dir = os.path.join(data_root, f"{StudyInstanceUID}_registered")
            register_case_series(scan_dir, output_dir)
            # Update paths in pairs_df to registered versions
            pairs_df.loc[pairs_df['StudyInstanceUID'] == StudyInstanceUID, 'input_path'] = pairs_df['input_path'].apply(
                lambda p: p.replace(scan_dir, output_dir).replace('.nii.gz', '_registered.nii.gz')
            )
            pairs_df.loc[pairs_df['StudyInstanceUID'] == StudyInstanceUID, 'target_path'] = pairs_df['target_path'].apply(
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
            "StudyInstanceUID": row['StudyInstanceUID']
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

def register_case_series(scan_dir, output_dir, labels_csv, non_contrast_phase='Non-contrast'):
    """
    Perform rigid registration on series of a case using non-contrast as atlas.
    
    Args:
        scan_dir: Directory of the StudyInstanceUID containing series UID-named NIfTI files.
        output_dir: Directory to save registered volumes.
        labels_csv: Path to CSV with StudyInstanceUID, series_uid, and phase information.
        non_contrast_phase: The phase label for non-contrast (default: 'NC').
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load labels
    labels_df = pd.read_csv(labels_csv)
    StudyInstanceUID = os.path.basename(scan_dir)  # Assuming scan_dir is named after StudyInstanceUID
    print("StudyInstanceUID", StudyInstanceUID)
    scan_labels = labels_df[labels_df['StudyInstanceUID'] == StudyInstanceUID]
    
    if scan_labels.empty:
        print(f"No labels found for StudyInstanceUID: {StudyInstanceUID}")
        return
    
    # Find non-contrast file
    nc_row = scan_labels[scan_labels['Label'] == non_contrast_phase]
    if nc_row.empty:
        print(f"Non-contrast phase '{non_contrast_phase}' not found for {StudyInstanceUID}")
        return
    print("nc_row", nc_row)
    nc_series_uid = nc_row['SeriesInstanceUID'].values[0]  # Assuming 'series_uid' column
    fixed_path = os.path.join(scan_dir, f"{nc_series_uid}.nii.gz")  # Adjust extension if needed
    if not os.path.exists(fixed_path):
        print(f"Non-contrast file not found: {fixed_path}")
        return
    
    fixed = sitk.ReadImage(fixed_path)
    
    # Find other NIfTI files
    nifti_files = [f for f in os.listdir(scan_dir) if f.endswith('.nii.gz') or f.endswith('.nii')]
    
    for moving_file in nifti_files:
        if moving_file == f"{nc_series_uid}.nii.gz":
            # Copy non-contrast as is
            import shutil
            shutil.copy(os.path.join(scan_dir, moving_file), os.path.join(output_dir, moving_file))
            continue
        
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

def validate_dataset(data_folders, labels_csv, output_csv='dataset_pairs.csv'):
    """
    Validates the dataset structure, checks if NIfTI files can be loaded, and outputs available phase pairs to a CSV.
    
    Args:
        data_folders (list): List of paths to folders containing StudyInstanceUID subfolders with NIfTI files.
        labels_csv (str): Path to labels.csv with StudyInstanceUID and phase information.
        output_csv (str): Path to save the CSV with available pairs (default: 'dataset_pairs.csv').
    
    Returns:
        list: List of valid (input_path, target_path, input_phase, target_phase) tuples.
    """
    # Load labels
    labels_df = pd.read_csv(labels_csv)
    phase_map = {'NC': 0, 'A': 1, 'PV': 2, 'D': 3}  # Adjust as per your phases
    
    # Organize files by StudyInstanceUID
    scan_files = defaultdict(dict)
    for StudyInstanceUID in os.listdir(data_folders):
        scan_path = os.path.join(data_folders, StudyInstanceUID)
        print("scan path", scan_path)
        if os.path.isdir(scan_path):
            for series_file in os.listdir(scan_path):
                postfixes = ['.nii.gz', '.nii']
                if any(series_file.endswith(p) for p in postfixes):
                    print(series_file)
                    for p in postfixes:
                        if series_file.endswith(p):
                            series_id = series_file[: -len(p)]
                            break
                    print(series_id)
                    phase = labels_df[labels_df['SeriesInstanceUID'] == series_id]['Label'].values  # Assuming one phase per StudyInstanceUID, adjust if needed
                    if len(phase) == 0:
                        print(f"Warning: No phase found for series_id: {series_id}")
                        continue
                    phase = str(phase[0])
                    print("phase", phase)
                    series_file_path = os.path.join(scan_path, series_file)
                    print(series_file_path)
                    scan_files[StudyInstanceUID][phase] = series_file_path

    
    # Validate loading and collect pairs
    valid_pairs = []
    for StudyInstanceUID, phases in scan_files.items():
        available_phases = list(phases.keys())
        for input_phase in available_phases:
            for target_phase in available_phases:
                if input_phase != target_phase:
                    input_path = phases[input_phase]
                    target_path = phases[target_phase]
                    try:
                        # Try loading NIfTI files
                        nib.load(input_path)
                        nib.load(target_path)
                        valid_pairs.append((input_path, target_path, input_phase, target_phase))
                    except Exception as e:
                        print(f"Error loading files for {StudyInstanceUID} ({input_phase} -> {target_phase}): {e}")
    
    # Save to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['input_path', 'target_path', 'input_phase', 'target_phase'])
        writer.writerows(valid_pairs)
    
    print(f"Validation complete. {len(valid_pairs)} valid pairs found. CSV saved to {output_csv}")
    return valid_pairs