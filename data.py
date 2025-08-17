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

def prepare_data(data_dicts, batch_size=2, augmentation=True, spatial_size=(128,128,128)):
    """Prepare MONAI dataset and DataLoader with optional augmentation."""
    # Basic transforms
    transforms = [
        LoadImaged(keys=["input_volume", "target_volume"]),
        EnsureChannelFirstd(keys=["input_volume", "target_volume"]),
        ScaleIntensityRanged(keys=["input_volume", "target_volume"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0),
        Resized(keys=["input_volume", "target_volume"], spatial_size=spatial_size),
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

def prepare_dataset_from_folders(data_root, labels_csv, validation_split=0.2, seed=42, apply_registration=False, skip_prep=False):
    prepared_csv = os.path.join(data_root, 'prepared_pairs.csv')
    progress_file = os.path.join(data_root, 'registration_progress.json')
    
    if skip_prep and os.path.exists(prepared_csv):
        print(f"Loading cached prepared pairs from {prepared_csv}")
        pairs_df = pd.read_csv(prepared_csv)
    else:
        report, pairs_df_path = validate_dataset(data_root, labels_csv)
        print("Validation Report:")
        print(report)
        pairs_df = pd.read_csv(pairs_df_path)
        if 'error' in report:
            raise ValueError(report['error'])
        
        if apply_registration:
            import json
            completed_scans = []
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    completed_scans = json.load(f)
            print(f"Resuming registration. Completed: {len(completed_scans)} scans")
            
            all_scans = pairs_df['scan_id'].unique()
            remaining_scans = [s for s in all_scans if s not in completed_scans]
            
            for scan_id in remaining_scans:
                scan_dir = os.path.join(data_root, scan_id)
                output_dir = os.path.join(data_root, f"{scan_id}_registered")
                row = pairs_df[(pairs_df['scan_id'] == scan_id) & (pairs_df['input_phase'] == 'Non-contrast')]
                nc_path = ''
                if not row.empty:
                    nc_path = row.iloc[0]['input_path']
                else:
                    row = pairs_df[(pairs_df['scan_id'] == scan_id) & (pairs_df['target_phase'] == 'Non-contrast')]
                    if not row.empty:
                        nc_path = row.iloc[0]['target_path']
                register_case_series(scan_dir, output_dir, nc_path)
                # Update paths to match actual registered filenames
                # Registered files are saved as: output_dir / ("registered_" + original_filename)
                # Build new paths using basename to avoid incorrect string replacements
                pairs_df.loc[pairs_df['scan_id'] == scan_id, 'input_path'] = (
                    pairs_df.loc[pairs_df['scan_id'] == scan_id, 'input_path']
                    .apply(lambda p: os.path.join(output_dir, f"registered_{os.path.basename(p)}"))
                )
                pairs_df.loc[pairs_df['scan_id'] == scan_id, 'target_path'] = (
                    pairs_df.loc[pairs_df['scan_id'] == scan_id, 'target_path']
                    .apply(lambda p: os.path.join(output_dir, f"registered_{os.path.basename(p)}"))
                )
                # Mark as completed
                completed_scans.append(scan_id)
                with open(progress_file, 'w') as f:
                    json.dump(completed_scans, f)
                print(f"Completed registration for {scan_id}. Progress saved.")
            
            # Clean up progress file after completion
            if len(remaining_scans) == 0:
                print("All registrations complete.")
            else:
                print(f"Registered {len(remaining_scans)} remaining scans.")
        
        # Save the prepared pairs
        pairs_df.to_csv(prepared_csv, index=False)
        print(f"Saved prepared pairs to {prepared_csv}")
        # Optional: Remove progress file if all done
        if apply_registration and os.path.exists(progress_file):
            os.remove(progress_file)
    
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
    if not os.path.exists(non_contrast_file):
        print(f"Non-contrast file not found: {non_contrast_file}")
        return
    
    fixed = sitk.ReadImage(non_contrast_file)
    
    # Work with basename to properly exclude the non-contrast file from registration
    nc_basename = os.path.basename(non_contrast_file)
    nifti_files = [f for f in os.listdir(scan_dir) if f.endswith('.nii.gz') and f != nc_basename]
    
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
    non_contrast_seriesid = os.path.basename(non_contrast_file)
    shutil.copy(non_contrast_file, os.path.join(output_dir, non_contrast_seriesid))

def validate_dataset(data_folders, labels_csv, output_csv='dataset_pairs.csv'):
    """Validates the dataset structure and outputs available phase pairs to a CSV."""
    # Initialize report
    report = {"status": "success", "message": ""}
    
    try:
        # Load labels
        if not os.path.exists(labels_csv):
            report = {"status": "error", "error": f"Labels CSV not found: {labels_csv}"}
            return report, None
            
        labels_df = pd.read_csv(labels_csv)
        # Robust phase normalization -> integer ids (0: NC, 1: A, 2: PV, 3: D)
        def phase_to_id(label):
            s = str(label).strip().lower()
            if s in {"nc", "non-contrast", "non contrast", "noncontrast"}:
                return 0
            if s in {"a", "arterial", "aterial"}:  # include possible typo 'aterial'
                return 1
            if s in {"pv", "venous", "portal venous", "portal-venous", "portal_venous"}:
                return 2
            if s in {"d", "delayed", "delay"}:
                return 3
            return label
        available_phases = {'Non-contrast', 'Aterial', 'Venous'}
        # Organize files by scan_id
        scan_files = defaultdict(dict)
        for scan_id in os.listdir(data_folders):
            scan_path = os.path.join(data_folders, scan_id)
            # print("scan path", scan_path)
            if os.path.isdir(scan_path):
                for series_file in os.listdir(scan_path):
                    postfixes = ['.nii.gz', '.nii']
                    if any(series_file.endswith(p) for p in postfixes):
                        # print(series_file)
                        for p in postfixes:
                            if series_file.endswith(p):
                                series_id = series_file[: -len(p)]
                                break
                        # print(series_id)
                        if 'registered' in series_id:
                            continue
                        phase_arr = labels_df[labels_df['SeriesInstanceUID'] == series_id]['Label'].values
                        if len(phase_arr) == 0:
                            print(f"Warning: No phase found for series_id: {series_id}")
                            continue
                        phase = str(phase_arr[0])

                        # print("phase", phase)
                        series_file_path = os.path.join(scan_path, series_file)
                        # print(series_file_path)
                        scan_files[scan_id][phase] = series_file_path

        # print("scan files", scan_files)
        # Validate loading and collect pairs
        valid_pairs = []
        for scan_id, phases in scan_files.items():
            # print(scan_id, phases)
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
                            # valid_pairs.append((input_path, target_path, input_phase, target_phase))
                            valid_pairs.append([
                                input_path,
                                target_path,
                                phase_to_id(input_phase),
                                phase_to_id(target_phase),
                                scan_id
                            ])
                        except Exception as e:
                            print(f"Error loading files for {scan_id} ({input_phase} -> {target_phase}): {e}")
        
      
        # Save to CSV (ensure integer phase ids)
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['input_path', 'target_path', 'input_phase', 'target_phase', 'scan_id'])
            writer.writerows(valid_pairs)
        print(f"Validation complete. {len(valid_pairs)} valid pairs found. CSV saved to {output_csv}")
        del valid_pairs
        return report, output_csv
    
    except Exception as e:
        report = {"status": "error", "error": f"Validation failed: {str(e)}"}
        return report, None
        
    
    
    
    #     valid_pairs = []
    #     for scan_id, phases in scan_files.items():
    #         available_phases = list(phases.keys())
    #         for input_phase in available_phases:
    #             for target_phase in available_phases:
    #                 if input_phase != target_phase:
    #                     input_path = phases[input_phase]
    #                     target_path = phases[target_phase]
    #                     try:
    #                         # Try loading NIfTI files
    #                         nib.load(input_path)
    #                         nib.load(target_path)
    #                         valid_pairs.append({
    #                             "input_path": input_path, 
    #                             "target_path": target_path, 
    #                             "input_phase": phase_map.get(input_phase, input_phase), 
    #                             "target_phase": phase_map.get(target_phase, target_phase),
    #                             "scan_id": scan_id
    #                         })
    #                     except Exception as e:
    #                         print(f"Error loading files for {scan_id} ({input_phase} -> {target_phase}): {e}")
        
    #     # Create DataFrame
    #     pairs_df = pd.DataFrame(valid_pairs)
        
    #     # Save to CSV
    #     if not pairs_df.empty:
    #         pairs_df.to_csv(output_csv, index=False)
    #         report["message"] = f"Validation complete. {len(valid_pairs)} valid pairs found. CSV saved to {output_csv}"
    #         print(report["message"])
    #     else:
    #         report = {"status": "error", "error": "No valid pairs found. Check your data structure and labels.csv."}
    #         return report, None
            
    #     return report, pairs_df
        
    # except Exception as e:
    #     report = {"status": "error", "error": f"Validation failed: {str(e)}"}
    #     return report, None



