
import os
import csv
import nibabel as nib
import pandas as pd
import numpy as np
from collections import defaultdict
import json
import shutil

from monai.data import Dataset, DataLoader
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Resized,
    ToTensord,
    RandRotate90d,
    RandFlipd,
    RandShiftIntensityd,
    Compose
)

def prepare_data(data_dicts, batch_size=2, augmentation=False, spatial_size=(128,128,128)):
    """Prepare MONAI dataset and DataLoader with optional augmentation."""
    # Basic transforms
    transforms = [
        LoadImaged(keys=["input_path", "target_path"]),
        EnsureChannelFirstd(keys=["input_path", "target_path"]),
        ScaleIntensityRanged(keys=["input_path", "target_path"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True ),
        Resized(keys=["input_path", "target_path"], spatial_size=spatial_size, mode='trilinear'),
        ToTensord(keys=["input_path", "target_path", "input_phase", "target_phase"])
    ]
    
    # Add augmentation if requested
    if augmentation:
        aug_transforms = [
            RandRotate90d(keys=["input_path", "target_path"], prob=0.1, spatial_axes=(0, 1)),
            RandFlipd(keys=["input_path", "target_path"], prob=0.1, spatial_axis=0),
            RandShiftIntensityd(keys=["input_path", "target_path"], prob=0.1, offsets=0.05)
        ]
        transforms.extend(aug_transforms)
    
    transform = Compose(transforms)
    # FIXED: Add data validation
    print(f"🔍 Validating data paths...")
    valid_data_dicts = []
    invalid_count = 0
    
    for i, data_dict in enumerate(data_dicts):
        input_path = data_dict["input_path"]
        target_path = data_dict["target_path"]
        
        # Check if files exist
        if not os.path.exists(input_path):
            print(f"   ⚠️  Missing input: {input_path}")
            invalid_count += 1
            continue
            
        if not os.path.exists(target_path):
            print(f"   ⚠️  Missing target: {target_path}")
            invalid_count += 1
            continue
            
        # Quick file validation
        try:
            nib.load(input_path)
            nib.load(target_path)
            valid_data_dicts.append(data_dict)
        except Exception as e:
            print(f"   ⚠️  Invalid file {input_path}: {e}")
            invalid_count += 1
    
    if invalid_count > 0:
        print(f"   ⚠️  Excluded {invalid_count} invalid data pairs")
    
    print(f"   ✅ Using {len(valid_data_dicts)} valid data pairs")
    
    dataset = Dataset(valid_data_dicts, transform=transform)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,  # Reduced for stability
        pin_memory=True,
        persistent_workers=True  # Better performance
    )

    # dataset = Dataset(data_dicts, transform=transform)
    # return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

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
    """Prepare dataset from folder structure with improved phase handling"""
    
    prepared_csv = os.path.join(data_root, 'prepared_pairs.csv')
    progress_file = os.path.join(data_root, 'registration_progress.json')
    
    if skip_prep and os.path.exists(prepared_csv):
        print(f"📄 Loading cached prepared pairs from {prepared_csv}")
        pairs_df = pd.read_csv(prepared_csv)
    else:
        print(f"🔍 Validating dataset structure...")
        report, pairs_df_path = validate_dataset(data_root, labels_csv)
        print("📊 Validation Report:")
        print(report)
        
        if 'error' in report:
            raise ValueError(report['error'])
            
        pairs_df = pd.read_csv(pairs_df_path)
        
        if apply_registration:
            print(f"🔧 Applying volume registration...")
            import json
            completed_scans = []
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    completed_scans = json.load(f)
            print(f"   Resuming registration. Completed: {len(completed_scans)} scans")
            
            all_scans = pairs_df['scan_id'].unique()
            remaining_scans = [s for s in all_scans if s not in completed_scans]
            
            for scan_id in remaining_scans:
                scan_dir = os.path.join(data_root, scan_id)
                output_dir = os.path.join(data_root, f"{scan_id}_registered")
                
                # Find non-contrast volume as reference
                nc_row = pairs_df[(pairs_df['scan_id'] == scan_id) & (pairs_df['input_phase'] == 0)]
                if nc_row.empty:
                    nc_row = pairs_df[(pairs_df['scan_id'] == scan_id) & (pairs_df['target_phase'] == 0)]
                
                nc_path = nc_row.iloc[0]['input_path'] if not nc_row.empty else ''
                
                register_case_series(scan_dir, output_dir, nc_path)
                
                # Update paths to registered versions
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
                print(f"   Completed registration for {scan_id}")
            
            print(f"✅ Registration completed for all scans")
            
            # Clean up progress file
            if os.path.exists(progress_file):
                os.remove(progress_file)
        
        # Save the prepared pairs
        pairs_df.to_csv(prepared_csv, index=False)
        print(f"💾 Saved prepared pairs to {prepared_csv}")
    
    # FIXED: Validate phase consistency
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total pairs: {len(pairs_df)}")
    print(f"   Unique scans: {len(pairs_df['scan_id'].unique())}")
    
    # Check phase distribution
    input_phase_dist = pairs_df['input_phase'].value_counts().sort_index()
    target_phase_dist = pairs_df['target_phase'].value_counts().sort_index()
    
    phase_names = {0: 'Non-contrast', 1: 'Arterial', 2: 'Venous', 3: 'Delayed'}
    
    print(f"   Input phase distribution:")
    for phase, count in input_phase_dist.items():
        phase_name = phase_names.get(phase, f'Phase_{phase}')
        print(f"     {phase_name}: {count}")
    
    print(f"   Target phase distribution:")
    for phase, count in target_phase_dist.items():
        phase_name = phase_names.get(phase, f'Phase_{phase}')
        print(f"     {phase_name}: {count}")
    
    # Create data_dicts from pairs_df
    data_dicts = []
    for _, row in pairs_df.iterrows():
        data_dicts.append({
            "input_path": row['input_path'],
            "target_path": row['target_path'],
            "input_phase": int(row['input_phase']),  # Ensure integer
            "target_phase": int(row['target_phase']),  # Ensure integer
            "scan_id": row['scan_id']
        })
    
    # FIXED: Better train/val split (stratified by scan_id to avoid leakage)
    print(f"\n🔀 Creating train/validation split...")
    
    # Split by scan_id to avoid data leakage
    unique_scan_ids = list(pairs_df['scan_id'].unique())
    np.random.seed(seed)
    np.random.shuffle(unique_scan_ids)
    
    split_idx = int(len(unique_scan_ids) * (1 - validation_split))
    train_scan_ids = set(unique_scan_ids[:split_idx])
    val_scan_ids = set(unique_scan_ids[split_idx:])
    
    train_data_dicts = [d for d in data_dicts if d['scan_id'] in train_scan_ids]
    val_data_dicts = [d for d in data_dicts if d['scan_id'] in val_scan_ids]
    
    print(f"   Training scans: {len(train_scan_ids)}")
    print(f"   Validation scans: {len(val_scan_ids)}")
    print(f"   Training pairs: {len(train_data_dicts)}")
    print(f"   Validation pairs: {len(val_data_dicts)}")
    
    return train_data_dicts, val_data_dicts

import SimpleITK as sitk

def register_case_series(scan_dir, output_dir, non_contrast_file='non_contrast.nii.gz'):
    """Perform rigid registration on series of a case using non-contrast as atlas."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(non_contrast_file):
        print(f"⚠️  Non-contrast file not found: {non_contrast_file}")
        return
    
    try:
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
            print(f"   Registered {moving_file}")
        
        # Copy non-contrast as is
        import shutil
        nc_output = os.path.join(output_dir, f"registered_{nc_basename}")
        shutil.copy(non_contrast_file, nc_output)
        print(f"   Copied reference {nc_basename}")
        
    except Exception as e:
        print(f"❌ Error in registration for {scan_dir}: {e}")

def validate_dataset(data_folders, labels_csv, output_csv='dataset_pairs.csv'):
    """Validates the dataset structure with improved phase handling"""
    
    print(f"🔍 Validating dataset structure...")
    
    # Initialize report
    report = {"status": "success", "message": ""}
    
    try:
        # Load labels
        if not os.path.exists(labels_csv):
            report = {"status": "error", "error": f"Labels CSV not found: {labels_csv}"}
            return report, None
            
        labels_df = pd.read_csv(labels_csv)
        print(f"   Loaded {len(labels_df)} labels from CSV")
        
        # FIXED: Robust phase normalization
        def phase_to_id(label):
            s = str(label).strip().lower()
            
            # Non-contrast variations
            if s in {"nc", "non-contrast", "non contrast", "noncontrast", "pre", "precontrast"}:
                return 0
            # Arterial variations  
            elif s in {"a", "arterial", "aterial", "art"}:
                return 1
            # Venous variations
            elif s in {"pv", "venous", "portal venous", "portal-venous", "portal_venous", "v"}:
                return 2
            # Delayed variations
            elif s in {"d", "delayed", "delay"}:
                return 3
            else:
                print(f"   ⚠️  Unknown phase label: '{label}', mapping to original value")
                return label
        
        # Organize files by scan_id
        scan_files = defaultdict(dict)
        missing_labels = []
        
        for scan_id in os.listdir(data_folders):
            scan_path = os.path.join(data_folders, scan_id)
            
            if not os.path.isdir(scan_path):
                continue
                
            print(f"   Processing scan: {scan_id}")
            
            for series_file in os.listdir(scan_path):
                # Check for NIfTI files
                if not series_file.endswith(('.nii.gz', '.nii')):
                    continue
                    
                # Skip registered files
                if 'registered' not in series_file:
                    continue

                # Extract series ID
                if series_file.endswith('.nii.gz'):
                    series_id = series_file[:-7]  # Remove .nii.gz
                elif series_file.endswith('.nii'):
                    series_id = series_file[:-4]  # Remove .nii
                series_id = series_id.replace('registered_', '') if 'registered_' in series_id else series_id.replace('_registered', '')
                print("series_file", series_id)
                # Find phase label
                phase_rows = labels_df[labels_df['SeriesInstanceUID'] == series_id]
                
                if len(phase_rows) == 0:
                    missing_labels.append(series_id)
                    print('len(phase_rows) == 0', series_id)
                    continue
                
                phase_label = phase_rows.iloc[0]['Label']
                phase_id = phase_to_id(phase_label)
                
                series_file_path = os.path.join(scan_path, series_file)
                scan_files[scan_id][phase_id] = series_file_path
                
                print(f"     {series_file} -> Phase {phase_id} ({phase_label})")
        
        if missing_labels:
            print(f"   ⚠️  Warning: {len(missing_labels)} series without labels")
            if len(missing_labels) <= 10:
                print(f"     Missing: {missing_labels}")
        
        # Generate valid pairs and validate files
        valid_pairs = []
        file_errors = []
        
        for scan_id, phases in scan_files.items():
            available_phases = list(phases.keys())
            print(f"   Scan {scan_id}: {len(available_phases)} phases available")
            
            for input_phase in available_phases:
                for target_phase in available_phases:
                    if input_phase != target_phase:
                        input_path = phases[input_phase]
                        target_path = phases[target_phase]
                        
                        # Validate files can be loaded
                        try:
                            nib.load(input_path)
                            nib.load(target_path)
                            
                            valid_pairs.append([
                                input_path,
                                target_path,
                                input_phase,
                                target_phase,
                                scan_id
                            ])
                        except Exception as e:
                            file_errors.append(f"{scan_id}: {input_phase}->{target_phase} - {e}")
        
        if file_errors:
            print(f"   ⚠️  File loading errors: {len(file_errors)}")
            for error in file_errors[:5]:  # Show first 5 errors
                print(f"     {error}")
        
        # Save to CSV
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['input_path', 'target_path', 'input_phase', 'target_phase', 'scan_id'])
            writer.writerows(valid_pairs)
        
        print(f"✅ Validation complete:")
        print(f"   Valid pairs found: {len(valid_pairs)}")
        print(f"   Scans processed: {len(scan_files)}")
        print(f"   CSV saved to: {output_csv}")
        
        # Analysis of phase distribution
        phase_count = defaultdict(int)
        for pair in valid_pairs:
            phase_count[pair[2]] += 1  # input_phase
            phase_count[pair[3]] += 1  # target_phase
        
        phase_names = {0: 'Non-contrast', 1: 'Arterial', 2: 'Venous', 3: 'Delayed'}
        print(f"   Phase distribution:")
        for phase_id, count in sorted(phase_count.items()):
            phase_name = phase_names.get(phase_id, f'Phase_{phase_id}')
            print(f"     {phase_name}: {count} occurrences")
        
        report["message"] = f"Validation successful. {len(valid_pairs)} valid pairs found."
        return report, output_csv
        
    except Exception as e:
        report = {"status": "error", "error": f"Validation failed: {str(e)}"}
        print(f"❌ Validation error: {e}")
        return report, None

def create_phase_detection_dataset(data_dicts):
    """Create a unique dataset for phase detection training (remove duplicates)"""
    
    print(f"🔍 Creating unique phase detection dataset...")
    
    # Debug: Print first few entries of input data
    print("\nDEBUG: First 3 input data_dicts:")
    for i, d in enumerate(data_dicts[:3]):
        print(f"Entry {i}: {d}")
    
    unique_volumes = {}
    phase_counts = defaultdict(int)
    
    for data_dict in data_dicts:
        input_path = data_dict["volume"]
        input_phase = data_dict["phase"]
        scan_id = data_dict["scan_id"]
        
        # Use path as unique identifier
        if input_path not in unique_volumes:
            unique_volumes[input_path] = {
                "volume": input_path,
                "phase": input_phase,
                "scan_id": scan_id
            }
            phase_counts[input_phase] += 1
    
    # Debug: Print first few unique volumes
    print("\nDEBUG: First 3 unique volumes:")
    for i, (path, data) in enumerate(list(unique_volumes.items())[:3]):
        print(f"Volume {i}: Path={path}, Phase={data['phase']}, ID={data['scan_id']}")
    
    print(f"   Original pairs: {len(data_dicts)}")
    print(f"   Unique volumes: {len(unique_volumes)}")
    print(f"   Phase distribution:")
    
    phase_names = {0: 'Non-contrast', 1: 'Arterial', 2: 'Venous', 3: 'Delayed'}
    for phase, count in sorted(phase_counts.items()):
        phase_name = phase_names.get(phase, f'Phase_{phase}')
        print(f"     {phase_name}: {count} volumes")
    
    return list(unique_volumes.values())

def prepare_phase_detection_data(data_dicts, batch_size=4, spatial_size=(128,128,128)):
    """Prepare DataLoader specifically for phase detection training"""
    
    # Create unique dataset
    unique_data_dicts = create_phase_detection_dataset(data_dicts)
    
    # Define transforms with proper normalization
    transforms = [
        LoadImaged(keys=["volume"]),
        EnsureChannelFirstd(keys=["volume"]),
        ScaleIntensityRanged(
            keys=["volume"], 
            a_min=-1000,
            a_max=1000, 
            b_min=0.0, 
            b_max=1.0,
            clip=True
        ),
        Resized(keys=["volume"], spatial_size=spatial_size, mode='trilinear'),
        ToTensord(keys=["volume", "phase"])
    ]
    
    transform = Compose(transforms)
    dataset = Dataset(unique_data_dicts, transform=transform)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )

# Test function for data loading
def test_data_loading(data_path, labels_csv):
    """Test the data loading pipeline"""
    
    print(f"🧪 Testing data loading pipeline...")
    
    try:
        # Test dataset validation
        train_data_dicts, val_data_dicts = prepare_dataset_from_folders(
            data_path, labels_csv, validation_split=0.2, skip_prep=True
        )
        
        print(f"✅ Dataset preparation successful")
        print(f"   Training samples: {len(train_data_dicts)}")
        print(f"   Validation samples: {len(val_data_dicts)}")
        
        # Test data loader
        train_loader = prepare_data(
            train_data_dicts[:10], 
            batch_size=2, 
            spatial_size=(64, 64, 64)  # Smaller for testing
        )
        
        # Test a batch
        sample_batch = next(iter(train_loader))
        
        print(f"✅ Data loader test successful")
        print(f"   Input volume shape: {sample_batch['input_path'].shape}")
        print(f"   Target volume shape: {sample_batch['target_path'].shape}")
        print(f"   Input volume range: [{sample_batch['input_path'].min():.3f}, {sample_batch['input_path'].max():.3f}]")
        print(f"   Target volume range: [{sample_batch['target_path'].min():.3f}, {sample_batch['target_path'].max():.3f}]")
        print(f"   Input phases: {sample_batch['input_phase'].numpy()}")
        print(f"   Target phases: {sample_batch['target_phase'].numpy()}")
        
        # Test phase detection data loader
        phase_loader = prepare_phase_detection_data(
            train_data_dicts[:10],
            batch_size=2,
            spatial_size=(64, 64, 64)
        )
        
        phase_batch = next(iter(phase_loader))
        
        print(f"✅ Phase detection loader test successful")
        print(f"   Volume shape: {phase_batch['volume'].shape}")
        print(f"   Volume range: [{phase_batch['volume'].min():.3f}, {phase_batch['volume'].max():.3f}]")
        print(f"   Phases: {phase_batch['phase'].numpy()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

if __name__ == "__main__":
    # Example test
    test_data_loading('../ncct_cect/vindr_ds/original_volumes/Abdomen/raw_image/', '../ncct_cect/vindr_ds/original_volumes/Abdomen/raw_image/labels.csv')

