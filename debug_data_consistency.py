# Add this debugging function to your feature visualization code

def debug_data_consistency(data_loader, encoders, device='cuda', max_debug_batches=None):
    """
    Debug function to identify why different encoders process different numbers of cases
    """
    print("\n" + "="*80)
    print("🔍 DEBUGGING CASE COUNT DISCREPANCY")
    print("="*80)
    
    # First, let's check what the data loader actually contains
    print("\n📊 Data Loader Analysis:")
    total_expected_samples = 0
    unique_cases_in_loader = set()
    
    # Iterate through data loader to see what's actually there
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Analyzing Data Loader")):
        if max_debug_batches and batch_idx >= max_debug_batches:
            break
            
        input_phases = batch['input_phase']
        scan_ids = batch['scan_id']
        
        batch_cases = set(scan_ids)
        unique_cases_in_loader.update(batch_cases)
        total_expected_samples += len(input_phases)
        
        if batch_idx < 3:  # Show first few batches
            print(f"  Batch {batch_idx}: {len(input_phases)} samples, cases: {list(batch_cases)}")
    
    print(f"\n📈 Data Loader Summary:")
    print(f"  Total batches processed: {batch_idx + 1}")
    print(f"  Total samples expected: {total_expected_samples}")
    print(f"  Unique cases in data loader: {len(unique_cases_in_loader)}")
    print(f"  Cases: {sorted(list(unique_cases_in_loader))[:10]}...")  # Show first 10
    
    # Now test each encoder individually
    encoder_results = {}
    
    for encoder, encoder_name in encoders:
        print(f"\n🔍 Testing {encoder_name} Encoder:")
        
        encoder.eval()
        encoder.to(device)
        
        processed_samples = 0
        processed_cases = set()
        successful_batches = 0
        failed_batches = 0
        error_details = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Testing {encoder_name}")):
                if max_debug_batches and batch_idx >= max_debug_batches:
                    break
                
                try:
                    # Get batch data
                    input_volumes = batch['input_path'].to(device)
                    input_phases = batch['input_phase']
                    scan_ids = batch['scan_id']
                    
                    # Try to extract features
                    start_time = time.time()
                    features = encoder(input_volumes)
                    processing_time = time.time() - start_time
                    
                    # Success - record the data
                    batch_cases = set(scan_ids)
                    processed_cases.update(batch_cases)
                    processed_samples += len(input_phases)
                    successful_batches += 1
                    
                    if batch_idx < 5:  # Show first few successful batches
                        print(f"  ✅ Batch {batch_idx}: {features.shape} features in {processing_time:.2f}s")
                        print(f"     Cases: {list(batch_cases)}")
                        print(f"     Input shape: {input_volumes.shape}")
                
                except Exception as e:
                    failed_batches += 1
                    error_details.append((batch_idx, str(e), scan_ids if 'scan_ids' in locals() else []))
                    print(f"  ❌ Batch {batch_idx} FAILED: {str(e)[:100]}...")
                    
                    # Show more details for first few failures
                    if len(error_details) <= 3:
                        print(f"     Error details: {e}")
                        if 'input_volumes' in locals():
                            print(f"     Input shape was: {input_volumes.shape}")
                        if 'scan_ids' in locals():
                            print(f"     Failed cases: {scan_ids}")
        
        # Store results
        encoder_results[encoder_name] = {
            'processed_samples': processed_samples,
            'processed_cases': len(processed_cases),
            'successful_batches': successful_batches,
            'failed_batches': failed_batches,
            'cases_list': sorted(list(processed_cases)),
            'error_details': error_details
        }
        
        print(f"\n📊 {encoder_name} Results:")
        print(f"  Processed samples: {processed_samples}")
        print(f"  Unique cases: {len(processed_cases)}")
        print(f"  Successful batches: {successful_batches}")
        print(f"  Failed batches: {failed_batches}")
        
        if error_details:
            print(f"  Error types:")
            error_types = {}
            for _, error, _ in error_details:
                error_type = error.split(':')[0]
                error_types[error_type] = error_types.get(error_type, 0) + 1
            for error_type, count in error_types.items():
                print(f"    {error_type}: {count} occurrences")
    
    # Compare results
    print(f"\n🔍 COMPARISON ANALYSIS:")
    encoder_names = list(encoder_results.keys())
    
    if len(encoder_names) >= 2:
        enc1, enc2 = encoder_names[0], encoder_names[1]
        result1, result2 = encoder_results[enc1], encoder_results[enc2]
        
        print(f"\n{enc1} vs {enc2}:")
        print(f"  Cases: {result1['processed_cases']} vs {result2['processed_cases']}")
        print(f"  Samples: {result1['processed_samples']} vs {result2['processed_samples']}")
        print(f"  Failed batches: {result1['failed_batches']} vs {result2['failed_batches']}")
        
        # Find missing cases
        cases1 = set(result1['cases_list'])
        cases2 = set(result2['cases_list'])
        
        only_in_first = cases1 - cases2
        only_in_second = cases2 - cases1
        common_cases = cases1 & cases2
        
        print(f"\n📋 Case Analysis:")
        print(f"  Common cases: {len(common_cases)}")
        print(f"  Only in {enc1}: {len(only_in_first)}")
        print(f"  Only in {enc2}: {len(only_in_second)}")
        
        if only_in_first:
            print(f"  Cases only in {enc1}: {sorted(list(only_in_first))[:10]}...")
        
        if only_in_second:
            print(f"  Cases only in {enc2}: {sorted(list(only_in_second))[:10]}...")
    
    # Recommendations
    print(f"\n💡 DEBUGGING RECOMMENDATIONS:")
    
    max_cases = max(result['processed_cases'] for result in encoder_results.values())
    min_cases = min(result['processed_cases'] for result in encoder_results.values())
    
    if max_cases != min_cases:
        print(f"  ❌ Inconsistent case counts detected!")
        print(f"  ❌ Difference: {max_cases - min_cases} cases")
        
        # Find encoder with most failures
        most_failures = max(encoder_results.items(), key=lambda x: x[1]['failed_batches'])
        if most_failures[1]['failed_batches'] > 0:
            print(f"  🔧 {most_failures[0]} has {most_failures[1]['failed_batches']} failed batches")
            print(f"  🔧 Check memory usage, input preprocessing, or model compatibility")
        
        print(f"\n🔧 SOLUTIONS TO TRY:")
        print(f"  1. Add explicit error handling in feature extraction")
        print(f"  2. Check if one encoder runs out of GPU memory")
        print(f"  3. Verify input preprocessing consistency")
        print(f"  4. Use smaller batch sizes")
        print(f"  5. Check if caching is loading different data")
    
    else:
        print(f"  ✅ Case counts are consistent across encoders")
    
    return encoder_results

# Enhanced feature extraction function with better error handling
def extract_features_with_detailed_logging(encoder, data_loader, device='cuda', encoder_name='encoder'):
    """
    Feature extraction with detailed logging to identify issues
    """
    print(f"\n🔍 DETAILED FEATURE EXTRACTION: {encoder_name}")
    
    encoder.eval()
    encoder.to(device)
    
    all_features = []
    all_phases = []
    all_scan_ids = []
    
    successful_batches = 0
    failed_batches = 0
    processing_times = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Processing {encoder_name}")):
            try:
                batch_start = time.time()
                
                # Get input data
                input_volumes = batch['input_path'].to(device)
                input_phases = batch['input_phase']
                scan_ids = batch['scan_id']
                
                # Log batch info
                if batch_idx % 20 == 0:
                    print(f"  Processing batch {batch_idx}")
                    print(f"    Input shape: {input_volumes.shape}")
                    print(f"    Cases in batch: {scan_ids}")
                    print(f"    GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
                
                # Extract features
                features = encoder(input_volumes)
                
                # Record timing
                batch_time = time.time() - batch_start
                processing_times.append(batch_time)
                
                # Store results
                features_np = features.cpu().numpy()
                all_features.append(features_np)
                
                for i in range(len(input_phases)):
                    phase_label = input_phases[i].item() if torch.is_tensor(input_phases[i]) else input_phases[i]
                    all_phases.append(phase_label)
                    all_scan_ids.append(scan_ids[i])
                
                successful_batches += 1
                
            except Exception as e:
                failed_batches += 1
                print(f"  ❌ FAILED batch {batch_idx}: {e}")
                print(f"     Cases that failed: {scan_ids if 'scan_ids' in locals() else 'Unknown'}")
                
                # Clear GPU cache on failure
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                continue
    
    # Final summary
    print(f"\n📊 {encoder_name} Extraction Summary:")
    print(f"  Successful batches: {successful_batches}")
    print(f"  Failed batches: {failed_batches}")
    print(f"  Average batch time: {np.mean(processing_times):.2f}s")
    print(f"  Final features shape: {np.vstack(all_features).shape if all_features else 'No features'}")
    print(f"  Unique cases: {len(set(all_scan_ids))}")
    
    if all_features:
        features = np.vstack(all_features)
        return features, all_phases, all_scan_ids
    else:
        return np.array([]), [], []

# Add this to your main function, before the encoder processing loop:
def main_with_debugging():
    # ... your existing setup code ...
    
    # Add this debugging section before processing encoders:
    print("🐛 Running case count debugging...")
    
    # Debug data consistency
    debug_results = debug_data_consistency(data_loader, encoders, args.device, max_debug_batches=50)
    
    # Ask user if they want to continue
    response = input("\n❓ Continue with full processing? (y/n): ")
    if response.lower() != 'y':
        print("Debugging complete. Check the analysis above.")
        return
    
    # ... continue with your existing encoder processing loop ...

# Alternative quick check function you can add to your existing code:
def quick_case_count_check(data_loader, encoders, device='cuda'):
    """
    Quick function to check case counts without full processing
    """
    print("\n🔍 QUICK CASE COUNT CHECK:")
    
    # Check data loader
    all_cases_in_loader = set()
    sample_count = 0
    
    for batch_idx, batch in enumerate(data_loader):
        scan_ids = batch['scan_id']
        all_cases_in_loader.update(scan_ids)
        sample_count += len(scan_ids)
        
        if batch_idx >= 10:  # Just check first 10 batches
            break
    
    print(f"📊 Data Loader (first 10 batches): {len(all_cases_in_loader)} unique cases, {sample_count} samples")
    
    # Quick test each encoder
    for encoder, encoder_name in encoders:
        print(f"\n🧪 Testing {encoder_name}...")
        encoder.eval().to(device)
        
        processed_cases = set()
        processed_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= 5:  # Just test first 5 batches
                    break
                
                try:
                    input_volumes = batch['input_path'].to(device)
                    scan_ids = batch['scan_id']
                    
                    # Try processing
                    features = encoder(input_volumes)
                    
                    # Success
                    processed_cases.update(scan_ids)
                    processed_samples += len(scan_ids)
                    
                except Exception as e:
                    print(f"    ❌ Batch {batch_idx} failed: {str(e)[:50]}...")
        
        print(f"    ✅ {encoder_name}: {len(processed_cases)} cases, {processed_samples} samples processed")
        
        # Reset data loader for next encoder
        # Note: This might be the issue - data loader state not being reset!
    
    return True