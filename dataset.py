import numpy as np
import pandas as pd
import random
from scipy import stats
import time

def fixed_streamlit_classification(temperatures, hole_size='30mm', sensitivity_level='balanced'):
    """
    FIXED classification logic with PROPER thresholds that will create balance
    """
    if len(temperatures) == 0:
        return {'leak_status': 0, 'confidence': 0.0, 'criteria_met': []}
    
    temp_array = np.array(temperatures)
    
    # CORRECTED BASE THRESHOLD - much higher than original
    base_threshold = 320.0  # Increased from 305 to 320
    
    # Hole size factors (kept same)
    hole_factors = {'20mm': 0.7, '25mm': 0.85, '30mm': 1.0, '40mm': 1.3}
    factor = hole_factors.get(hole_size, 1.0)
    
    # Adjusted threshold
    adjusted_threshold = base_threshold * factor  # 320K for 30mm
    
    criteria = {}
    criteria_met = []
    
    # 1. Statistical outliers - STRICTER
    if len(temp_array) > 1:
        z_scores = np.abs(stats.zscore(temp_array))
        outlier_count = np.sum(z_scores > 2.5)  # Stricter: 2.5 instead of 2
        outlier_ratio = outlier_count / len(temp_array)
        criteria['outliers'] = outlier_ratio > (0.25 * factor)  # Higher threshold: 0.25
    else:
        criteria['outliers'] = False
    
    if criteria['outliers']:
        criteria_met.append(f"Outlier ratio significant")
    
    # 2. High temperature ratio - STRICTER  
    high_temp_count = np.sum(temp_array > adjusted_threshold)
    high_temp_ratio = high_temp_count / len(temp_array)
    criteria['high_temp_ratio'] = high_temp_ratio > (0.3 * factor)  # Much higher: 0.3
    
    if criteria['high_temp_ratio']:
        criteria_met.append(f"High temp ratio above threshold")
    
    # 3. Mean temperature - HIGHER THRESHOLD
    mean_threshold = adjusted_threshold + 5  # ADDED 5K instead of subtracting 3K
    criteria['mean_temp'] = temp_array.mean() > mean_threshold
    
    if criteria['mean_temp']:
        criteria_met.append(f"Mean temp above threshold")
    
    # 4. Temperature variability - MUCH STRICTER
    std_threshold = 15 * factor  # Much higher: 15 instead of 8
    criteria['temp_variance'] = temp_array.std() > std_threshold
    
    if criteria['temp_variance']:
        criteria_met.append(f"Temp variance high")
    
    # 5. Maximum temperature - MUCH HIGHER
    max_threshold = adjusted_threshold + (20 * factor)  # Much higher: +20 instead of +5
    criteria['max_temp'] = temp_array.max() > max_threshold
    
    if criteria['max_temp']:
        criteria_met.append(f"Max temp above threshold")
    
    # Decision logic - REQUIRE MORE CRITERIA
    criteria_count = sum(criteria.values())
    total_criteria = len(criteria)
    
    # Much stricter: need 4 out of 5 criteria for leak
    if sensitivity_level == 'strict':
        is_leak = criteria_count >= (total_criteria)  # All 5
    elif sensitivity_level == 'sensitive':
        is_leak = criteria_count >= 2  # 2 out of 5
    else:  # 'balanced'
        is_leak = criteria_count >= 4  # 4 out of 5 - MUCH STRICTER
    
    return {
        'leak_status': int(is_leak),
        'criteria_count': criteria_count,
        'criteria_met': criteria_met,
        'thresholds': {
            'adjusted_threshold': adjusted_threshold,
            'mean_threshold': mean_threshold,
            'std_threshold': std_threshold,
            'max_threshold': max_threshold
        }
    }

def generate_guaranteed_no_leak_temperatures(hole_size, file_size):
    """Generate temperatures that will DEFINITELY be classified as NO LEAK"""
    
    # Much more conservative temperatures
    temperatures = []
    
    for i in range(file_size):
        # Very low temperatures with minimal variance
        base_temp = np.random.normal(295.0, 3.0)  # Low mean, low std
        
        # Add small random noise
        noise = np.random.normal(0, 1.0)
        temperature = base_temp + noise
        
        # Hard cap at 305K to ensure NO criteria are met
        temperature = np.clip(temperature, 280.0, 305.0)
        temperatures.append(temperature)
    
    return temperatures

def generate_guaranteed_leak_temperatures(hole_size, file_size):
    """Generate temperatures that will DEFINITELY be classified as LEAK"""
    
    # Get the CORRECTED thresholds
    hole_factors = {'20mm': 0.7, '25mm': 0.85, '30mm': 1.0, '40mm': 1.3}
    factor = hole_factors.get(hole_size, 1.0)
    
    base_threshold = 320.0
    adjusted_threshold = base_threshold * factor
    mean_threshold = adjusted_threshold + 5
    std_threshold = 15 * factor
    max_threshold = adjusted_threshold + (20 * factor)
    
    temperatures = []
    
    # Strategy: Create temperatures that will pass 4+ criteria
    
    # 40% extremely hot points (ensures max_temp + high_temp_ratio)
    extreme_hot_count = int(file_size * 0.4)
    for i in range(extreme_hot_count):
        temp = np.random.normal(max_threshold + 30, 10.0)  # Way above max threshold
        temperatures.append(temp)
    
    # 30% very hot points (ensures mean_temp)
    very_hot_count = int(file_size * 0.3)  
    for i in range(very_hot_count):
        temp = np.random.normal(mean_threshold + 20, 8.0)  # Well above mean threshold
        temperatures.append(temp)
    
    # 20% highly variable points (ensures temp_variance)
    variable_count = int(file_size * 0.2)
    for i in range(variable_count):
        # Create extreme variance
        if i % 2 == 0:
            temp = np.random.normal(mean_threshold + 40, std_threshold + 20)  # High variance hot
        else:
            temp = np.random.normal(mean_threshold - 20, std_threshold + 15)  # High variance cool
        temperatures.append(temp)
    
    # Remaining points as extreme outliers (ensures outliers)
    remaining = file_size - extreme_hot_count - very_hot_count - variable_count
    for i in range(remaining):
        # Create definite outliers
        base_temp = np.random.normal(mean_threshold + 50, 25.0)  # Extreme temperatures
        outlier_boost = np.random.exponential(40)  # Big outlier boost
        temp = base_temp + outlier_boost
        temperatures.append(temp)
    
    # Ensure some extreme outliers exist
    outlier_indices = random.sample(range(len(temperatures)), min(int(file_size * 0.3), len(temperatures)))
    for idx in outlier_indices:
        temperatures[idx] += np.random.exponential(50)  # Massive outlier boost
    
    # Realistic bounds
    temperatures = [np.clip(t, 320.0, 600.0) for t in temperatures]
    
    return temperatures

def create_mega_balanced_dataset(target_total_samples=1000000, leak_ratio=0.4):
    """
    Create MEGA balanced dataset with 1M+ samples for all hole sizes
    """
    np.random.seed(42)
    random.seed(42)
    
    hole_sizes = ['20mm', '25mm', '30mm', '40mm']
    all_data = []
    
    print(f"🚀 MEGA BALANCED DATASET GENERATOR")
    print(f"=" * 80)
    print(f"🎯 Target total samples: {target_total_samples:,}")
    print(f"📊 Target leak ratio: {leak_ratio:.1%}")  
    print(f"🕳️ Hole sizes: {len(hole_sizes)} ({', '.join(hole_sizes)})")
    
    # Calculate samples per hole size
    samples_per_hole = target_total_samples // len(hole_sizes)
    print(f"📈 Samples per hole size: {samples_per_hole:,}")
    print()
    
    global_file_id = 0
    start_time = time.time()
    
    for hole_idx, hole_size in enumerate(hole_sizes):
        print(f"🔧 Processing {hole_size} ({hole_idx+1}/{len(hole_sizes)})...")
        hole_start_time = time.time()
        
        # Calculate files needed
        avg_file_size = 100
        files_needed = samples_per_hole // avg_file_size
        leak_files_needed = int(files_needed * leak_ratio)
        no_leak_files_needed = files_needed - leak_files_needed
        
        print(f"   🎯 Target: {leak_files_needed} leak files + {no_leak_files_needed} no-leak files = {files_needed} total")
        
        # Generate NO-LEAK files
        print(f"   🟢 Generating NO-LEAK files...")
        no_leak_generated = 0
        no_leak_attempts = 0
        
        while no_leak_generated < no_leak_files_needed and no_leak_attempts < no_leak_files_needed * 3:
            no_leak_attempts += 1
            file_size = np.random.randint(80, 120)
            
            # Random spatial location
            x_base = np.random.uniform(-3.0, 3.0)
            y_base = np.random.uniform(-3.0, 3.0)
            
            # Generate NO-LEAK temperatures
            temperatures = generate_guaranteed_no_leak_temperatures(hole_size, file_size)
            
            # Test classification with FIXED logic
            classification = fixed_streamlit_classification(temperatures, hole_size, 'balanced')
            
            if classification['leak_status'] == 0:  # Successfully NO-LEAK
                # Add to dataset
                for i, temp in enumerate(temperatures):
                    x_coord = x_base + np.random.normal(0, 0.05)
                    y_coord = y_base + np.random.normal(0, 0.05)
                    
                    all_data.append({
                        'nodenumber': len(all_data) + 1,
                        'x-coordinate': f"{x_coord:.9E}",
                        'y-coordinate': f"{y_coord:.9E}",
                        'temperature': f"{temp:.9E}",
                        'hole_size': hole_size,
                        'file_id': global_file_id,
                        'expected_class': 'no-leak'
                    })
                
                no_leak_generated += 1
                global_file_id += 1
                
                if no_leak_generated % 100 == 0:
                    print(f"     ✅ {no_leak_generated}/{no_leak_files_needed} no-leak files generated")
        
        print(f"   ✅ NO-LEAK completed: {no_leak_generated} files")
        
        # Generate LEAK files
        print(f"   🔴 Generating LEAK files...")
        leak_generated = 0
        leak_attempts = 0
        
        while leak_generated < leak_files_needed and leak_attempts < leak_files_needed * 3:
            leak_attempts += 1
            file_size = np.random.randint(90, 150)
            
            # Clustered hotspot locations
            if np.random.random() < 0.7:
                hotspots = [(1.5, 1.2), (-1.8, 0.9), (0.6, -1.7), (-2.1, -1.3), (2.3, -0.8)]
                hotspot = random.choice(hotspots)
                x_base = np.random.normal(hotspot[0], 0.3)
                y_base = np.random.normal(hotspot[1], 0.3)
            else:
                x_base = np.random.uniform(-3.0, 3.0)
                y_base = np.random.uniform(-3.0, 3.0)
            
            # Generate LEAK temperatures
            temperatures = generate_guaranteed_leak_temperatures(hole_size, file_size)
            
            # Test classification with FIXED logic
            classification = fixed_streamlit_classification(temperatures, hole_size, 'balanced')
            
            if classification['leak_status'] == 1:  # Successfully LEAK
                # Add to dataset
                for i, temp in enumerate(temperatures):
                    x_coord = x_base + np.random.normal(0, 0.08)
                    y_coord = y_base + np.random.normal(0, 0.08)
                    
                    all_data.append({
                        'nodenumber': len(all_data) + 1,
                        'x-coordinate': f"{x_coord:.9E}",
                        'y-coordinate': f"{y_coord:.9E}",
                        'temperature': f"{temp:.9E}",
                        'hole_size': hole_size,
                        'file_id': global_file_id,
                        'expected_class': 'leak'
                    })
                
                leak_generated += 1
                global_file_id += 1
                
                if leak_generated % 100 == 0:
                    print(f"     🔥 {leak_generated}/{leak_files_needed} leak files generated")
            else:
                if leak_attempts % 200 == 0:
                    print(f"     ⏳ Attempt {leak_attempts}: Generated file classified as no-leak")
        
        print(f"   ✅ LEAK completed: {leak_generated} files")
        
        hole_time = time.time() - hole_start_time
        print(f"   ⏱️ {hole_size} completed in {hole_time:.1f}s")
        print()
    
    # Create final DataFrame
    if not all_data:
        print("❌ ERROR: No data generated!")
        return pd.DataFrame(), {}
    
    print(f"📊 Creating final dataset...")
    df = pd.DataFrame(all_data)
    
    # Keep only the 4 required columns
    final_df = df[['nodenumber', 'x-coordinate', 'y-coordinate', 'temperature']].copy()
    
    # Shuffle dataset
    print(f"🔀 Shuffling dataset...")
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    final_df['nodenumber'] = range(1, len(final_df) + 1)
    
    # Validation
    print(f"🔍 Final validation...")
    df_with_meta = df.copy()
    
    validation_results = {}
    total_leak_samples = 0
    total_no_leak_samples = 0
    
    for hole_size in hole_sizes:
        hole_data = df_with_meta[df_with_meta['hole_size'] == hole_size]
        
        # Test a sample of files to verify classification
        unique_files = hole_data['file_id'].unique()
        sample_files = random.sample(list(unique_files), min(50, len(unique_files)))
        
        correct_leak = 0
        correct_no_leak = 0
        tested_leak = 0
        tested_no_leak = 0
        
        for file_id in sample_files:
            file_subset = hole_data[hole_data['file_id'] == file_id]
            expected = file_subset['expected_class'].iloc[0]
            temperatures = file_subset['temperature'].astype(float).values
            
            classification = fixed_streamlit_classification(temperatures, hole_size, 'balanced')
            predicted_leak = classification['leak_status'] == 1
            
            if expected == 'leak':
                tested_leak += 1
                if predicted_leak:
                    correct_leak += 1
            else:
                tested_no_leak += 1
                if not predicted_leak:
                    correct_no_leak += 1
        
        # Count samples
        hole_leak_samples = len(hole_data[hole_data['expected_class'] == 'leak'])
        hole_no_leak_samples = len(hole_data[hole_data['expected_class'] == 'no-leak'])
        
        total_leak_samples += hole_leak_samples
        total_no_leak_samples += hole_no_leak_samples
        
        validation_results[hole_size] = {
            'samples': len(hole_data),
            'leak_samples': hole_leak_samples,
            'no_leak_samples': hole_no_leak_samples,
            'leak_ratio': hole_leak_samples / len(hole_data),
            'tested_files': len(sample_files),
            'correct_leak': correct_leak,
            'correct_no_leak': correct_no_leak,
            'tested_leak': tested_leak,
            'tested_no_leak': tested_no_leak
        }
        
        print(f"   {hole_size}: {hole_leak_samples:,} leak + {hole_no_leak_samples:,} no-leak = {len(hole_data):,}")
        print(f"     Sample validation: {correct_leak}/{tested_leak} leak files, {correct_no_leak}/{tested_no_leak} no-leak files correct")
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 MEGA DATASET COMPLETED!")
    print(f"=" * 80)
    print(f"📊 FINAL STATISTICS:")
    print(f"   Total samples: {len(final_df):,}")
    print(f"   Leak samples: {total_leak_samples:,} ({total_leak_samples/len(final_df):.1%})")
    print(f"   No-leak samples: {total_no_leak_samples:,} ({total_no_leak_samples/len(final_df):.1%})")
    print(f"   Hole sizes: {len(hole_sizes)}")
    print(f"   Generation time: {total_time:.1f}s")
    
    temp_values = final_df['temperature'].astype(float)
    print(f"   Temperature range: {temp_values.min():.1f}K - {temp_values.max():.1f}K")
    print(f"   Mean temperature: {temp_values.mean():.1f}K")
    print(f"=" * 80)
    
    return final_df, validation_results

def save_mega_dataset(df, validation_results, filename="mega_balanced_dataset.csv"):
    """Save the mega dataset"""
    
    if df.empty:
        print("❌ Cannot save empty dataset!")
        return False
    
    print(f"\n💾 Saving mega dataset...")
    df.to_csv(filename, index=False)
    
    print(f"✅ MEGA DATASET SAVED!")
    print(f"📁 Filename: {filename}")
    print(f"📏 Shape: {df.shape}")
    print(f"📊 Columns: {list(df.columns)}")
    
    # Save validation summary
    summary_filename = filename.replace('.csv', '_validation_summary.txt')
    with open(summary_filename, 'w') as f:
        f.write("MEGA BALANCED DATASET VALIDATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write("FIXED CLASSIFICATION PARAMETERS:\n")
        f.write("- Base threshold: 320K (increased from 305K)\n")
        f.write("- Outlier threshold: >2.5 std, >25% ratio\n")
        f.write("- High temp ratio: >30%\n") 
        f.write("- Mean threshold: base + 5K\n")
        f.write("- Std threshold: 15K * factor\n")
        f.write("- Max threshold: base + 20K\n")
        f.write("- Decision: requires 4/5 criteria\n\n")
        
        for hole_size, stats in validation_results.items():
            f.write(f"{hole_size.upper()}:\n")
            f.write(f"  Samples: {stats['samples']:,}\n")
            f.write(f"  Leak: {stats['leak_samples']:,} ({stats['leak_ratio']:.1%})\n")
            f.write(f"  No-leak: {stats['no_leak_samples']:,}\n")
            f.write(f"  Validation: {stats['correct_leak']}/{stats['tested_leak']} leak, {stats['correct_no_leak']}/{stats['tested_no_leak']} no-leak correct\n\n")
    
    print(f"📋 Validation summary: {summary_filename}")
    
    return True

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("🚀 MEGA BALANCED DATASET GENERATOR")
    print("FIXED classification logic for proper balance!")
    print()
    
    # Generate 1M sample mega dataset
    dataset, validation = create_mega_balanced_dataset(
        target_total_samples=1000000,  # 1 MILLION samples!
        leak_ratio=0.4  # 40% leaks, 60% no-leaks
    )
    
    if not dataset.empty:
        # Show samples
        print(f"\n📋 Dataset Sample:")
        print(dataset.head(10).to_string(index=False))
        
        # Save the mega dataset
        save_mega_dataset(dataset, validation, "mega_balanced_dataset_1M.csv")
        
        print(f"\n🎉 SUCCESS! 1M+ sample dataset created!")
        print(f"🎉 Upload 'mega_balanced_dataset_1M.csv' to your Streamlit app!")
        print(f"🎉 GUARANTEED: Both leak and no-leak samples included!")
    else:
        print("❌ Failed to generate dataset!")