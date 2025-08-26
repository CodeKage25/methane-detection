import numpy as np
import pandas as pd
import random
import time
from pathlib import Path

def generate_realistic_dataset(total_samples=50000, leak_ratio=0.4):
    """
    Generate realistic methane leak dataset with explicit labels
    
    Args:
        total_samples: Total number of samples to generate
        leak_ratio: Proportion of leak samples (0.0 to 1.0)
    
    Returns:
        pandas.DataFrame: Dataset with temperature readings and leak labels
    """
    
    print(f"Generating realistic dataset with {total_samples:,} samples...")
    print(f"Target leak ratio: {leak_ratio:.1%}")
    
    np.random.seed(42)
    random.seed(42)
    
    data = []
    hole_sizes = ['20mm', '25mm', '30mm', '40mm']
    
    n_leak = int(total_samples * leak_ratio)
    n_no_leak = total_samples - n_leak
    
    print(f"Leak samples: {n_leak:,}")
    print(f"No-leak samples: {n_no_leak:,}")
    
    # Generate leak samples (elevated temperatures)
    print("\nGenerating leak samples...")
    for i in range(n_leak):
        hole_size = random.choice(hole_sizes)
        
        # Hole size affects temperature elevation
        hole_factors = {'20mm': 1.0, '25mm': 1.2, '30mm': 1.5, '40mm': 1.8}
        temp_boost = hole_factors[hole_size]
        
        # Base leak temperature: elevated but realistic
        base_temp = np.random.normal(312 + (temp_boost * 3), 6 + temp_boost)
        
        # Add some spatial clustering for leaks (realistic hotspots)
        if np.random.random() < 0.7:  # 70% clustered
            hotspots = [(1.2, 0.8), (-1.5, 1.1), (0.5, -1.3), (-2.0, -0.9)]
            center = random.choice(hotspots)
            x = np.random.normal(center[0], 0.4)
            y = np.random.normal(center[1], 0.4)
        else:  # 30% random distribution
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
        
        # Ensure realistic temperature bounds
        temperature = np.clip(base_temp, 305, 380)
        
        data.append({
            'nodenumber': len(data) + 1,
            'x-coordinate': f"{x:.6f}",
            'y-coordinate': f"{y:.6f}",
            'temperature': f"{temperature:.6f}",
            'hole_size': hole_size,
            'leak_status': 1
        })
        
        if (i + 1) % 5000 == 0:
            print(f"  Generated {i+1:,}/{n_leak:,} leak samples")
    
    # Generate no-leak samples (ambient temperatures)
    print("\nGenerating no-leak samples...")
    for i in range(n_no_leak):
        hole_size = random.choice(hole_sizes)
        
        # Normal ambient temperature with small variation
        base_temp = np.random.normal(300, 2.5)
        
        # Random spatial distribution for no-leak
        x = np.random.uniform(-3, 3)
        y = np.random.uniform(-3, 3)
        
        # Ensure realistic temperature bounds
        temperature = np.clip(base_temp, 295, 308)
        
        data.append({
            'nodenumber': len(data) + 1,
            'x-coordinate': f"{x:.6f}",
            'y-coordinate': f"{y:.6f}",
            'temperature': f"{temperature:.6f}",
            'hole_size': hole_size,
            'leak_status': 0
        })
        
        if (i + 1) % 5000 == 0:
            print(f"  Generated {i+1:,}/{n_no_leak:,} no-leak samples")
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['nodenumber'] = range(1, len(df) + 1)
    
    return df

def create_file_based_dataset(df, samples_per_file_range=(80, 150), output_dir="generated_data"):
    """
    Split dataset into individual files similar to original data structure
    
    Args:
        df: Generated dataset
        samples_per_file_range: Min/max samples per file
        output_dir: Directory to save files
    
    Returns:
        dict: Information about generated files
    """
    
    print(f"\nCreating file-based dataset structure...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Group by hole size and leak status
    file_info = {}
    
    for hole_size in df['hole_size'].unique():
        hole_data = df[df['hole_size'] == hole_size]
        hole_dir = output_path / hole_size
        hole_dir.mkdir(exist_ok=True)
        
        file_info[hole_size] = {'leak_files': 0, 'no_leak_files': 0, 'total_samples': 0}
        
        # Process leak files
        leak_data = hole_data[hole_data['leak_status'] == 1]
        current_idx = 0
        file_num = 1
        
        while current_idx < len(leak_data):
            file_size = np.random.randint(*samples_per_file_range)
            file_subset = leak_data.iloc[current_idx:current_idx + file_size]
            
            if len(file_subset) > 0:
                filename = hole_dir / f"{hole_size}_leak_file_{file_num:03d}.dat"
                
                # Save in expected format (space-separated, no headers)
                with open(filename, 'w') as f:
                    for _, row in file_subset.iterrows():
                        f.write(f"{row['nodenumber']} {row['x-coordinate']} {row['y-coordinate']} {row['temperature']}\n")
                
                file_info[hole_size]['leak_files'] += 1
                file_info[hole_size]['total_samples'] += len(file_subset)
                file_num += 1
                current_idx += file_size
        
        # Process no-leak files
        no_leak_data = hole_data[hole_data['leak_status'] == 0]
        current_idx = 0
        file_num = 1
        
        while current_idx < len(no_leak_data):
            file_size = np.random.randint(*samples_per_file_range)
            file_subset = no_leak_data.iloc[current_idx:current_idx + file_size]
            
            if len(file_subset) > 0:
                filename = hole_dir / f"{hole_size}_no_leak_file_{file_num:03d}.dat"
                
                # Save in expected format (space-separated, no headers)
                with open(filename, 'w') as f:
                    for _, row in file_subset.iterrows():
                        f.write(f"{row['nodenumber']} {row['x-coordinate']} {row['y-coordinate']} {row['temperature']}\n")
                
                file_info[hole_size]['no_leak_files'] += 1
                file_info[hole_size]['total_samples'] += len(file_subset)
                file_num += 1
                current_idx += file_size
    
    return file_info

def save_single_csv_dataset(df, filename="realistic_methane_dataset.csv"):
    """
    Save complete dataset as single CSV file for Streamlit upload
    
    Args:
        df: Generated dataset
        filename: Output filename
    """
    
    # Keep only the required columns for Streamlit
    output_df = df[['nodenumber', 'x-coordinate', 'y-coordinate', 'temperature']].copy()
    
    print(f"\nSaving dataset to {filename}...")
    output_df.to_csv(filename, index=False)
    
    print(f"Dataset saved successfully!")
    print(f"Shape: {output_df.shape}")
    print(f"Columns: {list(output_df.columns)}")
    
    # Show statistics
    temp_values = output_df['temperature'].astype(float)
    print(f"\nDataset Statistics:")
    print(f"Temperature range: {temp_values.min():.1f}K - {temp_values.max():.1f}K")
    print(f"Mean temperature: {temp_values.mean():.1f}K")
    print(f"Std temperature: {temp_values.std():.1f}K")
    
    # Calculate actual leak ratio using simple threshold
    threshold = 308.0  # Simple threshold for validation
    high_temp_samples = (temp_values > threshold).sum()
    actual_leak_ratio = high_temp_samples / len(temp_values)
    print(f"Samples above {threshold}K: {high_temp_samples:,} ({actual_leak_ratio:.1%})")

def validate_dataset_balance(df, threshold=308.0):
    """
    Validate that the dataset will create balanced classes in Streamlit
    
    Args:
        df: Generated dataset
        threshold: Temperature threshold for leak detection
    """
    
    print(f"\nValidating dataset balance...")
    print(f"Using threshold: {threshold}K")
    
    temp_values = df['temperature'].astype(float)
    
    # Simple threshold-based classification
    predicted_leaks = temp_values > threshold
    leak_count = predicted_leaks.sum()
    no_leak_count = len(predicted_leaks) - leak_count
    leak_ratio = leak_count / len(predicted_leaks)
    
    print(f"Predicted leak samples: {leak_count:,} ({leak_ratio:.1%})")
    print(f"Predicted no-leak samples: {no_leak_count:,} ({(1-leak_ratio):.1%})")
    
    # Check balance quality
    if 0.3 <= leak_ratio <= 0.7:
        print("Balance Quality: EXCELLENT - Good class balance")
    elif 0.2 <= leak_ratio <= 0.8:
        print("Balance Quality: GOOD - Acceptable class balance")
    else:
        print("Balance Quality: POOR - Imbalanced classes")
        print("Consider adjusting temperature ranges or thresholds")
    
    # Test with different hole sizes
    print(f"\nBalance by hole size:")
    for hole_size in sorted(df['hole_size'].unique()):
        subset = df[df['hole_size'] == hole_size]
        subset_temps = subset['temperature'].astype(float)
        subset_leaks = (subset_temps > threshold).sum()
        subset_ratio = subset_leaks / len(subset_temps)
        print(f"  {hole_size}: {subset_leaks:,}/{len(subset_temps):,} ({subset_ratio:.1%})")

# Main execution
if __name__ == "__main__":
    print("Realistic Methane Leak Dataset Generator")
    print("=" * 50)
    
    # Generate the dataset
    dataset = generate_realistic_dataset(
        total_samples=50000,  # 50K samples for good balance
        leak_ratio=0.4        # 40% leak, 60% no-leak
    )
    
    # Validate balance
    validate_dataset_balance(dataset, threshold=308.0)
    
    # Save as single CSV for easy Streamlit upload
    save_single_csv_dataset(dataset, "realistic_methane_dataset.csv")
    
    # Optional: Create file-based structure
    create_files = input("\nCreate individual files? (y/n): ").lower().strip()
    if create_files == 'y':
        file_info = create_file_based_dataset(dataset)
        
        print(f"\nFile-based dataset created:")
        for hole_size, info in file_info.items():
            print(f"{hole_size}: {info['leak_files']} leak files, {info['no_leak_files']} no-leak files")
            print(f"  Total samples: {info['total_samples']:,}")
    
    print(f"\nDataset generation complete!")
    print(f"Upload 'realistic_methane_dataset.csv' to your Streamlit app")