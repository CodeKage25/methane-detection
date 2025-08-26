import pandas as pd
import numpy as np
import random
import time
from pathlib import Path

def generate_balanced_methane_dataset(total_samples=20000, leak_ratio=0.4):
    """
    Generate realistic balanced methane leak dataset with explicit labels
    
    Args:
        total_samples: Total number of samples to generate
        leak_ratio: Proportion of leak samples (0.0 to 1.0)
    
    Returns:
        pandas.DataFrame: Dataset with temperature readings and explicit leak labels
    """
    
    print(f"Generating balanced methane dataset...")
    print(f"Total samples: {total_samples:,}")
    print(f"Target leak ratio: {leak_ratio:.1%}")
    
    np.random.seed(42)
    random.seed(42)
    
    data = []
    hole_sizes = ['20mm', '25mm', '30mm', '40mm']
    
    n_leak = int(total_samples * leak_ratio)
    n_no_leak = total_samples - n_leak
    
    print(f"\nGenerating {n_leak:,} leak samples...")
    print(f"Generating {n_no_leak:,} no-leak samples...")
    
    # Generate NO-LEAK samples (ambient/normal temperatures)
    for i in range(n_no_leak):
        hole_size = random.choice(hole_sizes)
        
        # Normal ambient temperature with small variation
        # Temperature range: 295-308K (clearly below leak threshold)
        base_temp = np.random.normal(301, 2.5)
        temperature = np.clip(base_temp, 295, 308)
        
        # Random spatial distribution for no-leak
        x = np.random.uniform(-3, 3)
        y = np.random.uniform(-3, 3)
        
        data.append({
            'nodenumber': len(data) + 1,
            'x-coordinate': f"{x:.6f}",
            'y-coordinate': f"{y:.6f}",
            'temperature': f"{temperature:.6f}",
            'hole_size': hole_size,
            'leak_status': 0  # Explicit NO LEAK label
        })
        
        if (i + 1) % 2000 == 0:
            print(f"  Generated {i+1:,}/{n_no_leak:,} no-leak samples")
    
    # Generate LEAK samples (elevated temperatures)  
    for i in range(n_leak):
        hole_size = random.choice(hole_sizes)
        
        # Hole size affects temperature elevation
        hole_factors = {'20mm': 1.0, '25mm': 1.2, '30mm': 1.5, '40mm': 1.8}
        temp_boost = hole_factors[hole_size]
        
        # Leak temperature: clearly elevated above ambient
        # Temperature range: 310-380K (clearly above no-leak threshold)
        base_temp = np.random.normal(320 + (temp_boost * 4), 8 + temp_boost)
        temperature = np.clip(base_temp, 310, 380)
        
        # Add some spatial clustering for leaks (realistic hotspots)
        if np.random.random() < 0.6:  # 60% clustered around hotspots
            hotspots = [(1.2, 0.8), (-1.5, 1.1), (0.5, -1.3), (-2.0, -0.9), (2.2, -1.5)]
            center = random.choice(hotspots)
            x = np.random.normal(center[0], 0.5)
            y = np.random.normal(center[1], 0.5)
        else:  # 40% random distribution
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
        
        data.append({
            'nodenumber': len(data) + 1,
            'x-coordinate': f"{x:.6f}",
            'y-coordinate': f"{y:.6f}",
            'temperature': f"{temperature:.6f}",
            'hole_size': hole_size,
            'leak_status': 1  # Explicit LEAK label
        })
        
        if (i + 1) % 2000 == 0:
            print(f"  Generated {i+1:,}/{n_leak:,} leak samples")
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['nodenumber'] = range(1, len(df) + 1)
    
    return df

def validate_dataset_separation(df, threshold=308.0):
    """
    Validate that the dataset has clear temperature separation
    
    Args:
        df: Generated dataset
        threshold: Temperature threshold for validation
    """
    
    print(f"\nValidating dataset separation...")
    print(f"Using threshold: {threshold}K")
    
    leak_temps = df[df['leak_status'] == 1]['temperature'].astype(float)
    no_leak_temps = df[df['leak_status'] == 0]['temperature'].astype(float)
    
    print(f"\nTemperature Statistics:")
    print(f"No-Leak samples:")
    print(f"  Count: {len(no_leak_temps):,}")
    print(f"  Range: {no_leak_temps.min():.1f}K - {no_leak_temps.max():.1f}K")
    print(f"  Mean: {no_leak_temps.mean():.1f}K")
    print(f"  Above {threshold}K: {(no_leak_temps > threshold).sum():,} ({(no_leak_temps > threshold).mean():.1%})")
    
    print(f"\nLeak samples:")
    print(f"  Count: {len(leak_temps):,}")
    print(f"  Range: {leak_temps.min():.1f}K - {leak_temps.max():.1f}K")
    print(f"  Mean: {leak_temps.mean():.1f}K")
    print(f"  Above {threshold}K: {(leak_temps > threshold).sum():,} ({(leak_temps > threshold).mean():.1%})")
    
    # Check separation quality
    overlap_no_leak = (no_leak_temps > threshold).mean()
    overlap_leak = (leak_temps <= threshold).mean()
    
    if overlap_no_leak < 0.1 and overlap_leak < 0.1:
        print(f"\nSeparation Quality: EXCELLENT - Clear temperature separation")
    elif overlap_no_leak < 0.2 and overlap_leak < 0.2:
        print(f"\nSeparation Quality: GOOD - Acceptable separation")
    else:
        print(f"\nSeparation Quality: POOR - High overlap, may cause classification issues")
        
    return {
        'no_leak_above_threshold': overlap_no_leak,
        'leak_below_threshold': overlap_leak,
        'separation_quality': 'excellent' if (overlap_no_leak < 0.1 and overlap_leak < 0.1) else 'good' if (overlap_no_leak < 0.2 and overlap_leak < 0.2) else 'poor'
    }

def save_dataset(df, filename="balanced_methane_dataset.csv"):
    """
    Save the balanced dataset to CSV
    
    Args:
        df: Generated dataset
        filename: Output filename
    """
    
    print(f"\nSaving dataset to {filename}...")
    df.to_csv(filename, index=False)
    
    print(f"Dataset saved successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Show final statistics
    leak_count = df['leak_status'].sum()
    total_count = len(df)
    leak_ratio = leak_count / total_count
    
    temp_values = df['temperature'].astype(float)
    print(f"\nFinal Dataset Statistics:")
    print(f"Total samples: {total_count:,}")
    print(f"Leak samples: {leak_count:,} ({leak_ratio:.1%})")
    print(f"No-leak samples: {total_count - leak_count:,} ({(1-leak_ratio):.1%})")
    print(f"Temperature range: {temp_values.min():.1f}K - {temp_values.max():.1f}K")
    print(f"Mean temperature: {temp_values.mean():.1f}K")
    
    # Distribution by hole size
    print(f"\nDistribution by hole size:")
    for hole_size in sorted(df['hole_size'].unique()):
        subset = df[df['hole_size'] == hole_size]
        hole_leak_count = subset['leak_status'].sum()
        hole_total = len(subset)
        hole_ratio = hole_leak_count / hole_total
        print(f"  {hole_size}: {hole_leak_count:,}/{hole_total:,} leaks ({hole_ratio:.1%})")

def create_test_scenarios():
    """Create additional test datasets for different scenarios"""
    
    scenarios = {
        'high_leak_ratio': {'total_samples': 15000, 'leak_ratio': 0.7},
        'low_leak_ratio': {'total_samples': 15000, 'leak_ratio': 0.2},
        'large_dataset': {'total_samples': 50000, 'leak_ratio': 0.4},
        'small_dataset': {'total_samples': 5000, 'leak_ratio': 0.4}
    }
    
    print(f"\nCreating test scenarios...")
    
    for scenario_name, params in scenarios.items():
        print(f"\nGenerating {scenario_name}...")
        df = generate_balanced_methane_dataset(**params)
        filename = f"methane_dataset_{scenario_name}.csv"
        save_dataset(df, filename)

# Main execution
if __name__ == "__main__":
    print("Balanced Methane Leak Dataset Generator")
    print("=" * 60)
    
    # Generate main balanced dataset
    print("\n1. Generating main balanced dataset...")
    dataset = generate_balanced_methane_dataset(
        total_samples=20000,  # 20K samples 
        leak_ratio=0.4        # 40% leak, 60% no-leak
    )
    
    # Validate separation
    validation_results = validate_dataset_separation(dataset, threshold=308.0)
    
    # Save main dataset
    save_dataset(dataset, "balanced_methane_dataset.csv")
    
    # Generate test scenarios
    create_scenarios = input("\nGenerate additional test scenarios? (y/n): ").lower().strip()
    if create_scenarios == 'y':
        create_test_scenarios()
    
    print(f"\n" + "=" * 60)
    print(f"Dataset generation complete!")
    print(f"\nMain file: balanced_methane_dataset.csv")
    print(f"Upload this file using the 'Direct CSV Upload' option in your Streamlit app")
    print(f"\nKey features:")
    print(f"- Clear temperature separation (no-leak: ~295-308K, leak: ~310-380K)")
    print(f"- Balanced classes (40% leak, 60% no-leak)")
    print(f"- Realistic spatial clustering for leak samples")
    print(f"- Hole size variations affecting temperature ranges")
    print(f"- Explicit leak_status labels (no classification needed)")