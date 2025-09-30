#!/usr/bin/env python3
"""
Merge CSV files of the same model.
Group by model name and generate 2 files per model (promotion and prevention).
"""

import os
import pandas as pd
import glob
from pathlib import Path

def merge_csv_by_model():
    """Merge CSV files of the same model"""
    
    # Set paths
    csv_input_dir = "./csv_output"
    csv_output_dir = "./csv_output_merged"
    
    # Create output directory
    os.makedirs(csv_output_dir, exist_ok=True)
    
    # Collect all CSV files
    csv_files = glob.glob(os.path.join(csv_input_dir, "*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found")
        return
    
    print(f"🔍 Found {len(csv_files)} CSV files")
    
    # Group by model name
    model_groups = {}
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        # Extract model name (strip trailing _number)
        if '_' in filename:
            parts = filename.rsplit('_', 1)  # 从右边分割一次
            if len(parts) == 2 and parts[1].replace('.csv', '').isdigit():
                model_name = parts[0]
                if model_name not in model_groups:
                    model_groups[model_name] = []
                model_groups[model_name].append(csv_file)
    
    print(f"📊 Found {len(model_groups)} distinct models")
    
    # Process each model group
    for model_name, files in model_groups.items():
        print(f"\n📁 Processing model: {model_name} ({len(files)} files)")
        
        # Read and merge all files
        all_dataframes = []
        
        for file_path in sorted(files):
            try:
                df = pd.read_csv(file_path)
                # Add source file information
                df['source_file'] = os.path.basename(file_path)
                all_dataframes.append(df)
                print(f"  ✅ Read: {os.path.basename(file_path)} ({len(df)} rows)")
            except Exception as e:
                print(f"  ❌ Failed to read: {os.path.basename(file_path)} - {e}")
        
        if not all_dataframes:
            print(f"  ⚠️ Model {model_name} has no valid data")
            continue
        
        # Concatenate all data
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"  📊 After merge total: {len(merged_df)} rows")
        
        # Check ad_type distribution
        ad_type_counts = merged_df['ad_type'].value_counts()
        print(f"  📈 ad_type distribution: {dict(ad_type_counts)}")
        
        # Group by ad_type and save
        for ad_type in merged_df['ad_type'].unique():
            if pd.isna(ad_type):
                continue
                
            # Filter data
            filtered_df = merged_df[merged_df['ad_type'] == ad_type].copy()
            
            # Remove temporary column
            if 'source_file' in filtered_df.columns:
                filtered_df = filtered_df.drop('source_file', axis=1)
            
            # Generate output filename
            output_filename = f"{model_name}_{ad_type}.csv"
            output_path = os.path.join(csv_output_dir, output_filename)
            
            # Save file
            filtered_df.to_csv(output_path, index=False)
            print(f"  💾 Saved: {output_filename} ({len(filtered_df)} rows)")
    
    print(f"\n✅ Merge complete! Output dir: {csv_output_dir}")
    
    # Show final stats
    print("\n📊 Final stats:")
    merged_files = glob.glob(os.path.join(csv_output_dir, "*.csv"))
    for file_path in sorted(merged_files):
        filename = os.path.basename(file_path)
        df = pd.read_csv(file_path)
        print(f"  {filename}: {len(df)} rows")

def analyze_merged_data():
    """Analyze merged data"""
    
    csv_output_dir = "./csv_output_merged"
    
    if not os.path.exists(csv_output_dir):
        print("❌ Merged output directory does not exist")
        return
    
    print("\n🔍 Data analysis:")
    
    # 获取所有合并后的文件
    merged_files = glob.glob(os.path.join(csv_output_dir, "*.csv"))
    
    if not merged_files:
        print("❌ No merged files found")
        return
    
    # Aggregate by model
    model_stats = {}
    
    for file_path in merged_files:
        filename = os.path.basename(file_path)
        # Extract model name and ad_type
        parts = filename.replace('.csv', '').rsplit('_', 1)
        if len(parts) == 2:
            model_name, ad_type = parts
            if model_name not in model_stats:
                model_stats[model_name] = {}
            
            df = pd.read_csv(file_path)
            model_stats[model_name][ad_type] = len(df)
    
    # Display stats
    print("\n📈 Model-wise statistics:")
    for model_name, stats in model_stats.items():
        print(f"  {model_name}:")
        for ad_type, count in stats.items():
            print(f"    {ad_type}: {count} rows")
        total = sum(stats.values())
        print(f"    Total: {total} rows")

if __name__ == "__main__":
    print("🚀 Start merging CSV files...")
    merge_csv_by_model()
    analyze_merged_data()
    print("\n🎉 Done!")
