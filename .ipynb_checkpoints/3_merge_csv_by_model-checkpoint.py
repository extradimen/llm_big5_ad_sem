#!/usr/bin/env python3
"""
合并相同模型的CSV文件
按模型名称分组，每个模型生成2个文件（promotion和prevention）
"""

import os
import pandas as pd
import glob
from pathlib import Path

def merge_csv_by_model():
    """合并相同模型的CSV文件"""
    
    # 设置路径
    csv_input_dir = "./csv_output"
    csv_output_dir = "./csv_output_merged"
    
    # 创建输出目录
    os.makedirs(csv_output_dir, exist_ok=True)
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(csv_input_dir, "*.csv"))
    
    if not csv_files:
        print("❌ 没有找到CSV文件")
        return
    
    print(f"🔍 找到 {len(csv_files)} 个CSV文件")
    
    # 按模型名称分组
    model_groups = {}
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        # 提取模型名称（去掉最后的_数字）
        if '_' in filename:
            parts = filename.rsplit('_', 1)  # 从右边分割一次
            if len(parts) == 2 and parts[1].replace('.csv', '').isdigit():
                model_name = parts[0]
                if model_name not in model_groups:
                    model_groups[model_name] = []
                model_groups[model_name].append(csv_file)
    
    print(f"📊 找到 {len(model_groups)} 个不同的模型")
    
    # 处理每个模型组
    for model_name, files in model_groups.items():
        print(f"\n📁 处理模型: {model_name} ({len(files)} 个文件)")
        
        # 读取并合并所有文件
        all_dataframes = []
        
        for file_path in sorted(files):
            try:
                df = pd.read_csv(file_path)
                # 添加来源文件信息
                df['source_file'] = os.path.basename(file_path)
                all_dataframes.append(df)
                print(f"  ✅ 读取: {os.path.basename(file_path)} ({len(df)} 条记录)")
            except Exception as e:
                print(f"  ❌ 读取失败: {os.path.basename(file_path)} - {e}")
        
        if not all_dataframes:
            print(f"  ⚠️ 模型 {model_name} 没有有效数据")
            continue
        
        # 合并所有数据
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"  📊 合并后总计: {len(merged_df)} 条记录")
        
        # 检查ad_type的分布
        ad_type_counts = merged_df['ad_type'].value_counts()
        print(f"  📈 ad_type分布: {dict(ad_type_counts)}")
        
        # 按ad_type分组并保存
        for ad_type in merged_df['ad_type'].unique():
            if pd.isna(ad_type):
                continue
                
            # 筛选数据
            filtered_df = merged_df[merged_df['ad_type'] == ad_type].copy()
            
            # 移除临时列
            if 'source_file' in filtered_df.columns:
                filtered_df = filtered_df.drop('source_file', axis=1)
            
            # 生成输出文件名
            output_filename = f"{model_name}_{ad_type}.csv"
            output_path = os.path.join(csv_output_dir, output_filename)
            
            # 保存文件
            filtered_df.to_csv(output_path, index=False)
            print(f"  💾 保存: {output_filename} ({len(filtered_df)} 条记录)")
    
    print(f"\n✅ 合并完成！输出目录: {csv_output_dir}")
    
    # 显示最终统计
    print("\n📊 最终统计:")
    merged_files = glob.glob(os.path.join(csv_output_dir, "*.csv"))
    for file_path in sorted(merged_files):
        filename = os.path.basename(file_path)
        df = pd.read_csv(file_path)
        print(f"  {filename}: {len(df)} 条记录")

def analyze_merged_data():
    """分析合并后的数据"""
    
    csv_output_dir = "./csv_output_merged"
    
    if not os.path.exists(csv_output_dir):
        print("❌ 合并输出目录不存在")
        return
    
    print("\n🔍 数据分析:")
    
    # 获取所有合并后的文件
    merged_files = glob.glob(os.path.join(csv_output_dir, "*.csv"))
    
    if not merged_files:
        print("❌ 没有找到合并后的文件")
        return
    
    # 按模型分组统计
    model_stats = {}
    
    for file_path in merged_files:
        filename = os.path.basename(file_path)
        # 提取模型名称和ad_type
        parts = filename.replace('.csv', '').rsplit('_', 1)
        if len(parts) == 2:
            model_name, ad_type = parts
            if model_name not in model_stats:
                model_stats[model_name] = {}
            
            df = pd.read_csv(file_path)
            model_stats[model_name][ad_type] = len(df)
    
    # 显示统计结果
    print("\n📈 各模型数据统计:")
    for model_name, stats in model_stats.items():
        print(f"  {model_name}:")
        for ad_type, count in stats.items():
            print(f"    {ad_type}: {count} 条记录")
        total = sum(stats.values())
        print(f"    总计: {total} 条记录")

if __name__ == "__main__":
    print("🚀 开始合并CSV文件...")
    merge_csv_by_model()
    analyze_merged_data()
    print("\n🎉 处理完成！")
