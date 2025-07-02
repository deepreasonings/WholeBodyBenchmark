#!/usr/bin/env python3
import os
import pandas as pd
import glob
import csv
import re

# 评估维度
DIMENSIONS = [
    'subject_consistency', 
    'background_consistency', 
    'motion_smoothness', 
    'dynamic_degree', 
    'aesthetic_quality', 
    'imaging_quality'
]

def ensure_dir(directory):
    """确保目录存在，不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def read_unmatched_info(csv_path):
    """读取CSV文件中unmatched=1的视频信息，返回一个字典，记录需要排除的视频"""
    unmatched_videos = {}
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # 跳过标题行
            
            for row in reader:
                if len(row) < 7:  # 确保有足够的列, 包括unmatched列
                    continue  # 跳过不完整的行
                    
                model_num = row[0]
                video_type = row[1]
                video_path = row[2]
                unmatched = row[6] if len(row) > 6 else "0"
                
                # 提取视频文件名（去掉扩展名和路径）
                video_name = os.path.basename(video_path)
                video_id = os.path.splitext(video_name)[0]
                
                # 如果unmatched=1，记录需要排除的视频
                if unmatched == "1":
                    if model_num not in unmatched_videos:
                        unmatched_videos[model_num] = {}
                        
                    if video_type not in unmatched_videos[model_num]:
                        unmatched_videos[model_num][video_type] = []
                        
                    unmatched_videos[model_num][video_type].append(video_id)
        
        # 打印一下找到的不匹配视频信息，方便调试
        for model, types in unmatched_videos.items():
            for type_name, videos in types.items():
                print(f"模型 {model}, 类型 {type_name} 有 {len(videos)} 个需要排除的视频")
        
        return unmatched_videos
    except Exception as e:
        print(f"读取unmatched信息出错: {e}")
        return {}

def parse_filename(file_path):
    """解析文件名，提取模型号和类型"""
    file_name = os.path.basename(file_path)
    
    # 处理原始得分文件，如 1_raw_score.csv 或 4-mp4_raw_score.csv
    if "_raw_score.csv" in file_name:
        # 提取模型号
        model_match = re.match(r'(\d+(?:-mp4)?)', file_name)
        if model_match:
            model_num = model_match.group(1)
            return model_num, "raw"
    
    # 处理其他类型文件，如 1-face-score.csv 或 4-mp4-hand-score.csv
    parts = file_name.split('-')
    if len(parts) >= 2:
        model_num = parts[0]
        # 检查是否有特殊模型号
        if len(parts) > 2 and parts[1] == "mp4":
            model_num = f"{model_num}-mp4"
            score_type = parts[2]
        else:
            score_type = parts[1]
        
        # 清理score_type（移除可能的后缀如"-score.csv"）
        score_type = score_type.replace("-score.csv", "").replace("score.csv", "")
        return model_num, score_type
    
    return None, None

def process_csv_file(file_path, unmatched_videos, output_dir):
    """处理单个CSV文件，剔除unmatched=1的视频，计算平均值并保存结果"""
    # 解析文件名，获取模型号和类型
    model_num, score_type = parse_filename(file_path)
    
    if not model_num or not score_type:
        print(f"无法从文件名 {os.path.basename(file_path)} 解析模型号和类型，跳过处理")
        return None
    
    print(f"处理文件: {os.path.basename(file_path)}, 模型号: {model_num}, 类型: {score_type}")
    
    # 读取CSV文件
    try:
        df = pd.read_csv(file_path)
        
        # 创建一份副本，用于保存剔除后的数据
        filtered_df = df.copy()
        
        # 检查CSV中的video_id列名格式
        video_id_col = None
        for col in filtered_df.columns:
            if col.lower() in ['video_id', 'video_name', 'videoname', 'videoid']:
                video_id_col = col
                break
        
        if not video_id_col and 'video_id' in filtered_df.columns:
            video_id_col = 'video_id'
        elif not video_id_col:
            print(f"警告：无法找到视频ID列，尝试使用第一列作为视频ID")
            if len(filtered_df.columns) > 0:
                video_id_col = filtered_df.columns[0]
        
        # 检查并剔除unmatched=1的视频
        excluded_count = 0
        if model_num in unmatched_videos and video_id_col:
            original_count = len(filtered_df)
            
            # 对于每种类型的视频
            for video_type in unmatched_videos[model_num]:
                # 基于当前CSV文件的类型，只处理相关类型
                if (score_type == "face" and video_type == "face") or \
                   (score_type == "hand" and video_type == "hand") or \
                   (score_type == "raw" and video_type == "body"):
                    
                    # 获取要排除的视频ID列表
                    exclude_videos = unmatched_videos[model_num][video_type]
                    
                    # 调试信息
                    print(f"当前文件类型: {score_type}, 排除类型: {video_type}")
                    print(f"排除前行数: {len(filtered_df)}")
                    print(f"需要排除的视频数量: {len(exclude_videos)}")
                    
                    # 遍历要排除的视频ID
                    for video_id in exclude_videos:
                        # 从groundtruth提取视频的基本名称（例如"video1"）
                        video_base = video_id
                        
                        # 检查CSV中的格式（可能是"video1"、"video1.mp4"或其他格式）
                        # 尝试几种可能的格式进行匹配
                        for pattern in [f"^{video_base}$", f"^{video_base}\.mp4$", f"^{video_base}-"]:
                            # 获取过滤前的行数
                            before_count = len(filtered_df)
                            
                            # 使用正则表达式匹配
                            filtered_df = filtered_df[~filtered_df[video_id_col].str.contains(pattern, regex=True)]
                            
                            # 计算本次排除的行数
                            current_excluded = before_count - len(filtered_df)
                            if current_excluded > 0:
                                excluded_count += current_excluded
                                print(f"  - 使用模式 '{pattern}' 排除了 {current_excluded} 行")
            
            # 打印总体排除情况
            if original_count > len(filtered_df):
                print(f"总共排除了 {original_count - len(filtered_df)} 个unmatched视频")
        else:
            print(f"没有找到模型 {model_num} 的unmatched信息或CSV中没有视频ID列")
        
        # 计算平均值，保留全0行数据
        if len(filtered_df) > 0:
            mean_values = {}
            for dim in DIMENSIONS:
                if dim in filtered_df.columns:
                    # 计算所有行的平均值，包括0值行
                    mean_values[dim] = filtered_df[dim].mean()
                else:
                    mean_values[dim] = 0.0
            
            # 打印平均值供参考
            print("计算得到的平均值:")
            for dim, value in mean_values.items():
                print(f"  - {dim}: {value:.4f}")
        else:
            # 如果没有有效数据，设置所有维度为0
            mean_values = {dim: 0.0 for dim in DIMENSIONS}
            print("警告：没有有效数据，所有维度的平均值设置为0")
        
        # 生成输出文件名
        output_file = os.path.join(output_dir, f"{model_num}_{score_type}_filtered.csv")
        
        # 保存过滤后的数据
        filtered_df.to_csv(output_file, index=False)
        print(f"已保存过滤后的数据到 {output_file}")
        
        return {
            'model': model_num,
            'type': score_type,
            'mean_values': mean_values,
            'sample_count': len(filtered_df)
        }
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_csv_formats(input_dir):
    """分析CSV文件格式，检查每个CSV的结构和视频ID格式"""
    print("\n=== 分析CSV文件格式 ===")
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    for file_path in csv_files:
        if "_filtered.csv" in file_path or "all_models_summary.csv" in file_path:
            continue
            
        try:
            df = pd.read_csv(file_path)
            file_name = os.path.basename(file_path)
            print(f"\n文件: {file_name}")
            print(f"列数: {len(df.columns)}, 行数: {len(df)}")
            print(f"列名: {', '.join(df.columns.tolist())}")
            
            # 检查是否有视频ID列
            potential_id_cols = []
            for col in df.columns:
                if col.lower() in ['video_id', 'video_name', 'videoname', 'videoid'] or "video" in col.lower():
                    potential_id_cols.append(col)
            
            if potential_id_cols:
                print(f"可能的视频ID列: {', '.join(potential_id_cols)}")
                # 检查第一列的值格式
                first_col = potential_id_cols[0]
                if len(df) > 0:
                    first_values = df[first_col].iloc[:5].tolist()
                    print(f"示例值: {first_values}")
            else:
                print("警告：未找到明确的视频ID列")
                
        except Exception as e:
            print(f"分析文件 {file_path} 时出错: {e}")
    
    print("\n=== 文件格式分析完成 ===\n")

def main():
    # 输入和输出目录
    input_dir = "SCres"
    ensure_dir(input_dir)  # 确保目录存在
    
    # 首先分析CSV文件格式
    analyze_csv_formats(input_dir)
    
    # 读取unmatched信息
    unmatched_videos = read_unmatched_info("groundtruth_results.csv")
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 存储所有模型和类型的均值结果
    all_results = []
    
    # 处理每个CSV文件
    for file_path in csv_files:
        # 跳过之前处理生成的文件
        if "_filtered.csv" in file_path or "all_models_summary.csv" in file_path:
            continue
            
        result = process_csv_file(file_path, unmatched_videos, input_dir)
        if result:
            all_results.append(result)
    
    # 生成汇总结果
    if all_results:
        # 创建DataFrame
        summary_data = []
        for result in all_results:
            row = {'Model': result['model'], 'Type': result['type'], 'SampleCount': result['sample_count']}
            # 添加各个维度的平均值
            for dim in DIMENSIONS:
                row[dim] = result['mean_values'][dim]
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # 保存汇总结果
        summary_file = os.path.join(input_dir, "all_models_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"汇总结果已保存到 {summary_file}")
    else:
        print("没有生成任何结果，请检查输入文件和筛选条件")

if __name__ == "__main__":
    main() 