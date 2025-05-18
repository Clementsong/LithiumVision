# -*- coding: utf-8 -*-

"""
筛选候选材料

此脚本从分析结果中筛选出有潜力的锂离子超导体候选材料，
根据稳定性、新颖性等指标进行排序，并输出结果。
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('filter')

def parse_arguments():
    parser = argparse.ArgumentParser(description='筛选有潜力的锂离子超导体候选材料')
    
    parser.add_argument('--input_csv', type=str, required=False,
                        help='输入CSV文件路径，包含分析过的结构信息')
    
    parser.add_argument('--input_dir', type=str, required=False,
                        help='输入目录路径，将使用该目录下的analyzed_structures.csv文件')
    
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径，默认为results/candidates/top_candidates.csv')
    
    parser.add_argument('--top_n', type=int, default=20,
                        help='输出的顶级候选材料数量，默认: 20')
    
    parser.add_argument('--e_hull_max', type=float, default=0.1,
                        help='能量上凸包最大值(eV/atom)，用于筛选稳定结构，默认: 0.1')
    
    parser.add_argument('--only_novel', action='store_true',
                        help='仅包含新颖结构，默认: False')
    
    parser.add_argument('--only_containing_li', action='store_true',
                        help='仅包含含锂结构，默认: False')
    
    parser.add_argument('--multiple_files', action='store_true',
                        help='是否处理多个分析结果文件，默认: False')
    
    parser.add_argument('--search_dir', type=str, default=None,
                        help='搜索分析结果文件的目录，当--multiple_files为True时使用')
    
    return parser.parse_args()

def setup_input_output_paths(args):
    """设置输入和输出路径"""
    # 获取项目根目录
    project_root = Path(__file__).resolve().parent.parent
    
    # 设置输入路径
    input_paths = []
    if args.input_csv:
        input_paths.append(Path(args.input_csv))
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        input_file = input_dir / "analyzed_structures.csv"
        if input_file.exists():
            input_paths.append(input_file)
        else:
            logger.error(f"未在 {input_dir} 找到 analyzed_structures.csv 文件")
            sys.exit(1)
    elif args.multiple_files and args.search_dir:
        search_dir = Path(args.search_dir)
        for csv_file in search_dir.glob('**/analyzed_structures.csv'):
            input_paths.append(csv_file)
        if not input_paths:
            logger.error(f"在 {search_dir} 及其子目录中未找到任何 analyzed_structures.csv 文件")
            sys.exit(1)
    else:
        # 默认在data/analyzed目录下查找
        analyzed_dir = project_root / "data" / "analyzed"
        if not analyzed_dir.exists():
            logger.error(f"目录 {analyzed_dir} 不存在，请指定输入文件或目录")
            sys.exit(1)
        
        for csv_file in analyzed_dir.glob('**/analyzed_structures.csv'):
            input_paths.append(csv_file)
        
        if not input_paths:
            logger.error(f"在 {analyzed_dir} 及其子目录中未找到任何 analyzed_structures.csv 文件")
            sys.exit(1)
    
    # 设置输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = project_root / "results" / "candidates" / "top_candidates.csv"
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    return input_paths, output_path

def load_data(input_paths):
    """加载数据并合并"""
    all_data = []
    
    for input_path in input_paths:
        try:
            df = pd.read_csv(input_path)
            # 添加来源信息
            df['source_file'] = input_path.stem
            df['source_dir'] = input_path.parent.name
            all_data.append(df)
            logger.info(f"从 {input_path} 加载了 {len(df)} 条记录")
        except Exception as e:
            logger.error(f"加载 {input_path} 失败: {e}")
    
    if not all_data:
        logger.error("未能加载任何数据")
        sys.exit(1)
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"共加载了 {len(combined_df)} 条记录")
    
    return combined_df

def preprocess_data(df):
    """预处理数据"""
    # 处理列表类型的字段
    for col in ['elements', 'mp_ids']:
        if col in df.columns:
            try:
                df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
            except Exception as e:
                logger.warning(f"处理 {col} 列时出错: {e}")
    
    # 将mp_matches列转换为列表（如果存在）
    if 'mp_matches' in df.columns:
        try:
            df['mp_matches'] = df['mp_matches'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        except Exception as e:
            logger.warning(f"处理 mp_matches 列时出错: {e}")
    
    # 确保布尔类型字段正确
    for col in ['is_novel', 'is_stable']:
        if col in df.columns:
            try:
                df[col] = df[col].astype(bool)
            except Exception as e:
                logger.warning(f"将 {col} 转换为布尔类型时出错: {e}")
    
    # 添加包含锂的指示器
    df['contains_li'] = df['elements'].apply(lambda x: 'Li' in x if isinstance(x, list) else False)
    
    return df

def filter_and_score_candidates(df, args):
    """根据条件筛选候选材料并评分"""
    # 应用筛选条件
    filtered_df = df.copy()
    
    # 筛选含锂的结构（如果需要）
    if args.only_containing_li:
        filtered_df = filtered_df[filtered_df['contains_li']]
        logger.info(f"筛选后剩余 {len(filtered_df)} 条含锂结构记录")
    
    # 筛选新颖结构（如果需要）
    if args.only_novel:
        filtered_df = filtered_df[filtered_df['is_novel']]
        logger.info(f"筛选后剩余 {len(filtered_df)} 条新颖结构记录")
    
    # 计算稳定性分数
    # 对于已知结构：基于energy_above_hull评分
    # 对于新颖结构：假设接近生成时的能量目标
    def stability_score(row):
        # 如果有MP匹配并有energy_above_hull
        if row.get('e_above_hull') is not None:
            # 针对不同范围的能量上凸包给予不同分数
            e_hull = float(row['e_above_hull'])
            if e_hull <= 0.01:  # 极其稳定
                return 100
            elif e_hull <= 0.05:  # 非常稳定
                return 90 - (e_hull - 0.01) * 1000  # 80-90分
            elif e_hull <= args.e_hull_max:  # 稳定
                return 70 - (e_hull - 0.05) * 400  # 50-70分
            else:  # 不太稳定
                return max(0, 50 - (e_hull - args.e_hull_max) * 200)  # 0-50分
        # 新颖结构，没有MP匹配
        elif row.get('is_novel', False):
            # 查看来源目录以估计生成时的目标energy_above_hull
            source_dir = row.get('source_dir', '')
            if 'ehull' in source_dir:
                try:
                    # 尝试从目录名中提取energy_above_hull值
                    e_hull_str = source_dir.split('ehull_')[1].split('_')[0]
                    e_hull_target = float(e_hull_str)
                    # 基于目标值评分，但给予较低的信心度
                    if e_hull_target <= 0.01:
                        return 70  # 较低信心，最高70分
                    elif e_hull_target <= 0.05:
                        return 60 - (e_hull_target - 0.01) * 500  # 40-60分
                    elif e_hull_target <= args.e_hull_max:
                        return 30 - (e_hull_target - 0.05) * 100  # 20-30分
                    else:
                        return max(0, 20 - (e_hull_target - args.e_hull_max) * 100)  # 0-20分
                except (IndexError, ValueError):
                    return 20  # 无法确定目标值，给予默认分数
            else:
                return 20  # 无法确定目标值，给予默认分数
        else:
            return 0  # 无法评估稳定性
    
    # 应用稳定性评分
    filtered_df['stability_score'] = filtered_df.apply(stability_score, axis=1)
    
    # 添加总分（可以结合多个指标）
    filtered_df['total_score'] = filtered_df['stability_score']
    
    # 根据总分排序
    sorted_df = filtered_df.sort_values('total_score', ascending=False)
    
    # 获取前N名候选材料
    top_candidates = sorted_df.head(args.top_n)
    
    return top_candidates, sorted_df

def save_results(top_candidates, all_sorted, output_path):
    """保存筛选结果"""
    # 保存顶级候选材料
    top_candidates.to_csv(output_path, index=False)
    logger.info(f"已将 {len(top_candidates)} 个顶级候选材料保存到 {output_path}")
    
    # 保存所有排序后的候选材料
    all_sorted_path = output_path.parent / f"all_candidates_{output_path.stem}.csv"
    all_sorted.to_csv(all_sorted_path, index=False)
    logger.info(f"已将全部 {len(all_sorted)} 个排序候选材料保存到 {all_sorted_path}")
    
    # 生成摘要统计信息
    summary = {
        "total_candidates": len(all_sorted),
        "top_candidates": len(top_candidates),
        "novel_candidates": int(all_sorted['is_novel'].sum()),
        "stable_candidates": int(all_sorted['is_stable'].sum()),
        "novel_and_stable": int((all_sorted['is_novel'] & all_sorted['is_stable']).sum()),
        "containing_li": int(all_sorted['contains_li'].sum()),
        "score_statistics": {
            "min": float(all_sorted['total_score'].min()),
            "max": float(all_sorted['total_score'].max()),
            "mean": float(all_sorted['total_score'].mean()),
            "median": float(all_sorted['total_score'].median()),
            "std": float(all_sorted['total_score'].std())
        }
    }
    
    # 保存摘要统计信息
    summary_path = output_path.parent / f"candidates_summary_{output_path.stem}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"已将摘要统计信息保存到 {summary_path}")
    
    # 输出简要摘要到控制台
    logger.info("\n候选材料筛选摘要：")
    logger.info(f"总候选材料数：{summary['total_candidates']}")
    logger.info(f"顶级候选材料数：{summary['top_candidates']}")
    logger.info(f"新颖候选材料数：{summary['novel_candidates']}")
    logger.info(f"稳定候选材料数：{summary['stable_candidates']}")
    logger.info(f"新颖且稳定的候选材料数：{summary['novel_and_stable']}")
    logger.info(f"含锂候选材料数：{summary['containing_li']}")
    logger.info(f"分数范围：{summary['score_statistics']['min']:.2f} - {summary['score_statistics']['max']:.2f}")
    logger.info(f"平均分数：{summary['score_statistics']['mean']:.2f}")
    
    # 显示顶级候选材料的基本信息
    logger.info("\n顶级候选材料：")
    top_info = top_candidates[['formula', 'is_novel', 'is_stable', 'e_above_hull', 'stability_score']].head(5)
    for idx, row in top_info.iterrows():
        e_hull_str = f"{row['e_above_hull']:.4f}" if pd.notna(row['e_above_hull']) else "N/A"
        logger.info(f"{row['formula']}: 新颖={row['is_novel']}, 稳定={row['is_stable']}, E_hull={e_hull_str}, 分数={row['stability_score']:.2f}")
    
    return summary_path, all_sorted_path

def main():
    args = parse_arguments()
    
    # 设置输入和输出路径
    input_paths, output_path = setup_input_output_paths(args)
    
    # 加载数据
    df = load_data(input_paths)
    
    # 预处理数据
    df = preprocess_data(df)
    
    # 筛选并评分候选材料
    top_candidates, all_sorted = filter_and_score_candidates(df, args)
    
    # 保存结果
    save_results(top_candidates, all_sorted, output_path)
    
    logger.info("候选材料筛选完成")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 