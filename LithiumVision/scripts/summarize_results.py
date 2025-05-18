#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
汇总结果

此脚本汇总整个LithiumVision工作流程的结果，
生成综合报告和最终的候选材料清单。
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('summarize')

def parse_arguments():
    parser = argparse.ArgumentParser(description='汇总LithiumVision项目结果')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录路径，默认为results/tables')
    
    parser.add_argument('--candidates_file', type=str, default=None,
                        help='候选材料CSV文件路径，默认为最新的top_candidates.csv文件')
    
    parser.add_argument('--generation_log', type=str, default=None,
                        help='生成活动日志CSV文件路径，默认为results/tables/generation_campaign_log.csv')
    
    parser.add_argument('--analyzed_dirs', type=str, nargs='+', default=None,
                        help='分析结果目录路径列表，默认为data/analyzed下的所有目录')
    
    parser.add_argument('--visualizations_dir', type=str, default=None,
                        help='可视化图像目录路径，默认为results/figures/structures')
    
    parser.add_argument('--report_format', type=str, default='md',
                        choices=['md', 'html', 'txt'],
                        help='报告格式，默认: md')
    
    return parser.parse_args()

def setup_paths(args):
    """设置输入和输出路径"""
    # 获取项目根目录
    project_root = Path(__file__).resolve().parent.parent
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "results" / "tables"
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置候选材料文件路径
    if args.candidates_file:
        candidates_file = Path(args.candidates_file)
    else:
        # 查找最新的候选材料文件
        candidates_dir = project_root / "results" / "candidates"
        candidates_files = list(candidates_dir.glob("top_candidates*.csv"))
        if candidates_files:
            # 按修改时间排序，选择最新的
            candidates_file = sorted(candidates_files, key=lambda x: x.stat().st_mtime)[-1]
        else:
            candidates_file = None
            logger.warning("未找到候选材料文件")
    
    # 设置生成活动日志路径
    if args.generation_log:
        generation_log = Path(args.generation_log)
    else:
        generation_log = project_root / "results" / "tables" / "generation_campaign_log.csv"
    
    # 设置分析结果目录
    if args.analyzed_dirs:
        analyzed_dirs = [Path(d) for d in args.analyzed_dirs]
    else:
        # 查找data/analyzed下的所有目录
        analyzed_dirs = []
        analyzed_root = project_root / "data" / "analyzed"
        if analyzed_root.exists():
            analyzed_dirs = [d for d in analyzed_root.iterdir() if d.is_dir()]
    
    # 设置可视化图像目录
    if args.visualizations_dir:
        visualizations_dir = Path(args.visualizations_dir)
    else:
        visualizations_dir = project_root / "results" / "figures" / "structures"
    
    return output_dir, candidates_file, generation_log, analyzed_dirs, visualizations_dir

def load_generation_log(log_path):
    """加载生成活动日志"""
    if not log_path.exists():
        logger.warning(f"生成活动日志 {log_path} 不存在")
        return None
    
    try:
        df = pd.read_csv(log_path)
        logger.info(f"从 {log_path} 加载了 {len(df)} 条生成活动记录")
        return df
    except Exception as e:
        logger.error(f"加载生成活动日志失败: {e}")
        return None

def load_candidates(candidates_file):
    """加载候选材料数据"""
    if candidates_file is None or not candidates_file.exists():
        logger.warning("未提供有效的候选材料文件")
        return None
    
    try:
        df = pd.read_csv(candidates_file)
        logger.info(f"从 {candidates_file} 加载了 {len(df)} 条候选材料记录")
        return df
    except Exception as e:
        logger.error(f"加载候选材料文件失败: {e}")
        return None

def collect_analysis_results(analyzed_dirs):
    """收集所有分析结果的摘要"""
    summaries = []
    
    for analyzed_dir in analyzed_dirs:
        summary_file = analyzed_dir / "summary_statistics.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                    summary['dir_name'] = analyzed_dir.name
                    summaries.append(summary)
                    logger.info(f"从 {summary_file} 加载了分析摘要")
            except Exception as e:
                logger.warning(f"加载分析摘要 {summary_file} 失败: {e}")
    
    logger.info(f"共加载了 {len(summaries)} 个分析摘要")
    return summaries

def collect_visualizations(visualizations_dir):
    """收集结构可视化信息"""
    if not visualizations_dir.exists():
        logger.warning(f"可视化目录 {visualizations_dir} 不存在")
        return None
    
    # 查找visualization_info.json文件
    info_file = visualizations_dir / "visualization_info.json"
    if not info_file.exists():
        logger.warning(f"可视化信息文件 {info_file} 不存在")
        return None
    
    try:
        with open(info_file, 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        # 检查summary_visualization文件是否存在
        if info.get('summary_visualization'):
            summary_path = Path(info['summary_visualization'])
            if not summary_path.exists():
                logger.warning(f"汇总可视化图像 {summary_path} 不存在")
                info['summary_visualization'] = None
        
        logger.info(f"从 {info_file} 加载了可视化信息")
        return info
    except Exception as e:
        logger.error(f"加载可视化信息失败: {e}")
        return None

def create_chemical_systems_plot(generation_log, output_dir):
    """创建化学体系分布图"""
    if generation_log is None:
        logger.warning("无法创建化学体系分布图：缺少生成活动日志")
        return None
    
    try:
        # 统计每个化学体系的生成数量
        system_counts = generation_log['目标化学体系'].value_counts()
        
        # 创建条形图
        plt.figure(figsize=(10, 6))
        system_counts.plot(kind='bar', color='skyblue')
        plt.title('化学体系分布')
        plt.xlabel('化学体系')
        plt.ylabel('生成活动数量')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图像
        output_path = output_dir / "chemical_systems_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"已创建化学体系分布图: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"创建化学体系分布图失败: {e}")
        return None

def create_e_hull_distribution_plot(analysis_summaries, output_dir):
    """创建energy_above_hull分布图"""
    if not analysis_summaries:
        logger.warning("无法创建energy_above_hull分布图：缺少分析摘要")
        return None
    
    try:
        # 收集所有的energy_above_hull值和对应的bin边界
        all_bins = []
        all_bin_edges = []
        labels = []
        
        for summary in analysis_summaries:
            if 'e_hull_statistics' in summary and 'bins' in summary['e_hull_statistics'] and 'bin_edges' in summary['e_hull_statistics']:
                bins = summary['e_hull_statistics']['bins']
                bin_edges = summary['e_hull_statistics']['bin_edges']
                
                if bins and bin_edges and len(bin_edges) > len(bins):
                    all_bins.append(bins)
                    all_bin_edges.append(bin_edges)
                    labels.append(summary['dir_name'])
        
        if not all_bins:
            logger.warning("没有找到有效的energy_above_hull分布数据")
            return None
        
        # 创建堆叠条形图
        plt.figure(figsize=(12, 7))
        
        # 使用第一个bin_edges作为x轴位置
        x = np.arange(len(all_bins[0]))
        width = 0.8 / len(all_bins)
        
        for i, (bins, label) in enumerate(zip(all_bins, labels)):
            plt.bar(x + i * width, bins, width, label=label)
        
        # 使用第一个bin_edges中点作为x轴标签
        bin_centers = [f"{(all_bin_edges[0][i] + all_bin_edges[0][i+1])/2:.3f}" for i in range(len(all_bin_edges[0])-1)]
        plt.xticks(x + width * len(all_bins) / 2, bin_centers, rotation=45)
        
        plt.title('Energy Above Hull分布')
        plt.xlabel('Energy Above Hull (eV/atom)')
        plt.ylabel('结构数量')
        plt.legend()
        plt.tight_layout()
        
        # 保存图像
        output_path = output_dir / "energy_above_hull_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"已创建Energy Above Hull分布图: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"创建Energy Above Hull分布图失败: {e}")
        return None

def create_stability_novelty_plot(analysis_summaries, output_dir):
    """创建稳定性与新颖性散点图"""
    if not analysis_summaries:
        logger.warning("无法创建稳定性与新颖性图：缺少分析摘要")
        return None
    
    try:
        # 准备数据
        data = []
        for summary in analysis_summaries:
            if 'stable_percentage' in summary and 'novel_percentage' in summary:
                data.append({
                    'dir_name': summary['dir_name'],
                    'stable_percentage': summary['stable_percentage'],
                    'novel_percentage': summary['novel_percentage'],
                    'total_structures': summary['total_structures']
                })
        
        if not data:
            logger.warning("没有找到有效的稳定性与新颖性数据")
            return None
        
        # 创建散点图
        plt.figure(figsize=(10, 8))
        
        # 获取点大小（按总结构数量缩放）
        sizes = [item['total_structures'] / 10 for item in data]
        min_size = 50
        max_size = 300
        if sizes:
            sizes = [max(min_size, min(max_size, s)) for s in sizes]
        
        # 绘制散点
        for i, item in enumerate(data):
            plt.scatter(
                item['novel_percentage'], 
                item['stable_percentage'],
                s=sizes[i],
                alpha=0.7,
                label=item['dir_name']
            )
        
        # 添加数据标签
        for i, item in enumerate(data):
            plt.annotate(
                item['dir_name'],
                (item['novel_percentage'], item['stable_percentage']),
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        plt.title('结构稳定性与新颖性')
        plt.xlabel('新颖结构百分比 (%)')
        plt.ylabel('稳定结构百分比 (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存图像
        output_path = output_dir / "stability_novelty_plot.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"已创建稳定性与新颖性图: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"创建稳定性与新颖性图失败: {e}")
        return None

def generate_summary_statistics(generation_log, candidates, analysis_summaries):
    """生成总体统计摘要"""
    stats = {
        "generation": {
            "total_activities": 0,
            "total_structures_generated": 0,
            "chemical_systems": [],
            "success_rate": 0
        },
        "analysis": {
            "total_structures_analyzed": 0,
            "total_novel_structures": 0,
            "total_stable_structures": 0,
            "novel_percentage": 0,
            "stable_percentage": 0
        },
        "candidates": {
            "total_candidates": 0,
            "top_candidates": 0,
            "novel_candidates": 0,
            "stable_candidates": 0,
            "novel_and_stable": 0
        }
    }
    
    # 生成活动统计
    if generation_log is not None:
        stats["generation"]["total_activities"] = len(generation_log)
        stats["generation"]["total_structures_generated"] = generation_log['生成的CIF数量'].sum()
        stats["generation"]["chemical_systems"] = list(generation_log['目标化学体系'].unique())
        
        successful = generation_log[generation_log['状态'] == '成功']
        if len(generation_log) > 0:
            stats["generation"]["success_rate"] = len(successful) / len(generation_log) * 100
    
    # 分析结果统计
    if analysis_summaries:
        stats["analysis"]["total_structures_analyzed"] = sum(s.get('total_structures', 0) for s in analysis_summaries)
        stats["analysis"]["total_novel_structures"] = sum(s.get('novel_structures', 0) for s in analysis_summaries)
        stats["analysis"]["total_stable_structures"] = sum(s.get('stable_structures', 0) for s in analysis_summaries)
        
        if stats["analysis"]["total_structures_analyzed"] > 0:
            stats["analysis"]["novel_percentage"] = stats["analysis"]["total_novel_structures"] / stats["analysis"]["total_structures_analyzed"] * 100
            stats["analysis"]["stable_percentage"] = stats["analysis"]["total_stable_structures"] / stats["analysis"]["total_structures_analyzed"] * 100
    
    # 候选材料统计
    if candidates is not None:
        stats["candidates"]["total_candidates"] = len(candidates)
        
        if 'is_novel' in candidates.columns:
            stats["candidates"]["novel_candidates"] = int(candidates['is_novel'].sum())
        
        if 'is_stable' in candidates.columns:
            stats["candidates"]["stable_candidates"] = int(candidates['is_stable'].sum())
        
        if 'is_novel' in candidates.columns and 'is_stable' in candidates.columns:
            stats["candidates"]["novel_and_stable"] = int((candidates['is_novel'] & candidates['is_stable']).sum())
    
    return stats

def create_markdown_report(stats, output_dir, candidates, chem_sys_plot, e_hull_plot, stability_novelty_plot, vis_info):
    """生成Markdown格式的总结报告"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 创建报告文件路径
    report_path = output_dir / "LithiumVision_summary_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # 标题
        f.write("# LithiumVision: 锂离子超导体快速筛选与评估报告\n\n")
        f.write(f"生成时间: {now}\n\n")
        
        # 摘要统计
        f.write("## 1. 总体统计摘要\n\n")
        
        # 生成活动统计
        f.write("### 1.1 生成活动统计\n\n")
        f.write(f"- **总生成活动数**: {stats['generation']['total_activities']}\n")
        f.write(f"- **总生成结构数**: {stats['generation']['total_structures_generated']}\n")
        f.write(f"- **生成成功率**: {stats['generation']['success_rate']:.2f}%\n")
        f.write(f"- **探索的化学体系**: {', '.join(stats['generation']['chemical_systems'])}\n\n")
        
        # 添加化学体系分布图
        if chem_sys_plot:
            f.write("#### 化学体系分布\n\n")
            f.write(f"![化学体系分布]({chem_sys_plot.name})\n\n")
        
        # 分析结果统计
        f.write("### 1.2 分析结果统计\n\n")
        f.write(f"- **分析的结构总数**: {stats['analysis']['total_structures_analyzed']}\n")
        f.write(f"- **新颖结构数**: {stats['analysis']['total_novel_structures']} ({stats['analysis']['novel_percentage']:.2f}%)\n")
        f.write(f"- **稳定结构数**: {stats['analysis']['total_stable_structures']} ({stats['analysis']['stable_percentage']:.2f}%)\n\n")
        
        # 添加能量分布图和稳定性-新颖性图
        if e_hull_plot:
            f.write("#### Energy Above Hull 分布\n\n")
            f.write(f"![Energy Above Hull 分布]({e_hull_plot.name})\n\n")
        
        if stability_novelty_plot:
            f.write("#### 稳定性与新颖性关系\n\n")
            f.write(f"![稳定性与新颖性]({stability_novelty_plot.name})\n\n")
        
        # 候选材料统计
        f.write("### 1.3 候选材料统计\n\n")
        f.write(f"- **总候选材料数**: {stats['candidates']['total_candidates']}\n")
        f.write(f"- **新颖候选材料数**: {stats['candidates']['novel_candidates']}\n")
        f.write(f"- **稳定候选材料数**: {stats['candidates']['stable_candidates']}\n")
        f.write(f"- **新颖且稳定的候选材料数**: {stats['candidates']['novel_and_stable']}\n\n")
        
        # 顶级候选材料列表
        f.write("## 2. 高潜力锂离子超导体候选材料\n\n")
        
        if candidates is not None and len(candidates) > 0:
            # 创建表格头
            f.write("| 排名 | 化学式 | 新颖性 | 稳定性 | E_hull (eV/atom) | 稳定性评分 |\n")
            f.write("|------|--------|--------|--------|-----------------|------------|\n")
            
            # 添加表格内容
            for idx, row in candidates.head(10).iterrows():
                formula = row.get('formula', 'N/A')
                is_novel = "✓" if row.get('is_novel', False) else "✗"
                is_stable = "✓" if row.get('is_stable', False) else "✗"
                e_hull = f"{row.get('e_above_hull', 'N/A'):.4f}" if pd.notna(row.get('e_above_hull')) else "N/A"
                score = f"{row.get('stability_score', 'N/A'):.2f}" if pd.notna(row.get('stability_score')) else "N/A"
                
                f.write(f"| {idx+1} | {formula} | {is_novel} | {is_stable} | {e_hull} | {score} |\n")
            
            f.write("\n")
        else:
            f.write("*未找到候选材料数据*\n\n")
        
        # 结构可视化
        f.write("## 3. 结构可视化\n\n")
        
        if vis_info and vis_info.get('summary_visualization'):
            summary_vis_path = Path(vis_info['summary_visualization']).name
            f.write(f"![候选材料结构]({summary_vis_path})\n\n")
            
            f.write("### 3.1 可视化统计\n\n")
            f.write(f"- **可视化的候选材料数**: {vis_info.get('total_candidates', 'N/A')}\n")
            f.write(f"- **成功的可视化数**: {vis_info.get('successful_visualizations', 'N/A')}\n")
            f.write(f"- **失败的可视化数**: {vis_info.get('failed_visualizations', 'N/A')}\n\n")
        else:
            f.write("*未找到结构可视化数据*\n\n")
        
        # 结论
        f.write("## 4. 结论与展望\n\n")
        f.write("本项目利用MatterGen生成模型和开源数据库，实现了锂离子超导体候选材料的快速筛选与评估。")
        f.write("通过系统化的生成、分析和筛选流程，我们发现了一批具有潜力的候选材料。\n\n")
        
        if stats['candidates']['novel_and_stable'] > 0:
            f.write(f"特别值得注意的是，我们发现了{stats['candidates']['novel_and_stable']}个既新颖又稳定的候选材料，")
            f.write("这些材料可能成为下一代固态电池的潜在电解质材料。\n\n")
        
        f.write("### 后续工作\n\n")
        f.write("- **深入计算验证**: 对顶级候选材料进行DFT计算，验证其稳定性和电子结构\n")
        f.write("- **离子导电率计算**: 使用AIMD或NEB方法计算候选材料的离子导电率\n")
        f.write("- **定制化模型训练**: 基于现有数据集训练针对锂离子导体的专用模型\n")
        f.write("- **实验合成与表征**: 与实验团队合作，尝试合成并表征最有前景的候选材料\n\n")
    
    logger.info(f"已生成Markdown摘要报告: {report_path}")
    return report_path

def main():
    args = parse_arguments()
    
    # 设置路径
    output_dir, candidates_file, generation_log, analyzed_dirs, visualizations_dir = setup_paths(args)
    
    # 加载数据
    generation_data = load_generation_log(generation_log)
    candidates_data = load_candidates(candidates_file)
    analysis_summaries = collect_analysis_results(analyzed_dirs)
    visualization_info = collect_visualizations(visualizations_dir)
    
    # 创建可视化
    chem_sys_plot = create_chemical_systems_plot(generation_data, output_dir)
    e_hull_plot = create_e_hull_distribution_plot(analysis_summaries, output_dir)
    stability_novelty_plot = create_stability_novelty_plot(analysis_summaries, output_dir)
    
    # 生成统计摘要
    summary_stats = generate_summary_statistics(generation_data, candidates_data, analysis_summaries)
    
    # 生成报告
    if args.report_format == 'md':
        report_path = create_markdown_report(
            summary_stats, output_dir, candidates_data,
            chem_sys_plot, e_hull_plot, stability_novelty_plot,
            visualization_info
        )
    # 可以添加其他格式的报告生成函数
    
    # 保存摘要统计数据
    summary_json_path = output_dir / "summary_statistics.json"
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"摘要统计数据已保存到 {summary_json_path}")
    logger.info("汇总完成")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 