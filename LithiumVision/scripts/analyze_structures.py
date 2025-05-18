#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分析生成的晶体结构

此脚本读取MatterGen生成的CIF文件，使用Pymatgen分析其结构特性，
并通过Materials Project API查询是否存在匹配的已知材料及其性质。
"""

import os
import sys
import argparse
import logging
import json
import time
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import traceback
from io import StringIO

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('analyze')

# 检查所需库
try:
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
    from mp_api.client import MPRester
    logger.info("依赖库加载成功")
except ImportError as e:
    logger.error(f"依赖库导入失败: {e}")
    logger.error("请确保已安装pymatgen和mp-api")
    sys.exit(1)

def parse_arguments():
    parser = argparse.ArgumentParser(description='分析MatterGen生成的晶体结构')
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='包含生成的CIF文件的目录')
    
    parser.add_argument('--mp_api_key', type=str, default=None,
                        help='Materials Project API密钥，默认从环境变量MP_API_KEY获取')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='分析结果输出目录，默认为input_dir对应的data/analyzed目录')
    
    parser.add_argument('--max_workers', type=int, default=4,
                        help='并行处理的最大线程数，默认: 4')
    
    parser.add_argument('--structure_matcher', action='store_true',
                        help='是否使用StructureMatcher进行严格结构匹配（计算密集型），默认: False')
    
    parser.add_argument('--e_hull_threshold', type=float, default=0.2,
                        help='能量上凸包阈值(eV/atom)，超过此值的结构被视为不稳定，默认: 0.2')
    
    return parser.parse_args()

def setup_output_dir(input_dir, output_dir=None):
    """设置输出目录"""
    input_path = Path(input_dir)
    
    if output_dir is None:
        # 获取项目根目录
        project_root = Path(__file__).resolve().parent.parent
        # 从输入目录名提取相关信息
        dir_name = input_path.name
        output_dir = project_root / "data" / "analyzed" / dir_name
    else:
        output_dir = Path(output_dir)
    
    # 确保目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir

def get_mp_api_key(provided_key=None):
    """获取Materials Project API密钥"""
    if provided_key:
        return provided_key
    
    mp_api_key = os.environ.get('MP_API_KEY')
    if not mp_api_key:
        logger.warning("未设置Materials Project API密钥，请设置MP_API_KEY环境变量或使用--mp_api_key参数")
        return None
    
    return mp_api_key

def extract_cif_files(input_dir):
    """从ZIP文件中提取CIF文件"""
    input_path = Path(input_dir)
    cif_zip = input_path / "generated_crystals_cif.zip"
    
    if not cif_zip.exists():
        logger.error(f"未找到CIF压缩包: {cif_zip}")
        return []
    
    cif_extract_dir = input_path / "extracted_cifs"
    
    # 如果已经解压过，直接返回CIF文件列表
    if cif_extract_dir.exists():
        cif_files = list(cif_extract_dir.glob('**/*.cif'))
        logger.info(f"使用已解压的 {len(cif_files)} 个CIF文件")
        return cif_files
    
    # 解压CIF文件
    cif_extract_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(cif_zip, 'r') as zip_ref:
            logger.info(f"正在解压 {cif_zip}")
            cif_names = [name for name in zip_ref.namelist() if name.endswith('.cif')]
            for cif_name in tqdm(cif_names, desc="解压CIF文件"):
                zip_ref.extract(cif_name, cif_extract_dir)
        
        cif_files = list(cif_extract_dir.glob('**/*.cif'))
        logger.info(f"成功解压了 {len(cif_files)} 个CIF文件到 {cif_extract_dir}")
        return cif_files
    except Exception as e:
        logger.error(f"解压CIF文件失败: {e}")
        return []

def analyze_single_structure(cif_file, mp_rester, use_structure_matcher=False):
    """分析单个晶体结构并查询MP数据库"""
    try:
        # 读取CIF文件
        structure = Structure.from_file(cif_file)
        
        # 获取基本信息
        formula = structure.composition.reduced_formula
        elements = sorted([str(el) for el in structure.composition.elements])
        chemsys = "-".join(elements)
        num_sites = len(structure)
        density = structure.density
        volume = structure.volume
        
        # 初始化结果字典
        result = {
            "cif_file": str(cif_file.name),
            "formula": formula,
            "elements": elements,
            "chemsys": chemsys,
            "num_sites": num_sites,
            "density": density,
            "volume": volume,
            "mp_matches": [],
            "mp_ids": [],
            "formation_energy": None,
            "e_above_hull": None,
            "is_stable": False,
            "is_novel": True  # 默认为新颖的，除非在MP中找到匹配
        }
        
        # 如果没有MP API密钥，只返回结构信息
        if mp_rester is None:
            return result
        
        # 通过化学式在MP中查询
        docs = mp_rester.materials.summary.search(
            formula=formula,
            fields=["material_id", "formula_pretty", "formation_energy_per_atom", 
                    "energy_above_hull", "structure", "theoretical"]
        )
        
        if docs:
            result["is_novel"] = False  # 找到了化学式匹配
            
            # 遍历所有匹配项
            for doc in docs:
                mp_id = doc.material_id
                mp_structure = doc.structure
                
                # 记录匹配项信息
                match_info = {
                    "mp_id": mp_id,
                    "formula": doc.formula_pretty,
                    "formation_energy": doc.formation_energy_per_atom,
                    "e_above_hull": doc.energy_above_hull,
                    "theoretical": doc.theoretical
                }
                
                # 使用StructureMatcher进行严格结构匹配
                structure_match = False
                if use_structure_matcher:
                    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, 
                                             primitive_cell=True, scale=True, 
                                             comparator=ElementComparator())
                    structure_match = matcher.fit(structure, mp_structure)
                    match_info["structure_match"] = structure_match
                
                result["mp_matches"].append(match_info)
                result["mp_ids"].append(mp_id)
            
            # 如果有匹配的MP条目，选择能量最低的作为参考
            if result["mp_matches"]:
                # 按energy_above_hull排序
                sorted_matches = sorted(result["mp_matches"], 
                                       key=lambda x: x["e_above_hull"] if x["e_above_hull"] is not None else float('inf'))
                best_match = sorted_matches[0]
                
                result["formation_energy"] = best_match["formation_energy"]
                result["e_above_hull"] = best_match["e_above_hull"]
                result["is_stable"] = best_match["e_above_hull"] <= args.e_hull_threshold
        
        return result
    
    except Exception as e:
        logger.error(f"处理文件 {cif_file} 时出错: {e}")
        logger.debug(traceback.format_exc())
        return {
            "cif_file": str(cif_file.name),
            "error": str(e)
        }

def analyze_structures(cif_files, mp_api_key, use_structure_matcher, max_workers):
    """并行分析多个晶体结构"""
    results = []
    
    # 设置MP API
    mp_rester = None
    if mp_api_key:
        try:
            mp_rester = MPRester(mp_api_key)
            logger.info("成功连接到Materials Project API")
        except Exception as e:
            logger.error(f"连接Materials Project API失败: {e}")
    else:
        logger.warning("未提供MP API密钥，将跳过MP数据查询")
    
    # 并行处理文件
    logger.info(f"开始分析 {len(cif_files)} 个CIF文件，使用 {max_workers} 个工作线程")
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cif = {
            executor.submit(analyze_single_structure, cif_file, mp_rester, use_structure_matcher): cif_file
            for cif_file in cif_files
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_cif), 
                          total=len(cif_files), desc="分析结构"):
            cif_file = future_to_cif[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"处理 {cif_file} 的任务失败: {e}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"分析完成，耗时 {elapsed_time:.2f} 秒")
    
    return results

def generate_summary_statistics(results):
    """生成汇总统计信息"""
    total_structures = len(results)
    novel_structures = sum(1 for r in results if r.get('is_novel', False))
    known_structures = total_structures - novel_structures
    stable_structures = sum(1 for r in results if r.get('is_stable', False))
    
    # 统计化学体系分布
    chemsys_counts = {}
    for r in results:
        chemsys = r.get('chemsys')
        if chemsys:
            chemsys_counts[chemsys] = chemsys_counts.get(chemsys, 0) + 1
    
    # energy_above_hull分布（仅对已知结构）
    e_hull_values = [r.get('e_above_hull') for r in results if r.get('e_above_hull') is not None]
    
    e_hull_stats = {
        "count": len(e_hull_values),
        "min": min(e_hull_values) if e_hull_values else None,
        "max": max(e_hull_values) if e_hull_values else None,
        "mean": np.mean(e_hull_values) if e_hull_values else None,
        "median": np.median(e_hull_values) if e_hull_values else None,
        "std": np.std(e_hull_values) if e_hull_values else None,
        "bins": np.histogram(e_hull_values, bins=10)[0].tolist() if len(e_hull_values) > 0 else [],
        "bin_edges": np.histogram(e_hull_values, bins=10)[1].tolist() if len(e_hull_values) > 0 else []
    }
    
    summary = {
        "total_structures": total_structures,
        "novel_structures": novel_structures,
        "known_structures": known_structures,
        "stable_structures": stable_structures,
        "novel_percentage": novel_structures / total_structures * 100 if total_structures > 0 else 0,
        "stable_percentage": stable_structures / total_structures * 100 if total_structures > 0 else 0,
        "chemsys_distribution": chemsys_counts,
        "e_hull_statistics": e_hull_stats
    }
    
    return summary

def save_results(results, summary, output_dir):
    """保存分析结果"""
    # 将结果保存为CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / "analyzed_structures.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"分析结果已保存到 {csv_path}")
    
    # 将摘要统计信息保存为JSON
    summary_path = output_dir / "summary_statistics.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"摘要统计信息已保存到 {summary_path}")
    
    # 只保留新颖且稳定的结构
    novel_stable = df[(df['is_novel'] == True) | (df['is_stable'] == True)]
    novel_stable_path = output_dir / "novel_stable_structures.csv"
    novel_stable.to_csv(novel_stable_path, index=False)
    logger.info(f"新颖与稳定结构信息已保存到 {novel_stable_path}")
    
    # 输出简要摘要信息到控制台
    summary_string = f"""
分析摘要：
总结构数: {summary['total_structures']}
新颖结构数: {summary['novel_structures']} ({summary['novel_percentage']:.2f}%)
已知结构数: {summary['known_structures']}
稳定结构数: {summary['stable_structures']} ({summary['stable_percentage']:.2f}%)

能量上凸包统计 (eV/atom):
  数量: {summary['e_hull_statistics']['count']}
  最小值: {summary['e_hull_statistics']['min']:.4f}
  最大值: {summary['e_hull_statistics']['max']:.4f}
  平均值: {summary['e_hull_statistics']['mean']:.4f}
  中位数: {summary['e_hull_statistics']['median']:.4f}
"""
    logger.info(summary_string)
    
    return csv_path, summary_path, novel_stable_path

def main():
    global args
    args = parse_arguments()
    
    # 设置输出目录
    output_dir = setup_output_dir(args.input_dir, args.output_dir)
    
    # 获取MP API密钥
    mp_api_key = get_mp_api_key(args.mp_api_key)
    
    # 提取CIF文件
    cif_files = extract_cif_files(args.input_dir)
    if not cif_files:
        logger.error("未找到CIF文件，退出")
        return 1
    
    # 分析结构
    results = analyze_structures(cif_files, mp_api_key, args.structure_matcher, args.max_workers)
    
    # 生成摘要统计信息
    summary = generate_summary_statistics(results)
    
    # 保存结果
    save_results(results, summary, output_dir)
    
    logger.info(f"分析完成，结果已保存到 {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 