#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
完整工作流脚本

此脚本按顺序执行整个LithiumVision工作流程，包括：
生成、分析、筛选和可视化步骤。
"""

import os
import sys
import argparse
import logging
import json
import subprocess
from pathlib import Path
import time
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('workflow')

def parse_arguments():
    parser = argparse.ArgumentParser(description='运行LithiumVision完整工作流程')
    
    parser.add_argument('--config_dir', type=str, default=None,
                        help='配置文件目录，默认为configs/')
    
    parser.add_argument('--generation_config', type=str, default='generation_configs.json',
                        help='生成配置文件名，默认为generation_configs.json')
    
    parser.add_argument('--analysis_config', type=str, default='analysis_configs.json',
                        help='分析配置文件名，默认为analysis_configs.json')
    
    parser.add_argument('--chemical_systems', type=str, nargs='+', default=None,
                        help='要处理的化学体系列表，默认为配置文件中的所有体系')
    
    parser.add_argument('--e_hull_targets', type=float, nargs='+', default=None,
                        help='目标energy_above_hull值列表，默认为配置文件中的值')
    
    parser.add_argument('--num_samples', type=int, default=None,
                        help='每个生成任务的样本数量，默认为配置文件中的值')
    
    parser.add_argument('--skip_generation', action='store_true',
                        help='跳过生成步骤，仅执行分析和筛选')
    
    parser.add_argument('--skip_analysis', action='store_true',
                        help='跳过分析步骤，仅执行生成和筛选')
    
    parser.add_argument('--skip_filtering', action='store_true',
                        help='跳过筛选步骤，仅执行生成和分析')
    
    parser.add_argument('--skip_visualization', action='store_true',
                        help='跳过可视化步骤')
    
    parser.add_argument('--mp_api_key', type=str, default=None,
                        help='Materials Project API密钥，优先级高于环境变量和配置文件')
    
    parser.add_argument('--output_report', action='store_true',
                        help='在工作流程结束时生成总结报告')
    
    return parser.parse_args()

def load_configs(args):
    """加载配置文件"""
    # 获取项目根目录
    project_root = Path(__file__).resolve().parent.parent
    
    # 设置配置文件目录
    if args.config_dir:
        config_dir = Path(args.config_dir)
    else:
        config_dir = project_root / "configs"
    
    # 加载生成配置
    generation_config_path = config_dir / args.generation_config
    if not generation_config_path.exists():
        logger.error(f"生成配置文件 {generation_config_path} 不存在")
        sys.exit(1)
    
    with open(generation_config_path, 'r', encoding='utf-8') as f:
        generation_config = json.load(f)
    
    # 加载分析配置
    analysis_config_path = config_dir / args.analysis_config
    if not analysis_config_path.exists():
        logger.error(f"分析配置文件 {analysis_config_path} 不存在")
        sys.exit(1)
    
    with open(analysis_config_path, 'r', encoding='utf-8') as f:
        analysis_config = json.load(f)
    
    # 应用命令行参数覆盖配置
    if args.chemical_systems:
        # 过滤配置中的化学体系
        filtered_systems = []
        for sys_config in generation_config.get("chemical_systems", []):
            if sys_config.get("name") in args.chemical_systems:
                filtered_systems.append(sys_config)
        generation_config["chemical_systems"] = filtered_systems
    
    if args.e_hull_targets:
        for sys_config in generation_config.get("chemical_systems", []):
            sys_config["e_hull_targets"] = args.e_hull_targets
    
    if args.num_samples:
        for sys_config in generation_config.get("chemical_systems", []):
            sys_config["num_samples"] = args.num_samples
    
    if args.mp_api_key:
        analysis_config["default"]["mp_api_key"] = args.mp_api_key
    
    return generation_config, analysis_config

def run_generation(generation_config):
    """运行生成步骤"""
    logger.info("开始生成步骤")
    
    # 获取项目根目录
    project_root = Path(__file__).resolve().parent.parent
    
    # 获取默认配置
    default_config = generation_config.get("default", {})
    batch_config = generation_config.get("batch_generation", {})
    
    # 准备输出基础目录
    output_base_dir = batch_config.get("output_base_dir", "data/generated")
    if not Path(output_base_dir).is_absolute():
        output_base_dir = project_root / output_base_dir
    
    Path(output_base_dir).mkdir(parents=True, exist_ok=True)
    
    # 遍历每个化学体系
    results = []
    for system_config in generation_config.get("chemical_systems", []):
        chem_sys = system_config.get("name")
        if not chem_sys:
            logger.warning("跳过未指定名称的化学体系")
            continue
        
        logger.info(f"处理化学体系: {chem_sys}")
        
        # 遍历每个目标energy_above_hull值
        for e_hull in system_config.get("e_hull_targets", [0.0]):
            logger.info(f"处理目标energy_above_hull值: {e_hull}")
            
            # 获取生成参数
            pretrained_model = system_config.get("pretrained_model", default_config.get("pretrained_model", "chemical_system_energy_above_hull"))
            batch_size = system_config.get("batch_size", default_config.get("batch_size", 8))
            guidance_scale = system_config.get("guidance_scale", default_config.get("guidance_scale", 1.0))
            record_trajectories = system_config.get("record_trajectories", default_config.get("record_trajectories", False))
            num_samples = system_config.get("num_samples", 100)
            
            # 构建输出目录名
            dir_name = f"{chem_sys.replace('-', '')}_ehull_{e_hull}"
            output_dir = Path(output_base_dir) / dir_name
            
            # 构建命令
            cmd = [
                sys.executable,
                str(project_root / "scripts" / "generate_structures.py"),
                f"--chem_sys={chem_sys}",
                f"--e_hull={e_hull}",
                f"--num_samples={num_samples}",
                f"--batch_size={batch_size}",
                f"--guidance_scale={guidance_scale}",
                f"--pretrained_model={pretrained_model}",
                f"--output_dir={output_dir}"
            ]
            
            if record_trajectories:
                cmd.append("--record_trajectories")
            
            # 运行命令
            logger.info(f"执行命令: {' '.join(cmd)}")
            try:
                start_time = time.time()
                process = subprocess.run(cmd, check=True, capture_output=True, text=True)
                elapsed_time = time.time() - start_time
                
                logger.info(f"生成完成，耗时 {elapsed_time:.2f} 秒")
                
                # 记录结果
                result = {
                    "chem_sys": chem_sys,
                    "e_hull": e_hull,
                    "output_dir": str(output_dir),
                    "success": True,
                    "elapsed_time": elapsed_time
                }
                results.append(result)
                
            except subprocess.CalledProcessError as e:
                logger.error(f"生成失败: {e}")
                logger.error(f"标准输出: {e.stdout}")
                logger.error(f"标准错误: {e.stderr}")
                
                # 记录失败结果
                result = {
                    "chem_sys": chem_sys,
                    "e_hull": e_hull,
                    "output_dir": str(output_dir),
                    "success": False,
                    "error": str(e)
                }
                results.append(result)
    
    # 保存所有生成结果
    results_path = project_root / "results" / "tables" / "generation_workflow_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"生成步骤完成，结果已保存到 {results_path}")
    
    return results

def run_analysis(generation_results, analysis_config):
    """运行分析步骤"""
    logger.info("开始分析步骤")
    
    # 获取项目根目录
    project_root = Path(__file__).resolve().parent.parent
    
    # 获取默认配置
    default_config = analysis_config.get("default", {})
    
    # 提取MP API密钥
    mp_api_key = default_config.get("mp_api_key")
    if mp_api_key is None:
        mp_api_key = os.environ.get("MP_API_KEY")
        if mp_api_key:
            logger.info("使用环境变量中的MP API密钥")
        else:
            logger.warning("未设置MP API密钥，将跳过MP数据查询")
    
    # 设置其他参数
    max_workers = default_config.get("max_workers", 4)
    structure_matcher = default_config.get("structure_matcher", False)
    e_hull_threshold = default_config.get("e_hull_threshold", 0.2)
    
    # 遍历每个生成结果
    results = []
    for gen_result in generation_results:
        if not gen_result.get("success", False):
            logger.warning(f"跳过失败的生成结果: {gen_result.get('chem_sys')}, {gen_result.get('e_hull')}")
            continue
        
        input_dir = gen_result.get("output_dir")
        if not input_dir:
            logger.warning(f"跳过缺少输出目录的生成结果: {gen_result.get('chem_sys')}, {gen_result.get('e_hull')}")
            continue
        
        logger.info(f"分析生成结果: {input_dir}")
        
        # 构建命令
        cmd = [
            sys.executable,
            str(project_root / "scripts" / "analyze_structures.py"),
            f"--input_dir={input_dir}",
            f"--max_workers={max_workers}",
            f"--e_hull_threshold={e_hull_threshold}"
        ]
        
        if mp_api_key:
            cmd.append(f"--mp_api_key={mp_api_key}")
        
        if structure_matcher:
            cmd.append("--structure_matcher")
        
        # 运行命令
        logger.info(f"执行命令: {' '.join(cmd)}")
        try:
            start_time = time.time()
            process = subprocess.run(cmd, check=True, capture_output=True, text=True)
            elapsed_time = time.time() - start_time
            
            logger.info(f"分析完成，耗时 {elapsed_time:.2f} 秒")
            
            # 确定输出目录
            project_root = Path(__file__).resolve().parent.parent
            input_path = Path(input_dir)
            dir_name = input_path.name
            output_dir = project_root / "data" / "analyzed" / dir_name
            
            # 记录结果
            result = {
                "input_dir": input_dir,
                "output_dir": str(output_dir),
                "success": True,
                "elapsed_time": elapsed_time
            }
            results.append(result)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"分析失败: {e}")
            logger.error(f"标准输出: {e.stdout}")
            logger.error(f"标准错误: {e.stderr}")
            
            # 记录失败结果
            result = {
                "input_dir": input_dir,
                "success": False,
                "error": str(e)
            }
            results.append(result)
    
    # 保存所有分析结果
    results_path = project_root / "results" / "tables" / "analysis_workflow_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"分析步骤完成，结果已保存到 {results_path}")
    
    return results

def run_filtering(analysis_results, analysis_config):
    """运行筛选步骤"""
    logger.info("开始筛选步骤")
    
    # 获取项目根目录
    project_root = Path(__file__).resolve().parent.parent
    
    # 获取筛选配置
    filter_config = analysis_config.get("filtering", {})
    
    # 设置参数
    e_hull_max = filter_config.get("e_hull_max", 0.1)
    only_novel = filter_config.get("only_novel", False)
    only_containing_li = filter_config.get("only_containing_li", True)
    
    # 整理分析目录
    analyzed_dirs = []
    for result in analysis_results:
        if result.get("success", False) and result.get("output_dir"):
            analyzed_dirs.append(result.get("output_dir"))
    
    if not analyzed_dirs:
        logger.warning("没有成功的分析结果，跳过筛选步骤")
        return None
    
    # 构建命令
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "filter_candidates.py"),
        f"--e_hull_max={e_hull_max}",
        "--multiple_files"
    ]
    
    if len(analyzed_dirs) == 1:
        cmd.append(f"--input_dir={analyzed_dirs[0]}")
    else:
        cmd.append(f"--search_dir={project_root / 'data' / 'analyzed'}")
    
    if only_novel:
        cmd.append("--only_novel")
    
    if only_containing_li:
        cmd.append("--only_containing_li")
    
    # 运行命令
    logger.info(f"执行命令: {' '.join(cmd)}")
    try:
        start_time = time.time()
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed_time = time.time() - start_time
        
        logger.info(f"筛选完成，耗时 {elapsed_time:.2f} 秒")
        
        # 确定输出文件
        output_file = project_root / "results" / "candidates" / "top_candidates.csv"
        
        # 记录结果
        result = {
            "analyzed_dirs": analyzed_dirs,
            "output_file": str(output_file),
            "success": True,
            "elapsed_time": elapsed_time
        }
        
    except subprocess.CalledProcessError as e:
        logger.error(f"筛选失败: {e}")
        logger.error(f"标准输出: {e.stdout}")
        logger.error(f"标准错误: {e.stderr}")
        
        # 记录失败结果
        result = {
            "analyzed_dirs": analyzed_dirs,
            "success": False,
            "error": str(e)
        }
    
    # 保存筛选结果
    results_path = project_root / "results" / "tables" / "filtering_workflow_result.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"筛选步骤完成，结果已保存到 {results_path}")
    
    return result

def run_visualization(filtering_result, analysis_config):
    """运行可视化步骤"""
    logger.info("开始可视化步骤")
    
    if not filtering_result or not filtering_result.get("success", False):
        logger.warning("筛选步骤失败，跳过可视化步骤")
        return None
    
    # 获取项目根目录
    project_root = Path(__file__).resolve().parent.parent
    
    # 获取可视化配置
    vis_config = analysis_config.get("visualization", {})
    
    # 设置参数
    top_candidates = vis_config.get("top_candidates", 10)
    format = vis_config.get("format", "png")
    dpi = vis_config.get("dpi", 300)
    
    # 获取筛选结果文件
    candidates_file = filtering_result.get("output_file")
    if not candidates_file or not Path(candidates_file).exists():
        logger.warning(f"候选材料文件 {candidates_file} 不存在，跳过可视化步骤")
        return None
    
    # 构建命令
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "visualize_structures.py"),
        f"--candidates={candidates_file}",
        f"--top={top_candidates}",
        f"--format={format}",
        f"--dpi={dpi}"
    ]
    
    # 运行命令
    logger.info(f"执行命令: {' '.join(cmd)}")
    try:
        start_time = time.time()
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed_time = time.time() - start_time
        
        logger.info(f"可视化完成，耗时 {elapsed_time:.2f} 秒")
        
        # 确定输出目录
        output_dir = project_root / "results" / "figures" / "structures"
        
        # 记录结果
        result = {
            "candidates_file": candidates_file,
            "output_dir": str(output_dir),
            "success": True,
            "elapsed_time": elapsed_time
        }
        
    except subprocess.CalledProcessError as e:
        logger.error(f"可视化失败: {e}")
        logger.error(f"标准输出: {e.stdout}")
        logger.error(f"标准错误: {e.stderr}")
        
        # 记录失败结果
        result = {
            "candidates_file": candidates_file,
            "success": False,
            "error": str(e)
        }
    
    # 保存可视化结果
    results_path = project_root / "results" / "tables" / "visualization_workflow_result.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"可视化步骤完成，结果已保存到 {results_path}")
    
    return result

def generate_summary_report():
    """生成总结报告"""
    logger.info("生成总结报告")
    
    # 获取项目根目录
    project_root = Path(__file__).resolve().parent.parent
    
    # 构建命令
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "summarize_results.py"),
        "--report_format=md"
    ]
    
    # 运行命令
    logger.info(f"执行命令: {' '.join(cmd)}")
    try:
        start_time = time.time()
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed_time = time.time() - start_time
        
        logger.info(f"报告生成完成，耗时 {elapsed_time:.2f} 秒")
        
        # 确定输出文件
        output_file = project_root / "results" / "tables" / "LithiumVision_summary_report.md"
        
        if output_file.exists():
            logger.info(f"总结报告已保存到 {output_file}")
            return True
        else:
            logger.warning(f"未找到生成的报告文件 {output_file}")
            return False
        
    except subprocess.CalledProcessError as e:
        logger.error(f"报告生成失败: {e}")
        logger.error(f"标准输出: {e.stdout}")
        logger.error(f"标准错误: {e.stderr}")
        return False

def main():
    args = parse_arguments()
    
    # 加载配置
    generation_config, analysis_config = load_configs(args)
    
    # 记录工作流开始时间
    workflow_start_time = time.time()
    
    # 执行生成步骤
    generation_results = None
    if not args.skip_generation:
        generation_results = run_generation(generation_config)
    else:
        logger.info("跳过生成步骤")
        # 尝试从之前的结果文件中加载
        result_path = Path(__file__).resolve().parent.parent / "results" / "tables" / "generation_workflow_results.json"
        if result_path.exists():
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    generation_results = json.load(f)
                logger.info(f"从 {result_path} 加载了之前的生成结果")
            except Exception as e:
                logger.warning(f"无法加载之前的生成结果: {e}")
    
    # 执行分析步骤
    analysis_results = None
    if not args.skip_analysis and generation_results:
        analysis_results = run_analysis(generation_results, analysis_config)
    elif args.skip_analysis:
        logger.info("跳过分析步骤")
        # 尝试从之前的结果文件中加载
        result_path = Path(__file__).resolve().parent.parent / "results" / "tables" / "analysis_workflow_results.json"
        if result_path.exists():
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    analysis_results = json.load(f)
                logger.info(f"从 {result_path} 加载了之前的分析结果")
            except Exception as e:
                logger.warning(f"无法加载之前的分析结果: {e}")
    
    # 执行筛选步骤
    filtering_result = None
    if not args.skip_filtering and analysis_results:
        filtering_result = run_filtering(analysis_results, analysis_config)
    elif args.skip_filtering:
        logger.info("跳过筛选步骤")
        # 尝试从之前的结果文件中加载
        result_path = Path(__file__).resolve().parent.parent / "results" / "tables" / "filtering_workflow_result.json"
        if result_path.exists():
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    filtering_result = json.load(f)
                logger.info(f"从 {result_path} 加载了之前的筛选结果")
            except Exception as e:
                logger.warning(f"无法加载之前的筛选结果: {e}")
    
    # 执行可视化步骤
    visualization_result = None
    if not args.skip_visualization and filtering_result:
        visualization_result = run_visualization(filtering_result, analysis_config)
    elif args.skip_visualization:
        logger.info("跳过可视化步骤")
    
    # 生成总结报告
    if args.output_report:
        generate_summary_report()
    
    # 计算总耗时
    workflow_elapsed_time = time.time() - workflow_start_time
    
    logger.info(f"工作流程执行完成，总耗时 {workflow_elapsed_time:.2f} 秒")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 