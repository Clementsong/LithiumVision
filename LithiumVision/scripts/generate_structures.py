#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用MatterGen生成晶体结构

此脚本基于指定的化学体系和目标energy_above_hull值，
使用MatterGen生成晶体结构，并将结果保存到指定目录。
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('generate')

# 检查MatterGen是否已安装
try:
    import mattergen
    logger.info(f"已加载MatterGen，版本：{getattr(mattergen, '__version__', '未知')}")
except ImportError:
    logger.error("未找到MatterGen。请确保已安装MatterGen及其依赖。")
    sys.exit(1)

def parse_arguments():
    parser = argparse.ArgumentParser(description='使用MatterGen生成锂离子导体晶体结构')
    
    parser.add_argument('--chem_sys', type=str, required=True,
                        help='目标化学体系，例如"Li-P-S"')
    
    parser.add_argument('--e_hull', type=float, default=0.05,
                        help='目标energy_above_hull值 (eV/atom)，默认: 0.05')
    
    parser.add_argument('--num_samples', type=int, default=100,
                        help='要生成的样本数量，默认: 100')
    
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批处理大小，根据GPU内存调整，默认: 8')
    
    parser.add_argument('--guidance_scale', type=float, default=1.0,
                        help='条件引导强度，默认: 1.0')
    
    parser.add_argument('--pretrained_model', type=str, default='chemical_system_energy_above_hull',
                        help='预训练模型名称，默认: chemical_system_energy_above_hull')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录，默认: data/generated/{chem_sys}_ehull_{e_hull}')
    
    parser.add_argument('--record_trajectories', action='store_true',
                        help='是否记录轨迹，默认: False')
    
    return parser.parse_args()

def create_output_dir(chem_sys, e_hull, output_dir=None):
    """创建输出目录"""
    if output_dir is None:
        # 获取项目根目录
        project_root = Path(__file__).resolve().parent.parent
        # 格式化目录名
        dir_name = f"{chem_sys.replace('-', '')}_ehull_{e_hull}"
        output_dir = project_root / "data" / "generated" / dir_name
    else:
        output_dir = Path(output_dir)
    
    # 确保目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir

def log_generation_campaign(args, output_dir, start_time, end_time=None, status="开始", cif_count=None):
    """记录生成活动信息到日志文件"""
    log_file = Path(__file__).resolve().parent.parent / "results" / "tables" / "generation_campaign_log.csv"
    
    # 如果文件不存在，创建并写入表头
    if not log_file.exists():
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("活动ID,目标化学体系,目标energy_above_hull,num_samples,guidance_scale,输出路径,开始时间,结束时间,状态,生成的CIF数量\n")
    
    # 生成唯一的活动ID
    activity_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 格式化时间
    start_time_str = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    end_time_str = "" if end_time is None else datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
    
    # 写入日志
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{activity_id},{args.chem_sys},{args.e_hull},{args.num_samples},{args.guidance_scale},"
                f"{output_dir},{start_time_str},{end_time_str},{status},{cif_count if cif_count is not None else ''}\n")
    
    return activity_id

def run_mattergen_generate(args, output_dir):
    """运行MatterGen生成命令"""
    logger.info(f"开始生成结构，输出到: {output_dir}")
    
    # 创建条件属性字典
    properties_to_condition_on = {
        'chemical_system': args.chem_sys, 
        'energy_above_hull': args.e_hull
    }
    
    # 将字典转换为JSON字符串
    properties_str = json.dumps(properties_to_condition_on)
    
    # 构建MatterGen命令
    cmd = [
        'python', '-m', 'mattergen.scripts.run',
        'mode=generate',
        f'pretrained_name={args.pretrained_model}',
        f'batch_size={args.batch_size}',
        f'num_samples_to_generate={args.num_samples}',
        f'results_path={output_dir}',
        f'properties_to_condition_on={properties_str}',
        f'guidance_scale={args.guidance_scale}',
        f'record_trajectories={str(args.record_trajectories).lower()}'
    ]
    
    # 将命令转换为字符串形式，用于日志记录
    cmd_str = ' '.join(cmd)
    logger.info(f"执行命令: {cmd_str}")
    
    # 运行命令
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("生成完成!")
        logger.debug(process.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"生成失败: {e}")
        logger.error(f"标准输出: {e.stdout}")
        logger.error(f"标准错误: {e.stderr}")
        return False

def count_generated_cifs(output_dir):
    """计算生成的CIF文件数量"""
    cif_zip = output_dir / "generated_crystals_cif.zip"
    if not cif_zip.exists():
        logger.warning(f"未找到CIF压缩包: {cif_zip}")
        return 0
    
    # 使用临时目录解压缩并计数
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            subprocess.run(['unzip', '-q', str(cif_zip), '-d', temp_dir], check=True)
            cif_files = list(Path(temp_dir).glob('**/*.cif'))
            return len(cif_files)
        except subprocess.CalledProcessError:
            logger.warning("无法解压CIF文件进行计数")
            return 0

def main():
    args = parse_arguments()
    
    # 创建输出目录
    output_dir = create_output_dir(args.chem_sys, args.e_hull, args.output_dir)
    
    # 记录生成开始信息
    start_time = time.time()
    activity_id = log_generation_campaign(args, output_dir, start_time)
    
    # 运行MatterGen生成
    success = run_mattergen_generate(args, output_dir)
    
    # 记录生成结束信息
    end_time = time.time()
    cif_count = count_generated_cifs(output_dir) if success else 0
    status = "成功" if success else "失败"
    log_generation_campaign(args, output_dir, start_time, end_time, status, cif_count)
    
    # 保存生成参数
    with open(output_dir / "generation_params.json", 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    
    if success:
        logger.info(f"成功生成了 {cif_count} 个晶体结构，保存在 {output_dir}")
        return 0
    else:
        logger.error("生成过程失败")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 