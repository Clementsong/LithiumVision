#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
环境设置脚本：检查并安装MatterGen及其依赖，配置API密钥等
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('setup')

def check_python_version():
    """检查Python版本是否满足要求"""
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 10):
        logger.error(f"当前Python版本 {python_version.major}.{python_version.minor}.{python_version.micro} "
                     f"不满足要求。需要Python 3.10+。")
        return False
    logger.info(f"Python版本检查通过: {python_version.major}.{python_version.minor}.{python_version.micro}")
    return True

def check_git_lfs():
    """检查Git LFS是否已安装"""
    try:
        subprocess.run(['git', 'lfs', '--version'], check=True, capture_output=True)
        logger.info("Git LFS检查通过")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("未检测到Git LFS，请安装后再继续")
        return False

def check_mattergen():
    """检查MatterGen是否已安装"""
    try:
        import mattergen
        logger.info(f"MatterGen已安装，版本: {getattr(mattergen, '__version__', '未知')}")
        return True
    except ImportError:
        logger.warning("未检测到MatterGen，请按照README中的说明安装")
        return False

def check_mp_api_key():
    """检查Materials Project API密钥是否设置"""
    mp_api_key = os.environ.get('MP_API_KEY')
    if not mp_api_key:
        logger.warning("未设置Materials Project API密钥，请设置MP_API_KEY环境变量")
        return False
    logger.info("Materials Project API密钥已设置")
    return True

def test_mattergen_models():
    """测试MatterGen模型是否可用"""
    try:
        # 这里应根据实际MatterGen API调整
        import mattergen
        # 假设的模型检测代码，需根据实际情况修改
        models = mattergen.list_available_models()
        logger.info(f"检测到MatterGen模型: {models}")
        return True
    except (ImportError, AttributeError) as e:
        logger.error(f"MatterGen模型测试失败: {e}")
        return False

def setup_mattergen(args):
    """设置MatterGen环境"""
    if not os.path.exists(args.mattergen_dir):
        logger.info(f"MatterGen目录不存在，正在克隆仓库到 {args.mattergen_dir}")
        subprocess.run(['git', 'clone', 'https://github.com/microsoft/mattergen.git', args.mattergen_dir], 
                      check=True)
    
    # 切换到MatterGen目录
    os.chdir(args.mattergen_dir)
    
    # 确保Git LFS文件已拉取
    subprocess.run(['git', 'lfs', 'install'], check=True)
    subprocess.run(['git', 'lfs', 'pull'], check=True)
    
    # 安装MatterGen
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], check=True)
    
    logger.info("MatterGen环境设置完成")

def setup_project_structure():
    """确保项目目录结构存在"""
    base_dir = Path(__file__).resolve().parent.parent
    directories = [
        base_dir / "data" / "generated",
        base_dir / "data" / "analyzed",
        base_dir / "results" / "figures",
        base_dir / "results" / "tables",
        base_dir / "results" / "candidates",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"确保目录存在: {directory}")

def main():
    parser = argparse.ArgumentParser(description='设置LithiumVision项目环境')
    parser.add_argument('--mattergen-dir', type=str, default='../mattergen',
                        help='MatterGen克隆目录 (默认: ../mattergen)')
    parser.add_argument('--skip-checks', action='store_true',
                        help='跳过环境检查')
    args = parser.parse_args()
    
    if not args.skip_checks:
        all_checks_passed = True
        all_checks_passed &= check_python_version()
        all_checks_passed &= check_git_lfs()
        all_checks_passed &= check_mp_api_key()
        
        if not all_checks_passed:
            logger.warning("部分环境检查未通过，请解决上述问题后再继续")
            return 1
    
    setup_project_structure()
    
    if not check_mattergen():
        try:
            setup_mattergen(args)
        except subprocess.SubprocessError as e:
            logger.error(f"MatterGen设置失败: {e}")
            return 1
    
    test_mattergen_models()
    
    logger.info("环境设置完成！可以开始使用LithiumVision了。")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 