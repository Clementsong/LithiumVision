#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Uses MatterGen to generate crystal structures.

This script takes specified chemical systems and target energy_above_hull values
to generate crystal structures using a pre-trained MatterGen model.
Outputs are saved to organized directories.
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
import subprocess
from datetime import datetime
import pandas as pd
import zipfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('generate_structures')

# Global variable for arguments to be accessible in helper functions if needed
args_gen = None
config_gen = None # For generation specific configs

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate crystal structures using MatterGen.')
    parser.add_argument('--chem_sys', type=str, required=True,
                        help='Target chemical system, e.g., "Li-P-S".')
    parser.add_argument('--e_hull', type=float, required=True,
                        help='Target energy_above_hull (eV/atom).')
    parser.add_argument('--num_samples', type=int, default=None, # Default will come from config
                        help='Number of samples to generate. Overrides config.')
    parser.add_argument('--batch_size', type=int, default=None, # Default will come from config
                        help='Batch size for generation. Overrides config.')
    parser.add_argument('--guidance_scale', type=float, default=None, # Default will come from config
                        help='Conditional guidance strength for MatterGen. Overrides config.')
    parser.add_argument('--pretrained_model', type=str, default=None, # Default will come from config
                        help='Name of the MatterGen pretrained model. Overrides config.')
    parser.add_argument('--mattergen_repo_path', type=str, default=None,
                        help='Path to the local MatterGen repository. Overrides config.')
    parser.add_argument('--output_dir_base', type=str, default=None,
                        help='Base output directory for generated structures. Overrides config (data/generated).')
    parser.add_argument('--record_trajectories', action='store_true', default=None, # Default will come from config
                        help='Whether to record trajectories during generation. Overrides config.')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Path to a JSON config file for generation parameters (e.g., ../configs/generation_configs.json).')
    return parser.parse_args()

def load_config_gen(config_file_path_str): # 参数改为字符串，函数内部转 Path
    """Loads generation configuration from a JSON file, ensuring UTF-8 encoding."""
    if not config_file_path_str: # 如果路径是 None 或空字符串
        logger.warning("No config file path provided to load_config_gen.")
        return {}
    
    config_file_path = Path(config_file_path_str) # 将字符串路径转换为 Path 对象
    if config_file_path.exists():
        try:
            # 明确指定使用 utf-8 编码打开文件
            with open(config_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {config_file_path}: {e}")
            return {} # 返回空字典或抛出异常，取决于您希望如何处理错误
        except Exception as e:
            logger.error(f"Error reading config file {config_file_path}: {e}")
            return {}
    else:
        logger.warning(f"Config file not found at: {config_file_path}")
        return {}

def get_effective_config_gen(cmd_args, file_config_data, target_chem_sys, target_e_hull):
    """
    Merges command-line arguments and file configurations for a specific generation task.
    File config can have a 'default' section and a 'chemical_systems' list with specific overrides.
    """
    # Start with global defaults from the config file
    config = file_config_data.get("default", {}).copy()

    # Find specific config for the current chemical_system if it exists
    system_specific_config = {}
    for sys_conf in file_config_data.get("chemical_systems", []):
        if sys_conf.get("name") == target_chem_sys:
            system_specific_config = sys_conf
            break
    
    # Override defaults with system-specific settings from config
    config.update({k: v for k, v in system_specific_config.items() if k not in ['name', 'e_hull_targets', 'description']})

    # Command-line arguments take highest precedence
    if cmd_args.num_samples is not None:
        config["num_samples"] = cmd_args.num_samples
    if cmd_args.batch_size is not None:
        config["batch_size"] = cmd_args.batch_size
    if cmd_args.guidance_scale is not None:
        config["guidance_scale"] = cmd_args.guidance_scale
    if cmd_args.pretrained_model is not None:
        config["pretrained_model"] = cmd_args.pretrained_model
    if cmd_args.mattergen_repo_path is not None:
        config["mattergen_repo_path"] = cmd_args.mattergen_repo_path
    if cmd_args.record_trajectories is not None: # Handles action='store_true'
        config["record_trajectories"] = cmd_args.record_trajectories
    
    # Set defaults if still not present after all overrides
    config.setdefault("num_samples", 16) # Smaller default for single run
    config.setdefault("batch_size", 4)
    config.setdefault("guidance_scale", 1.0)
    config.setdefault("pretrained_model", "chemical_system_energy_above_hull")
    config.setdefault("mattergen_repo_path", "../mattergen") # Relative to LithiumVision project root
    config.setdefault("record_trajectories", False)
    
    # Add the specific targets for this run
    config["chem_sys"] = target_chem_sys
    config["e_hull"] = target_e_hull

    return config

def setup_output_dir_gen(base_output_dir_str, chem_sys, e_hull):
    """Creates and returns the output directory for a specific generation run."""
    project_root = Path(__file__).resolve().parent.parent
    base_dir = Path(base_output_dir_str) if base_output_dir_str else project_root / "data" / "generated"
    
    # Sanitize chem_sys and e_hull for directory naming
    dir_name_suffix = f"{chem_sys.replace('-', '')}_ehull_{str(e_hull).replace('.', 'p')}"
    output_dir = base_dir / dir_name_suffix
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output for {chem_sys} @ e_hull={e_hull} will be in: {output_dir}")
    return output_dir

def log_generation_activity(log_data, status="Running", generated_cif_count=None):
    """Logs generation activity to a CSV file."""
    project_root = Path(__file__).resolve().parent.parent
    log_file_path = project_root / "results" / "tables" / "generation_campaign_log.csv"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["activity_id", "timestamp_start", "timestamp_end", "duration_seconds",
                  "chemical_system", "target_e_hull", "num_samples_requested", 
                  "batch_size_used", "guidance_scale_used", "pretrained_model_used",
                  "output_path", "status", "generated_cif_zip_count", "notes"]
    
    write_header = not log_file_path.exists()
    
    current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 确保无论什么状态都有timestamp_start
    if "timestamp_start" not in log_data:
        log_data["timestamp_start"] = current_timestamp
    
    if status == "Started":
        log_data["timestamp_start"] = current_timestamp
        log_data["status"] = status
    else: # Completed or Failed
        log_data["timestamp_end"] = current_timestamp
        log_data["status"] = status
        
        # 计算持续时间
        try:
            start_dt = datetime.strptime(log_data["timestamp_start"], '%Y-%m-%d %H:%M:%S')
            end_dt = datetime.strptime(log_data["timestamp_end"], '%Y-%m-%d %H:%M:%S')
            log_data["duration_seconds"] = (end_dt - start_dt).total_seconds()
        except (KeyError, ValueError) as e:
            # 如果无法计算持续时间，设为0
            logger.warning(f"Could not calculate duration: {e}")
            log_data["duration_seconds"] = 0
            
        # 设置generated_cif_zip_count
        if generated_cif_count is not None:
            log_data["generated_cif_zip_count"] = generated_cif_count
        else:
            log_data["generated_cif_zip_count"] = 0

    try:
        # 确保所有必需字段都有值，即使是空值
        for field in fieldnames:
            if field not in log_data:
                log_data[field] = ""
                
        with open(log_file_path, 'a', newline='') as csvfile:
            # Using pandas to easily append and handle headers
            # This is a bit heavy for simple logging but robust
            new_log_entry_df = pd.DataFrame([log_data])
            if write_header:
                 new_log_entry_df.to_csv(csvfile, header=True, index=False, columns=fieldnames)
            else:
                 new_log_entry_df.to_csv(csvfile, header=False, index=False, columns=fieldnames)

    except Exception as e:
        logger.error(f"Failed to write to generation log: {e}")


def run_mattergen_script(eff_config, output_path_for_run):
    """Constructs and runs the MatterGen generation command."""
    
    mattergen_repo_abs_path = Path(eff_config["mattergen_repo_path"]).resolve()
    mattergen_script = mattergen_repo_abs_path / "scripts" / "generate.py"

    if not mattergen_script.exists():
        logger.error(f"MatterGen generate script not found at: {mattergen_script}")
        logger.error("Please ensure 'mattergen_repo_path' in config or via --mattergen_repo_path is correct.")
        return False, 0

    properties_to_condition_on = {
        'chemical_system': eff_config["chem_sys"],
        'energy_above_hull': eff_config["e_hull"]
    }
    properties_str = str(properties_to_condition_on).replace(' ', '')
    
    cmd = [
        sys.executable,
        str(mattergen_script),
        f"--output_path={str(output_path_for_run)}",
        f"--pretrained_name={eff_config['pretrained_model']}",
        f"--batch_size={eff_config['batch_size']}",
        f"--num_batches={eff_config['num_samples'] // eff_config['batch_size'] + (1 if eff_config['num_samples'] % eff_config['batch_size'] > 0 else 0)}",
        f"--properties_to_condition_on={properties_str}",
        f"--diffusion_guidance_factor={eff_config['guidance_scale']}",
        f"--record_trajectories={'true' if eff_config['record_trajectories'] else 'false'}"
    ]
    
    logger.info(f"Executing MatterGen command: {' '.join(cmd)}")
    
    try:
        env = os.environ.copy()
        env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        logger.info("Set KMP_DUPLICATE_LIB_OK=TRUE to resolve OpenMP runtime conflicts")
        
        process = subprocess.run(cmd, cwd=mattergen_repo_abs_path, check=True, capture_output=True, text=True, env=env)
        logger.info("MatterGen generation completed successfully.")
        logger.debug(f"MatterGen STDOUT:\n{process.stdout}")
        
        cif_zip_path = output_path_for_run / "generated_crystals_cif.zip"
        count = 0
        if cif_zip_path.exists():
            try:
                with zipfile.ZipFile(cif_zip_path, 'r') as zf:
                    count = len([name for name in zf.namelist() if name.endswith('.cif')])
                logger.info(f"Found {count} CIF files in {cif_zip_path}.")
            except Exception as e_zip:
                logger.warning(f"Could not count CIFs in zip {cif_zip_path}: {e_zip}")
        else:
            logger.warning(f"Output CIF zip not found at {cif_zip_path}. Expected by MatterGen.")
            
        return True, count
    except subprocess.CalledProcessError as e:
        logger.error(f"MatterGen generation failed with exit code {e.returncode}.")
        logger.error(f"STDERR:\n{e.stderr}")
        logger.error(f"STDOUT (if any):\n{e.stdout}")
        return False, 0
    except FileNotFoundError:
        logger.error(f"MatterGen script or Python interpreter not found. Check paths. Attempted: {cmd[0]}, {cmd[1]}")
        return False, 0

def main():
    global args_gen, config_gen # Make global for easy access
    args_gen = parse_arguments()

    # Load file config if provided
    file_config_data = {}
    if args_gen.config_file:
        file_config_data = load_config_gen(args_gen.config_file)
    elif not args_gen.config_file : # If no config file try default path
        default_config_path = Path(__file__).resolve().parent.parent / "configs" / "generation_configs.json"
        if default_config_path.exists():
            logger.info(f"No config file specified, attempting to load default: {default_config_path}")
            file_config_data = load_config_gen(default_config_path)
        else:
            logger.warning(f"Default config file not found at {default_config_path}. Using command-line args and script defaults.")

    # Determine effective configuration for this specific run
    # Command line args like --chem_sys and --e_hull define the *specific task*
    effective_run_config = get_effective_config_gen(args_gen, file_config_data, args_gen.chem_sys, args_gen.e_hull)
    logger.info(f"Effective generation configuration for this run: {effective_run_config}")

    output_dir_for_run = setup_output_dir_gen(
        args_gen.output_dir_base or file_config_data.get("batch_generation", {}).get("output_base_dir"), # Base dir from CLI or config
        effective_run_config["chem_sys"],
        effective_run_config["e_hull"]
    )

    activity_id_str = f"gen_{effective_run_config['chem_sys'].replace('-', '')}_{str(effective_run_config['e_hull']).replace('.', 'p')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # 包含当前时间戳作为起始时间
    current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry_base = {
        "activity_id": activity_id_str,
        "timestamp_start": current_timestamp,  # 添加初始时间戳
        "chemical_system": effective_run_config["chem_sys"],
        "target_e_hull": effective_run_config["e_hull"],
        "num_samples_requested": effective_run_config["num_samples"],
        "batch_size_used": effective_run_config["batch_size"],
        "guidance_scale_used": effective_run_config["guidance_scale"],
        "pretrained_model_used": effective_run_config["pretrained_model"],
        "output_path": str(output_dir_for_run),
        "notes": ""
    }

    log_generation_activity(log_entry_base.copy(), status="Started")
    
    generation_successful, num_cifs = run_mattergen_script(effective_run_config, output_dir_for_run)
    
    if generation_successful:
        log_generation_activity(log_entry_base.copy(), status="Completed", generated_cif_count=num_cifs)
        # Save effective config used for this run into its output directory
        with open(output_dir_for_run / "generation_run_params.json", 'w') as f_params:
            json.dump(effective_run_config, f_params, indent=2)
        logger.info(f"Generation for {effective_run_config['chem_sys']} @ e_hull={effective_run_config['e_hull']} finished. {num_cifs} structures in output zip.")
        return 0
    else:
        log_generation_activity(log_entry_base.copy(), status="Failed", generated_cif_count=0)
        logger.error(f"Generation for {effective_run_config['chem_sys']} @ e_hull={effective_run_config['e_hull']} failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())