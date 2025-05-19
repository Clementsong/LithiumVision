#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Environment setup script for LithiumVision.

Checks for necessary prerequisites like Python version, Git LFS,
MatterGen installation, and API keys. It can also guide MatterGen setup
and create the project's directory structure.
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
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
logger = logging.getLogger('setup_environment')

# Project root assuming this script is in LithiumVision/scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def check_python_version(min_major=3, min_minor=10):
    """Checks if the current Python version meets the minimum requirement."""
    current_version = sys.version_info
    logger.info(f"Current Python version: {current_version.major}.{current_version.minor}.{current_version.micro}")
    if current_version.major < min_major or \
       (current_version.major == min_major and current_version.minor < min_minor):
        logger.error(f"Python version {min_major}.{min_minor} or higher is required. You have {current_version.major}.{current_version.minor}.")
        return False
    logger.info(f"Python version check passed (>= {min_major}.{min_minor}).")
    return True

def check_command_exists(command_array, help_message=""):
    """Checks if a command exists and is executable."""
    try:
        process = subprocess.run(command_array, check=True, capture_output=True, text=True)
        logger.info(f"Command '{command_array[0]}' found: {process.stdout.strip().splitlines()[0] if process.stdout else 'OK'}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"Command '{command_array[0]}' not found or not executable. {help_message} Error: {e}")
        return False

def check_git_lfs():
    """Checks for Git LFS."""
    return check_command_exists(['git', 'lfs', '--version'], "Please install Git LFS: https://git-lfs.github.com/")

def check_mattergen_installation(mattergen_dir_path_str):
    """Checks if MatterGen seems to be installed and its main script is present."""
    mattergen_dir = Path(mattergen_dir_path_str).resolve()
    if not mattergen_dir.is_dir():
        logger.warning(f"MatterGen directory not found at: {mattergen_dir}")
        return False
    
    # Check for a key file, e.g., the run script for MatterGen
    # Adjust this path if MatterGen's structure changes
    mattergen_run_script = mattergen_dir / "scripts" / "run.py" # Based on common MatterGen structure
    if not mattergen_run_script.exists():
        logger.warning(f"MatterGen run script not found at {mattergen_run_script}. Installation might be incomplete or path incorrect.")
        return False
        
    # Try importing mattergen package (if it's installed in the environment)
    # Inside the check_mattergen_installation function in scripts/setup_environment.py

    try:
        # Temporarily add mattergen_dir to path to check if it's importable from there
        original_sys_path = list(sys.path)
        # Corrected line: use mattergen_dir.parent
        sys.path.insert(0, str(mattergen_dir.parent)) # Add parent of mattergen_dir
        sys.path.insert(0, str(mattergen_dir))       # Add mattergen_dir itself

        import mattergen # Attempt to import
        logger.info(f"Successfully imported 'mattergen' package. Version: {getattr(mattergen, '__version__', 'unknown')}")
        # After successful import, it's good practice to check if it's the one from the expected path
        # This is a bit more involved, but for now, importing is a good sign.
        is_setup_correctly = True
    except ImportError:
        logger.warning("Could not import 'mattergen' package. It might not be installed correctly in the Python environment or path is wrong.")
        logger.warning(f"Attempted to check in: {mattergen_dir} (and its parent for package structure)")
        logger.warning("Ensure you have run 'python setup.py develop' or 'pip install -e .' inside the MatterGen directory if it's a package, and your environment is active.")
        is_setup_correctly = False
    finally:
        sys.path = original_sys_path # Restore original sys.path

    if not is_setup_correctly:
        return False

    # Check for a key file, e.g., the run script for MatterGen
    mattergen_run_script = mattergen_dir / "scripts" / "run.py"
    if not mattergen_run_script.exists():
        logger.warning(f"MatterGen run script not found at {mattergen_run_script}. Installation might be incomplete or path incorrect.")
        return False

    logger.info(f"MatterGen installation at {mattergen_dir} seems plausible (import successful and run script found).")
    return True


def prompt_mattergen_setup(default_mattergen_path_str):
    """Prompts user to set up MatterGen if not found."""
    mattergen_path = Path(default_mattergen_path_str).resolve()
    
    if check_mattergen_installation(str(mattergen_path)):
        logger.info(f"MatterGen appears to be set up at {mattergen_path}.")
        return True

    logger.warning(f"MatterGen not found or not set up at the default path: {mattergen_path}")
    user_choice = input(f"MatterGen repository path is configured as '{mattergen_path}'.\n"
                        "If this is incorrect or MatterGen is not cloned/installed there, please update it.\n"
                        "Do you want to:\n"
                        "1. Specify a different path to an existing MatterGen clone?\n"
                        "2. Attempt to clone MatterGen to the default path (requires Git)?\n"
                        "3. Skip MatterGen setup (workflow might fail)?\n"
                        "Enter choice (1, 2, or 3): ").strip()

    if user_choice == '1':
        new_path_str = input("Enter the correct absolute path to your MatterGen repository: ").strip()
        new_path = Path(new_path_str).resolve()
        if check_mattergen_installation(str(new_path)):
            logger.info(f"MatterGen found at new path: {new_path}. Update your generation_configs.json if this is permanent.")
            # TODO: Optionally update config here, or just advise user.
            return True
        else:
            logger.error(f"MatterGen still not found or setup correctly at {new_path}.")
            return False
            
    elif user_choice == '2':
        if not mattergen_path.parent.exists():
            mattergen_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Attempting to clone MatterGen into {mattergen_path}...")
        try:
            subprocess.run(['git', 'clone', 'https://github.com/microsoft/mattergen.git', str(mattergen_path)], check=True)
            logger.info("MatterGen cloned successfully. Now attempting to install Git LFS files and dependencies.")
            
            if not check_git_lfs(): return False # LFS is critical

            subprocess.run(['git', 'lfs', 'install'], cwd=mattergen_path, check=True)
            subprocess.run(['git', 'lfs', 'pull'], cwd=mattergen_path, check=True)
            logger.info("Git LFS files pulled.")
            
            logger.info("Attempting to install MatterGen (pip install -e .)...")
            # Ensure using the correct python interpreter for the pip install
            pip_install_cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
            subprocess.run(pip_install_cmd, cwd=mattergen_path, check=True)
            logger.info("MatterGen 'pip install -e .' executed.")
            
            return check_mattergen_installation(str(mattergen_path)) # Verify again
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone or setup MatterGen: {e}")
            return False
        except FileNotFoundError:
            logger.error("Git command not found. Please install Git.")
            return False
            
    elif user_choice == '3':
        logger.warning("Skipping MatterGen setup. Generation step will likely fail.")
        return False # Not successfully set up
    else:
        logger.warning("Invalid choice. Skipping MatterGen setup.")
        return False

def check_mp_api_key():
    """Checks if Materials Project API key is set as an environment variable."""
    api_key = os.environ.get('MP_API_KEY')
    if api_key:
        logger.info("MP_API_KEY environment variable found.")
        # Optionally, try a quick test query if mp-api is installed
        try:
            from pymatgen.ext.matproj import MPRester
            with MPRester(api_key) as mpr:
                # A lightweight query
                mpr.materials.summary.search(material_ids=["mp-149"], fields=["material_id"], chunk_size=1) 
            logger.info("MP_API_KEY seems valid (tested a small query).")
        except ImportError:
            logger.info("mp-api not installed, cannot test API key validity now.")
        except Exception as e:
            logger.warning(f"MP_API_KEY found, but a test query failed: {e}. Key might be invalid or MP API down.")
            logger.warning("Ensure your API key is correct and has permissions.")
        return True
    else:
        logger.warning("MP_API_KEY environment variable not set.")
        logger.warning("Please set it: export MP_API_KEY='your_key_here'")
        logger.warning("Alternatively, provide it in 'configs/analysis_configs.json' or via --mp_api_key_override in run_workflow.py.")
        return False

def create_project_directories():
    """Creates the necessary data and results directory structure if they don't exist."""
    dirs_to_create = [
        PROJECT_ROOT / "data" / "generated",
        PROJECT_ROOT / "data" / "analyzed",
        PROJECT_ROOT / "results" / "candidates",
        PROJECT_ROOT / "results" / "figures" / "structures",
        PROJECT_ROOT / "results" / "tables",
        PROJECT_ROOT / "notebooks" # Ensure notebooks dir also exists
    ]
    for dir_path in dirs_to_create:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {dir_path}")
        except OSError as e:
            logger.error(f"Could not create directory {dir_path}: {e}")


def main_setup():
    parser = argparse.ArgumentParser(description="LithiumVision Environment Setup Helper")
    parser.add_argument('--mattergen_default_path', type=str, 
                        default=str(PROJECT_ROOT.parent / "mattergen"), # Default: one level up from LithiumVision, then 'mattergen'
                        help="Default expected path for the MatterGen repository.")
    parser.add_argument('--skip_mattergen_setup_prompt', action='store_true',
                        help="Skip interactive prompt for MatterGen setup if not found at default path.")
    parser.add_argument('--force_create_dirs', action='store_true',
                        help="Force creation of project directories.")

    args = parser.parse_args()

    logger.info("--- Starting LithiumVision Environment Check ---")
    all_ok = True

    if not check_python_version():
        all_ok = False
    
    # Check for conda if environment.yml exists
    if (PROJECT_ROOT / "environment.yml").exists():
        if not check_command_exists(['conda', '--version'], "Conda is recommended for managing the environment (environment.yml found)."):
            logger.warning("Consider installing Conda for easier environment setup.")
            # Not necessarily a failure if user manages env otherwise, but good to note.

    if not check_git_lfs():
        # Git LFS is crucial for MatterGen models
        logger.error("Git LFS is required for MatterGen. Please install it.")
        all_ok = False
        
    if not args.skip_mattergen_setup_prompt:
        if not prompt_mattergen_setup(args.mattergen_default_path):
            all_ok = False # If user skips or setup fails
    else: # Just check the default path if skipping prompt
        if not check_mattergen_installation(args.mattergen_default_path):
            logger.warning(f"MatterGen check skipped at prompt, but not found/setup at {args.mattergen_default_path}")
            all_ok = False


    if not check_mp_api_key():
        # This is a strong warning, not necessarily a blocker for all parts of setup
        logger.warning("MP API key setup is recommended for full functionality.")


    logger.info("--- Checking/Creating Project Directory Structure ---")
    create_project_directories()


    # Final summary
    if all_ok:
        logger.info("--- Environment Setup Check Completed ---")
        logger.info("Basic checks passed. Ensure MatterGen is correctly installed and models downloaded within its directory.")
        logger.info("Ensure Python dependencies from environment.yml or requirements.txt are installed in your active environment.")
        logger.info("e.g., using Conda: `conda env create -f environment.yml` then `conda activate lithiumvision`")
        logger.info("e.g., using pip: `pip install -r requirements.txt`")

    else:
        logger.error("--- Environment Setup Check Completed With Issues ---")
        logger.error("One or more critical checks failed. Please review the logs above and address the issues.")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main_setup())