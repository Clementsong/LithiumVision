#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualizes crystal structures of candidate materials.

This script takes a CSV file of candidate materials (typically the output
from `filter_candidates.py`), loads their CIF files, and generates
visualization images (e.g., PNGs) for the top N candidates.
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
import traceback
from tqdm import tqdm
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('visualize_structures')

# Check for required libraries
try:
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor # For potential ASE interoperability if needed
    # For basic plotting with Matplotlib backend:
    from pymatgen.vis.plotters import StructurePlotter 
    # For more advanced VTK-based plotting (optional, requires VTK):
    # from pymatgen.vis.structure_vtk import StructureVis 
    logger.info("Pymatgen visualization libraries loaded successfully.")
except ImportError as e:
    logger.error(f"Failed to import pymatgen visualization libraries: {e}")
    logger.error("Please ensure pymatgen is installed correctly. For advanced 3D views, VTK might be needed.")
    sys.exit(1)

# Global args and config
args_vis = None
config_vis = None # From analysis_configs.json -> visualization section

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Visualize crystal structures of candidate materials.')
    parser.add_argument('--candidates_csv', type=str, required=True,
                        help='Path to the CSV file containing candidate materials (e.g., top_candidates.csv).')
    # CIF files are expected to be found based on 'cif_file' and 'source_experiment' columns in the CSV,
    # relative to a base 'data/generated/' directory.
    parser.add_argument('--generated_data_dir', type=str, default=None, # Default: project_root/data/generated
                        help='Base directory where generated structures (original CIFs in subdirs) are stored.')
    parser.add_argument('--output_dir', type=str, default=None, # Default: project_root/results/figures/structures
                        help='Directory to save the visualization images.')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Path to a JSON config file for visualization parameters (e.g., ../configs/analysis_configs.json).')
    
    # Overrides for config values
    parser.add_argument('--top_n', type=int, default=None,
                        help='Number of top candidates to visualize. Overrides config.')
    parser.add_argument('--image_format', type=str, default=None, choices=['png', 'svg', 'pdf', 'jpg'],
                        help='Output image format. Overrides config.')
    parser.add_argument('--dpi', type=int, default=None,
                        help='DPI for raster image formats (png, jpg). Overrides config.')
    return parser.parse_args()

def load_config_vis(config_file_path):
    """Loads visualization configuration from a JSON file (expects analysis_configs.json structure)."""
    if config_file_path and Path(config_file_path).exists():
        with open(config_file_path, 'r') as f:
            full_config = json.load(f)
            # Extract the visualization relevant part
            return full_config.get("visualization", {})
    return {}

def get_effective_config_vis(cmd_args, file_config_data):
    """Merges command-line arguments and file configurations for visualization."""
    config = file_config_data # Start with visualization section from config file

    # Command-line arguments take precedence
    if cmd_args.top_n is not None:
        config["top_candidates_to_visualize"] = cmd_args.top_n
    if cmd_args.image_format is not None:
        config["image_format"] = cmd_args.image_format
    if cmd_args.dpi is not None:
        config["image_dpi"] = cmd_args.dpi
    
    # Set defaults if not present
    config.setdefault("top_candidates_to_visualize", 10)
    config.setdefault("image_format", "png")
    config.setdefault("image_dpi", 300)
    
    return config

def setup_paths_vis():
    """Sets up input CSV path, generated data path, and output image directory."""
    global args_vis # Use global args
    project_root = Path(__file__).resolve().parent.parent

    candidates_path = Path(args_vis.candidates_csv)
    if not candidates_path.exists():
        logger.error(f"Candidates CSV file not found: {candidates_path}")
        sys.exit(1)

    gen_data_dir = Path(args_vis.generated_data_dir) if args_vis.generated_data_dir else project_root / "data" / "generated"
    if not gen_data_dir.exists():
        logger.warning(f"Base directory for generated CIFs not found: {gen_data_dir}. CIF lookup might fail.")
        # Not exiting, as individual CIF paths might be absolute or resolvable differently.

    output_img_dir = Path(args_vis.output_dir) if args_vis.output_dir else project_root / "results" / "figures" / "structures"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading candidates from: {candidates_path}")
    logger.info(f"Looking for original CIFs under: {gen_data_dir}")
    logger.info(f"Saving visualization images to: {output_img_dir}")
    
    return candidates_path, gen_data_dir, output_img_dir

def load_candidates_df(candidates_csv_path, num_to_load):
    """Loads the top N candidates from the CSV file."""
    try:
        df = pd.read_csv(candidates_csv_path)
        logger.info(f"Loaded {df.shape[0]} total candidates from {candidates_csv_path}.")
        top_df = df.head(num_to_load)
        logger.info(f"Selected top {top_df.shape[0]} candidates for visualization.")
        return top_df
    except FileNotFoundError:
        logger.error(f"Candidates CSV file not found at: {candidates_csv_path}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logger.error(f"Candidates CSV file is empty: {candidates_csv_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading candidates CSV {candidates_csv_path}: {e}")
        sys.exit(1)


def get_cif_path_for_candidate(candidate_row, base_generated_dir):
    """
    Determines the path to the original CIF file for a candidate.
    Expects 'cif_file' (e.g., "0.cif") and 'source_experiment' (e.g., "LiPS_ehull_0.05") columns.
    The CIFs are assumed to be in an 'extracted_cifs' subdirectory after unzipping.
    """
    try:
        cif_filename = candidate_row['cif_file']
        source_exp_dirname = candidate_row['source_experiment'] # e.g., "LiPS_ehull_0.05"
        
        # Path: <base_generated_dir>/<source_exp_dirname>/extracted_cifs/<cif_filename>
        # This structure is created by analyze_structures.py when it extracts from the zip.
        # If analyze_structures.py changes its extraction logic, this needs to adapt.
        # The original zip is in <base_generated_dir>/<source_exp_dirname>/generated_crystals_cif.zip
        
        # Try the 'extracted_cifs' path first, as created by analyze_structures.py
        # This assumes analyze_structures.py has already run and extracted CIFs from their zips.
        # The `input_dir` for `analyze_structures.py` is typically `data/generated/chem_sys_ehull_X`.
        # `analyze_structures.py` then creates an `extracted_cifs` folder inside this `input_dir`.
        # So `source_exp_dirname` should align with the `chem_sys_ehull_X` folder name.

        cif_file_path = base_generated_dir / source_exp_dirname / "extracted_cifs" / cif_filename
        if cif_file_path.exists():
            return cif_file_path
        else:
            # Fallback: if 'extracted_cifs' isn't there, maybe the CIF is directly in source_exp_dirname
            # (less likely if following the established workflow, but a fallback)
            cif_file_path_alt = base_generated_dir / source_exp_dirname / cif_filename
            if cif_file_path_alt.exists():
                logger.warning(f"Found CIF at {cif_file_path_alt}, not in 'extracted_cifs'. Ensure analysis script ran as expected.")
                return cif_file_path_alt
            
            logger.warning(f"CIF file {cif_filename} not found for experiment {source_exp_dirname} at expected path: {cif_file_path} (or alt path).")
            return None
            
    except KeyError as e:
        logger.error(f"Candidate row missing expected column for CIF path reconstruction: {e}. Row data: {candidate_row.to_dict()}")
        return None
    except Exception as e_path:
        logger.error(f"Error constructing CIF path for candidate: {e_path}. Row data: {candidate_row.to_dict()}")
        return None


def visualize_and_save_structure(pymatgen_struct, output_image_path, img_format, img_dpi, title=None):
    """Visualizes a Pymatgen Structure and saves it to a file."""
    try:
        plotter = StructurePlotter(sgt=None) # sgt=None uses default settings
        # Common view: standard orientation, show unit cell
        plotter.show_bonds = True # Or False, depending on preference
        # Add atoms with labels could be an option if StructurePlotter supports it directly or via matplotlib context
        
        fig = plotter.get_plot(pymatgen_struct, plot_cell=True) # plot_cell=True shows unit cell
        
        if title:
            fig.suptitle(title, fontsize=10) # Use suptitle for figure-level title
        
        fig.savefig(output_image_path, format=img_format, dpi=img_dpi, bbox_inches='tight')
        plt.close(fig) # Close the figure to free memory
        logger.info(f"Saved visualization to: {output_image_path}")
        return True
    except Exception as e:
        logger.error(f"Error visualizing/saving structure for {output_image_path.name}: {e}")
        logger.debug(traceback.format_exc())
        return False

def create_summary_montage(image_paths, output_montage_path, img_format, img_dpi, title="Top Candidate Structures"):
    """Creates a montage of several structure images."""
    if not image_paths:
        logger.info("No images provided for montage.")
        return

    num_images = len(image_paths)
    # Simple layout: try to make it somewhat square-ish
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4)) # Adjust size as needed
    axes = np.array(axes).flatten() # Ensure axes is always a flat array

    for i, img_path_str in enumerate(image_paths):
        if i < len(axes):
            try:
                img_path = Path(img_path_str)
                img = plt.imread(img_path)
                axes[i].imshow(img)
                axes[i].set_title(img_path.stem, fontsize=8) # Use filename stem as title
                axes[i].axis('off')
            except FileNotFoundError:
                logger.warning(f"Image not found for montage: {img_path_str}")
                axes[i].text(0.5, 0.5, 'Image not found', ha='center', va='center')
                axes[i].axis('off')
            except Exception as e_img:
                 logger.warning(f"Could not load image {img_path_str} for montage: {e_img}")
                 axes[i].text(0.5, 0.5, 'Error loading', ha='center', va='center')
                 axes[i].axis('off')


    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to make space for suptitle
    fig.savefig(output_montage_path, format=img_format, dpi=img_dpi)
    plt.close(fig)
    logger.info(f"Saved montage of {num_images} images to: {output_montage_path}")


def main_visualize():
    global args_vis, config_vis
    args_vis = parse_arguments()

    file_config_data = {}
    if args_vis.config_file:
        file_config_data = load_config_vis(args_vis.config_file)
    elif not args_vis.config_file : # If no config file try default path
        default_config_path = Path(__file__).resolve().parent.parent / "configs" / "analysis_configs.json"
        if default_config_path.exists():
            logger.info(f"No config file specified, attempting to load default: {default_config_path}")
            file_config_data = load_config_vis(default_config_path)
        else:
            logger.warning(f"Default config file not found at {default_config_path}. Using command-line args and script defaults.")
    
    config_vis = get_effective_config_vis(args_vis, file_config_data)
    logger.info(f"Effective visualization configuration: {config_vis}")

    candidates_csv_p, generated_data_p, output_images_p = setup_paths_vis()
    
    df_to_visualize = load_candidates_df(candidates_csv_p, config_vis["top_candidates_to_visualize"])

    if df_to_visualize.empty:
        logger.info("No candidates to visualize.")
        return 0

    successful_visualizations = []
    failed_visualizations = 0
    
    for index, row in tqdm(df_to_visualize.iterrows(), total=df_to_visualize.shape[0], desc="Visualizing structures"):
        cif_file_abs_path = get_cif_path_for_candidate(row, generated_data_p)
        
        if cif_file_abs_path and cif_file_abs_path.exists():
            try:
                struct = Structure.from_file(cif_file_abs_path)
                
                # Create a meaningful image filename
                formula_sanitized = row.get('formula_generated', f'id_{index}').replace('/', '_')
                cif_stem = Path(row.get('cif_file', f'unk_{index}')).stem
                img_filename = f"{formula_sanitized}_cif_{cif_stem}.{config_vis['image_format']}"
                img_output_path = output_images_p / img_filename
                
                # Create a title for the plot
                plot_title_parts = [f"{formula_sanitized} (CIF: {cif_stem})"]
                if 'best_mp_e_above_hull' in row and pd.notna(row['best_mp_e_above_hull']):
                    plot_title_parts.append(f"Eнull: {row['best_mp_e_above_hull']:.3f} eV/atom")
                if 'is_novel_by_formula' in row:
                    plot_title_parts.append(f"Novel: {row['is_novel_by_formula']}")
                plot_title = "\n".join(plot_title_parts)

                if visualize_and_save_structure(struct, img_output_path, config_vis['image_format'], config_vis['image_dpi'], title=plot_title):
                    successful_visualizations.append(str(img_output_path))
                else:
                    failed_visualizations += 1
            except Exception as e_struct:
                logger.error(f"Could not process or visualize CIF {cif_file_abs_path}: {e_struct}")
                failed_visualizations += 1
        else:
            logger.warning(f"CIF file path not found or invalid for candidate index {index}, original cif_file: {row.get('cif_file', 'N/A')}")
            failed_visualizations += 1

    logger.info(f"Visualization process finished. Successful: {len(successful_visualizations)}, Failed: {failed_visualizations}")

    # Create a summary montage if there are successful visualizations
    if successful_visualizations:
        montage_filename = f"top_candidates_montage.{config_vis['image_format']}"
        montage_output_path = output_images_p.parent / montage_filename # Save in figures/, not figures/structures/
        create_summary_montage(successful_visualizations, montage_output_path, config_vis['image_format'], config_vis['image_dpi'])
    
    # Save a summary of the visualization task
    visualization_run_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "candidates_csv_source": str(candidates_csv_p),
        "num_candidates_processed": df_to_visualize.shape[0],
        "num_successful_visualizations": len(successful_visualizations),
        "num_failed_visualizations": failed_visualizations,
        "image_format_used": config_vis['image_format'],
        "image_dpi_used": config_vis['image_dpi'],
        "output_directory": str(output_images_p),
        "summary_montage_file": str(montage_output_path) if successful_visualizations else None,
        "visualized_image_paths": successful_visualizations
    }
    vis_info_json_path = output_images_p / "visualization_info.json"
    try:
        with open(vis_info_json_path, 'w') as f_info:
            json.dump(visualization_run_info, f_info, indent=2)
        logger.info(f"Visualization run summary saved to: {vis_info_json_path}")
    except Exception as e_json:
        logger.error(f"Could not save visualization_info.json: {e_json}")

    return 0 if failed_visualizations == 0 else 1


if __name__ == "__main__":
    sys.exit(main_visualize())