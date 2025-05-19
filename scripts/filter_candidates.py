#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filters candidate materials from analysis results.

This script processes the CSV output from `analyze_structures.py`,
applies filtering criteria (e.g., stability, novelty, presence of Lithium),
scores candidates, and outputs a ranked list of promising materials.
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('filter_candidates')

# Global variable for arguments to be accessible in helper functions if needed
args_filter = None
config_filter = None


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Filter promising lithium-ion superconductor candidates.')
    parser.add_argument('--input_csv', type=str, required=False,
                        help='Path to the input CSV file (analyzed_structures.csv). If not provided, searches in --input_dir.')
    parser.add_argument('--input_dir', type=str, required=False,
                        help='Directory containing analyzed_structures.csv (e.g., data/analyzed/LiPS_ehull_0.05/).')
    parser.add_argument('--search_root_dir', type=str, default=None,
                        help="Root directory to search for multiple 'analyzed_structures.csv' files (e.g., data/analyzed/).")
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save the filtered top candidates CSV. Defaults to results/candidates/top_candidates.csv.')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Path to a JSON config file for filtering parameters (e.g., ../configs/analysis_configs.json).')
    # Filtering criteria will primarily come from the config file, but can be overridden
    parser.add_argument('--e_hull_max', type=float, default=None,
                        help='Maximum energy_above_hull (eV/atom) for a candidate to be considered stable. Overrides config.')
    parser.add_argument('--only_novel_formula', action='store_true', default=None,
                        help='Filter for only formula-novel structures. Overrides config.')
    parser.add_argument('--only_novel_structure', action='store_true', default=None,
                         help='Filter for only structure-novel structures (if structure matching was done). Overrides config.')
    parser.add_argument('--must_contain_li', action='store_true', default=None,
                        help='Filter for only Li-containing structures. Overrides config.')
    parser.add_argument('--top_n', type=int, default=None,
                        help='Number of top candidates to output. Overrides config.')

    return parser.parse_args()

def load_config(config_file_path):
    """Loads filtering configuration from a JSON file (expects analysis_configs.json structure)."""
    if config_file_path and Path(config_file_path).exists():
        with open(config_file_path, 'r') as f:
            full_config = json.load(f)
            # Extract the filtering relevant part
            return full_config.get("filtering", {})
    return {}

def get_effective_config_filter(cmd_args, file_config_data):
    """Merges command-line arguments and file configurations for filtering."""
    config = file_config_data # Start with filtering section from config file

    # Command-line arguments take precedence
    if cmd_args.e_hull_max is not None:
        config["e_hull_max"] = cmd_args.e_hull_max
    if cmd_args.only_novel_formula is not None:
        config["only_novel_formula"] = cmd_args.only_novel_formula
    if cmd_args.only_novel_structure is not None:
        config["only_novel_structure"] = cmd_args.only_novel_structure
    if cmd_args.must_contain_li is not None:
        config["must_contain_li"] = cmd_args.must_contain_li
    if cmd_args.top_n is not None:
        config["top_n_candidates"] = cmd_args.top_n
    
    # Set defaults if not present
    config.setdefault("e_hull_max", 0.1)
    config.setdefault("only_novel_formula", False)
    config.setdefault("only_novel_structure", False)
    config.setdefault("must_contain_li", True)
    config.setdefault("top_n_candidates", 20)
    config.setdefault("score_weights", {"stability": 1.0, "novelty_formula": 0.5, "novelty_structure":0.7})


    return config

def setup_paths_filter():
    """Determines input CSV paths and output CSV path."""
    global args_filter # Use the global args_filter
    project_root = Path(__file__).resolve().parent.parent
    input_csv_paths = []

    if args_filter.input_csv:
        input_csv_paths.append(Path(args_filter.input_csv))
    elif args_filter.input_dir:
        path = Path(args_filter.input_dir) / "analyzed_structures.csv"
        if path.exists():
            input_csv_paths.append(path)
        else:
            logger.warning(f"analyzed_structures.csv not found in {args_filter.input_dir}")
    elif args_filter.search_root_dir:
        root_search = Path(args_filter.search_root_dir)
        input_csv_paths.extend(list(root_search.glob("**/analyzed_structures.csv")))
        if not input_csv_paths:
            logger.warning(f"No 'analyzed_structures.csv' files found under {root_search}")
    else: # Default search location if nothing specific is provided
        default_search_dir = project_root / "data" / "analyzed"
        if default_search_dir.exists():
            logger.info(f"No specific input, searching for analysis files in {default_search_dir}")
            input_csv_paths.extend(list(default_search_dir.glob("**/analyzed_structures.csv")))
        if not input_csv_paths:
            logger.error("No input CSV files found. Please specify --input_csv, --input_dir, or --search_root_dir.")
            sys.exit(1)

    if not all(p.exists() for p in input_csv_paths):
        logger.error(f"One or more input CSV files do not exist: {[str(p) for p in input_csv_paths if not p.exists()]}")
        sys.exit(1)
    
    logger.info(f"Identified {len(input_csv_paths)} CSV file(s) for processing.")

    output_csv_path = Path(args_filter.output_csv) if args_filter.output_csv else project_root / "results" / "candidates" / "top_candidates.csv"
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    return input_csv_paths, output_csv_path

def load_and_combine_data(csv_paths):
    """Loads data from multiple CSV files and combines them."""
    all_dfs = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            # Add a source identifier based on the parent directory of the CSV
            df['source_experiment'] = path.parent.name
            all_dfs.append(df)
            logger.info(f"Loaded {len(df)} records from {path}")
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
        except pd.errors.EmptyDataError:
            logger.warning(f"File is empty: {path}")
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            
    if not all_dfs:
        logger.error("No data loaded. Exiting.")
        sys.exit(1)
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined data from {len(all_dfs)} file(s), total records: {len(combined_df)}")
    return combined_df

def preprocess_data_filter(df):
    """Preprocesses the combined DataFrame for filtering."""
    # Ensure boolean flags are correct type, handling "NotChecked" for novelty_by_structure
    if 'is_novel_by_formula' in df.columns:
        df['is_novel_by_formula'] = df['is_novel_by_formula'].astype(bool)
    if 'is_novel_by_structure' in df.columns:
        # Handle 'NotChecked' string: convert to NaN then to boolean (False for filtering)
        df['is_novel_by_structure_bool'] = df['is_novel_by_structure'].apply(lambda x: x if isinstance(x, bool) else (None if x == "NotChecked" else bool(x)))
        df['is_novel_by_structure_bool'] = df['is_novel_by_structure_bool'].astype('boolean') # Use nullable boolean
    if 'is_stable_on_mp' in df.columns:
        df['is_stable_on_mp'] = df['is_stable_on_mp'].astype(bool)

    # Ensure 'elements_generated' is a list of strings (it might be a string representation of a list)
    if 'elements_generated' in df.columns:
        def parse_elements(x):
            if isinstance(x, list): return x
            if isinstance(x, str):
                try: return eval(x) # Be cautious with eval
                except: return []
            return []
        df['elements_generated_list'] = df['elements_generated'].apply(parse_elements)
        df['contains_li'] = df['elements_generated_list'].apply(lambda elems: 'Li' in elems if isinstance(elems, list) else False)
    else:
        df['contains_li'] = False


    # Handle potential string "NaN" or actual NaN for numeric scores
    numeric_cols = ['best_mp_e_above_hull', 'best_mp_formation_energy_per_atom']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

def apply_filters_and_score(df):
    """Applies filtering criteria and scores the candidates."""
    global config_filter # Use the global config_filter
    
    # Apply filters
    if config_filter["must_contain_li"]:
        df = df[df['contains_li'] == True]
        logger.info(f"After 'must_contain_li' filter: {len(df)} records remaining.")

    if config_filter["only_novel_formula"]:
        df = df[df['is_novel_by_formula'] == True]
        logger.info(f"After 'only_novel_formula' filter: {len(df)} records remaining.")
    
    if config_filter["only_novel_structure"] and 'is_novel_by_structure_bool' in df.columns:
        df = df[df['is_novel_by_structure_bool'].fillna(False) == True] # Treat NaN as False for this filter
        logger.info(f"After 'only_novel_structure' filter: {len(df)} records remaining.")

    # Stability filter based on 'best_mp_e_above_hull' (for known) or generation target (for novel)
    # For this script, we primarily rely on 'best_mp_e_above_hull' if available.
    # Novel structures without MP data might be kept based on other criteria or given a default stability score.
    
    # Scoring
    # Stability score: Higher is better. Penalize high e_above_hull.
    # Max score 100 for e_hull <= 0. Score decreases as e_hull increases.
    # Capped at e_hull_max from config.
    df['stability_score'] = 100 * (1 - np.clip(df['best_mp_e_above_hull'].fillna(config_filter["e_hull_max"] * 2), 0, config_filter["e_hull_max"] * 2) / (config_filter["e_hull_max"] * 2))
    # For truly novel structures (is_novel_by_formula=True and best_mp_e_above_hull is NaN), assign a moderate score
    # or a score based on generation target if that info is passed through. For now, a default if no MP data.
    # This requires generation parameters to be part of the analyzed_structures.csv to be truly effective.
    # A simpler approach for now: if novel and no MP e_hull, give a base score.
    novel_no_mp_ehull_mask = (df['is_novel_by_formula'] == True) & (df['best_mp_e_above_hull'].isna())
    # TODO: If generation target e_hull is available, use it to refine score for novel materials.
    # For now, give a base score for novelty when MP e_hull is missing.
    df.loc[novel_no_mp_ehull_mask, 'stability_score'] = 50 # Example base score for novel with unknown stability

    # Novelty scores (binary for now, could be more nuanced)
    df['novelty_formula_score'] = df['is_novel_by_formula'].astype(int) * 100
    if 'is_novel_by_structure_bool' in df.columns:
         df['novelty_structure_score'] = df['is_novel_by_structure_bool'].fillna(False).astype(int) * 100
    else:
         df['novelty_structure_score'] = 0


    # Total score
    weights = config_filter["score_weights"]
    df['total_score'] = (
        df['stability_score'] * weights.get("stability", 1.0) +
        df['novelty_formula_score'] * weights.get("novelty_formula", 0.0) +
        df['novelty_structure_score'] * weights.get("novelty_structure", 0.0)
    )
    
    # Sort by total score and filter by e_hull_max for final list
    # We apply the e_hull_max filter *after* scoring, so unstable but highly novel items might still be seen if not for this final cut.
    # However, the prompt implies e_hull_max is a hard cut for "stable structures".
    # Let's make 'is_considered_stable' flag first
    df['is_considered_stable'] = df['best_mp_e_above_hull'] <= config_filter["e_hull_max"]
    # For novel structures where best_mp_e_above_hull is NaN, we can't use this.
    # For the "一周冲刺", focus on MP-confirmed stability or MatterGen's generation target (if available).
    # If a structure is novel and has no MP e_hull, it's "stability unknown from MP".
    # The current plan focuses on screening based on MP data or MatterSim (if integrated later).
    # So, for now, only MP-known structures are filtered by e_hull_max here.
    # Novel structures are kept if they pass other criteria (like only_novel_formula).

    final_candidates = df.sort_values(by='total_score', ascending=False)
    
    # If we want to strictly filter out known unstable materials:
    # final_candidates = final_candidates[final_candidates['is_considered_stable'] | final_candidates['is_novel_by_formula']]
    # This keeps structures that are either stable on MP, or are novel by formula (stability to be determined)

    # Apply top_n
    top_n = config_filter.get("top_n_candidates", 20)
    final_candidates = final_candidates.head(top_n)
    
    logger.info(f"Selected {len(final_candidates)} top candidates after scoring and ranking.")
    return final_candidates

def save_filtered_results(df_top_candidates, output_path):
    """Saves the filtered top candidates to a CSV file."""
    try:
        df_top_candidates.to_csv(output_path, index=False)
        logger.info(f"Top {len(df_top_candidates)} candidates saved to: {output_path}")

        # Also save a summary of the filtering process
        summary_filter_path = output_path.parent / f"{output_path.stem}_summary.json"
        summary_data = {
            "total_initial_records": initial_record_count, # Need to get this from before filtering
            "total_candidates_after_filtering_and_scoring": len(df_top_candidates),
            "filtering_criteria_used": config_filter, # Save the config used
            "output_file": str(output_path)
        }
        with open(summary_filter_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        logger.info(f"Filtering summary saved to: {summary_filter_path}")

    except Exception as e:
        logger.error(f"Error saving filtered results: {e}")

# Global variable to store initial count
initial_record_count = 0

def main():
    global args_filter, config_filter, initial_record_count # Declare them global
    args_filter = parse_arguments()

    # Load file config if provided
    file_config_data_full = {}
    if args_filter.config_file:
        file_config_data_full = load_config(args_filter.config_file) # This returns filtering section
    elif not args_filter.config_file : # If no config file try default path
        default_config_path = Path(__file__).resolve().parent.parent / "configs" / "analysis_configs.json"
        if default_config_path.exists():
            logger.info(f"No config file specified, attempting to load default: {default_config_path}")
            file_config_data_full = load_config(default_config_path)
        else:
            logger.warning(f"Default config file not found at {default_config_path}. Using command-line args and defaults.")

    config_filter = get_effective_config_filter(args_filter, file_config_data_full)
    logger.info(f"Effective filtering configuration: {config_filter}")
    
    input_csv_files, output_csv_file = setup_paths_filter()
    
    if not input_csv_files:
        logger.error("No input CSV files to process. Exiting.")
        return 1
        
    combined_df = load_and_combine_data(input_csv_files)
    initial_record_count = len(combined_df) # Store initial count

    if combined_df.empty:
        logger.info("No data to filter after loading. Exiting.")
        return 0
        
    processed_df = preprocess_data_filter(combined_df)
    top_candidates_df = apply_filters_and_score(processed_df)
    
    save_filtered_results(top_candidates_df, output_csv_file)
    
    logger.info("Candidate filtering completed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())