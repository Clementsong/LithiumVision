#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Summarizes results from the LithiumVision workflow.

This script gathers data from various stages (generation logs, analysis summaries,
filtered candidates) and produces a consolidated report, including key statistics
and potentially plots.
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # For potentially nicer plots
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('summarize_results')

# Project root assuming this script is in LithiumVision/scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Summarize LithiumVision project results.')
    parser.add_argument('--results_dir', type=str, default=str(PROJECT_ROOT / "results"),
                        help='Root directory where results (candidates, figures, tables) are stored.')
    parser.add_argument('--data_dir', type=str, default=str(PROJECT_ROOT / "data"),
                        help='Root directory where data (generated, analyzed) is stored.')
    parser.add_argument('--output_report_name', type=str, default="LithiumVision_summary_report",
                        help='Base name for the output summary report file (extension will be added).')
    parser.add_argument('--report_format', type=str, default='md', choices=['md', 'txt', 'json'],
                        help='Format for the summary report (md, txt, json).')
    return parser.parse_args()

def find_latest_file(directory, pattern):
    """Finds the most recently modified file in a directory matching a pattern."""
    try:
        files = list(Path(directory).glob(pattern))
        if not files:
            return None
        return max(files, key=os.path.getmtime)
    except Exception as e:
        logger.warning(f"Error finding latest file with pattern {pattern} in {directory}: {e}")
        return None

def load_generation_log(tables_dir):
    """Loads the generation campaign log."""
    log_path = tables_dir / "generation_campaign_log.csv"
    if not log_path.exists():
        logger.warning(f"Generation log not found: {log_path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(log_path)
    except Exception as e:
        logger.error(f"Error loading generation log {log_path}: {e}")
        return pd.DataFrame()

def load_analysis_summaries(analyzed_data_dir):
    """Loads all analysis summary JSON files."""
    summaries = []
    for summary_file in analyzed_data_dir.glob("**/analysis_summary_statistics.json"):
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
                data['source_experiment_dir'] = summary_file.parent.name # Add identifier
                summaries.append(data)
        except Exception as e:
            logger.warning(f"Error loading analysis summary {summary_file}: {e}")
    logger.info(f"Loaded {len(summaries)} analysis summary files.")
    return summaries

def load_filtered_candidates(candidates_dir):
    """Loads the latest top_candidates.csv file."""
    # candidates_file = candidates_dir / "top_candidates.csv" # Original fixed name
    candidates_file = find_latest_file(candidates_dir, "top_candidates*.csv") # Find latest
    
    if not candidates_file or not candidates_file.exists():
        logger.warning(f"Filtered candidates CSV not found in {candidates_dir} (tried pattern top_candidates*.csv).")
        return pd.DataFrame()
    try:
        logger.info(f"Loading candidates from: {candidates_file}")
        return pd.read_csv(candidates_file)
    except Exception as e:
        logger.error(f"Error loading candidates CSV {candidates_file}: {e}")
        return pd.DataFrame()

def load_visualization_summary(figures_dir):
    """Loads the visualization_info.json file."""
    vis_info_path = figures_dir / "structures" / "visualization_info.json"
    if not vis_info_path.exists():
        logger.warning(f"Visualization info file not found: {vis_info_path}")
        return {}
    try:
        with open(vis_info_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading visualization info {vis_info_path}: {e}")
        return {}

def generate_overall_stats(gen_log_df, analysis_summaries_list, candidates_df, vis_summary_dict):
    """Generates high-level statistics for the report."""
    stats = {"project_summary": {}, "generation_summary": {}, "analysis_summary": {}, 
             "filtering_summary": {}, "visualization_summary": {}}

    # Project Summary
    stats["project_summary"]["report_generated_on"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Could add total runtime if logged by run_workflow.py

    # Generation Summary
    if not gen_log_df.empty:
        stats["generation_summary"]["total_generation_runs_logged"] = int(gen_log_df.shape[0])
        stats["generation_summary"]["unique_chemical_systems_generated"] = gen_log_df['chemical_system'].nunique()
        successful_gens = gen_log_df[gen_log_df['status'].str.lower() == 'completed']
        stats["generation_summary"]["successful_generation_runs"] = int(successful_gens.shape[0])
        if not successful_gens.empty and 'generated_cif_zip_count' in successful_gens.columns:
             stats["generation_summary"]["total_structures_in_successful_zips"] = int(successful_gens['generated_cif_zip_count'].sum())
        else:
             stats["generation_summary"]["total_structures_in_successful_zips"] = 0
    else:
        stats["generation_summary"] = {"message": "No generation log found."}

    # Analysis Summary
    if analysis_summaries_list:
        stats["analysis_summary"]["total_analysis_runs_summarized"] = len(analysis_summaries_list)
        stats["analysis_summary"]["total_structures_processed_in_analyses"] = sum(s.get('total_structures_processed', 0) for s in analysis_summaries_list)
        stats["analysis_summary"]["total_successfully_parsed_in_analyses"] = sum(s.get('successfully_parsed', 0) for s in analysis_summaries_list)
        stats["analysis_summary"]["total_novel_by_formula"] = sum(s.get('novel_by_formula', 0) for s in analysis_summaries_list)
        stats["analysis_summary"]["total_stable_on_mp"] = sum(s.get('stable_on_mp_count', 0) for s in analysis_summaries_list)
        # Can add more aggregated stats from e_hull_distribution etc.
    else:
        stats["analysis_summary"] = {"message": "No analysis summaries found."}

    # Filtering Summary
    if not candidates_df.empty:
        stats["filtering_summary"]["number_of_top_candidates_selected"] = int(candidates_df.shape[0])
        if 'is_novel_by_formula' in candidates_df.columns:
            stats["filtering_summary"]["novel_formula_in_top_candidates"] = int(candidates_df['is_novel_by_formula'].sum())
        if 'is_stable_on_mp' in candidates_df.columns: # Assuming this column exists after analysis
            stats["filtering_summary"]["stable_on_mp_in_top_candidates"] = int(candidates_df['is_stable_on_mp'].sum())
        # Could load the filter_summary.json for more details
        latest_filter_summary_file = find_latest_file(PROJECT_ROOT / "results" / "candidates", "*_summary.json")
        if latest_filter_summary_file:
            try:
                with open(latest_filter_summary_file, 'r') as f_filt_sum:
                    stats["filtering_summary"]["details_from_latest_run"] = json.load(f_filt_sum)
            except Exception as e_fs:
                 logger.warning(f"Could not load filter summary {latest_filter_summary_file}: {e_fs}")

    else:
        stats["filtering_summary"] = {"message": "No candidates data found."}
        
    # Visualization Summary
    if vis_summary_dict:
        stats["visualization_summary"] = vis_summary_dict
    else:
        stats["visualization_summary"] = {"message": "No visualization summary found."}
        
    return stats

def create_plots_for_summary(gen_log_df, analysis_summaries_list, candidates_df, figures_output_dir):
    """Creates and saves plots for the summary report."""
    plots_paths = {}
    sns.set_theme(style="whitegrid")

    # 1. Chemical Systems Generation Count (from gen_log)
    if not gen_log_df.empty and 'chemical_system' in gen_log_df.columns:
        plt.figure(figsize=(10, 6))
        sys_counts = gen_log_df['chemical_system'].value_counts()
        sns.barplot(x=sys_counts.index, y=sys_counts.values)
        plt.title('Number of Generation Runs per Chemical System')
        plt.xlabel('Chemical System')
        plt.ylabel('Number of Runs')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_path = figures_output_dir / "summary_chem_sys_runs.png"
        plt.savefig(plot_path); plt.close()
        plots_paths["chem_sys_runs_plot"] = str(plot_path.relative_to(PROJECT_ROOT)) # Relative for embedding in MD
        logger.info(f"Saved chemical system runs plot to {plot_path}")

    # 2. Distribution of E_above_hull for top candidates
    if not candidates_df.empty and 'best_mp_e_above_hull' in candidates_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(candidates_df['best_mp_e_above_hull'].dropna(), kde=True, bins=20)
        plt.title('Distribution of E_above_hull for Top Candidates (MP Matches)')
        plt.xlabel('Best MP E_above_hull (eV/atom)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plot_path = figures_output_dir / "summary_candidates_ehull_dist.png"
        plt.savefig(plot_path); plt.close()
        plots_paths["candidates_ehull_plot"] = str(plot_path.relative_to(PROJECT_ROOT))
        logger.info(f"Saved candidates E_hull distribution plot to {plot_path}")

    # 3. Novelty vs Stability Scatter Plot for candidates (if data allows)
    if not candidates_df.empty and 'is_novel_by_formula' in candidates_df.columns and 'best_mp_e_above_hull' in candidates_df.columns:
        plt.figure(figsize=(10,8))
        # Create a categorical 'stability_status' for plotting
        ehull_threshold = 0.1 # Example, should ideally come from config used in filtering
        
        def stability_label(row):
            if pd.isna(row['best_mp_e_above_hull']): return "Unknown (Novel)"
            return "Stable" if row['best_mp_e_above_hull'] <= ehull_threshold else "Less Stable"
        
        temp_df = candidates_df.copy()
        temp_df['stability_status_plot'] = temp_df.apply(stability_label, axis=1)

        sns.scatterplot(data=temp_df, x='total_score', y='best_mp_e_above_hull', hue='is_novel_by_formula', style='stability_status_plot', size='num_sites_generated', alpha=0.7, sizes=(50,300))
        plt.title('Candidate Overview: Score, Stability, Novelty, Size')
        plt.xlabel('Total Score (from filter_candidates.py)')
        plt.ylabel('Best MP E_above_hull (eV/atom)')
        plt.legend(title='Properties', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.axhline(ehull_threshold, color='r', linestyle='--', label=f'E_hull threshold ({ehull_threshold} eV/atom)')
        plt.gca().invert_yaxis() # Lower e_hull is better
        plt.tight_layout()
        plot_path = figures_output_dir / "summary_candidates_overview_scatter.png"
        plt.savefig(plot_path); plt.close()
        plots_paths["candidates_overview_plot"] = str(plot_path.relative_to(PROJECT_ROOT))
        logger.info(f"Saved candidates overview scatter plot to {plot_path}")

    return plots_paths


def format_report_md(stats, plots_paths, candidates_df, output_path_md):
    """Formats the summary statistics and plots into a Markdown report."""
    with open(output_path_md, 'w') as f:
        f.write(f"# LithiumVision Project Summary Report\n")
        f.write(f"*Generated on: {stats['project_summary']['report_generated_on']}*\n\n")

        f.write("## 1. Overall Project Statistics\n")
        # Add some high-level numbers here from various sections if desired

        f.write("## 2. Generation Summary\n")
        for k, v in stats["generation_summary"].items(): f.write(f"- **{k.replace('_', ' ').title()}**: {v}\n")
        if plots_paths.get("chem_sys_runs_plot"):
            f.write(f"\n![Chem Sys Runs Plot]({plots_paths['chem_sys_runs_plot']})\n")
        f.write("\n")

        f.write("## 3. Analysis Summary\n")
        for k, v in stats["analysis_summary"].items(): f.write(f"- **{k.replace('_', ' ').title()}**: {v}\n")
        # Could embed aggregated e_hull distribution text/plot here
        f.write("\n")

        f.write("## 4. Filtering & Candidate Summary\n")
        for k, v in stats["filtering_summary"].items():
            if k == "details_from_latest_run" and isinstance(v, dict):
                f.write("- **Details from Latest Filter Run**:\n")
                for lk, lv in v.items(): f.write(f"  - **{lk.replace('_', ' ').title()}**: {lv}\n")
            else:
                f.write(f"- **{k.replace('_', ' ').title()}**: {v}\n")
        if plots_paths.get("candidates_ehull_plot"):
            f.write(f"\n![Candidates E_hull Plot]({plots_paths['candidates_ehull_plot']})\n")
        if plots_paths.get("candidates_overview_plot"):
            f.write(f"\n![Candidates Overview Plot]({plots_paths['candidates_overview_plot']})\n")
        f.write("\n")
        
        f.write("### Top Candidates List (Snapshot)\n")
        if not candidates_df.empty:
            # Select key columns for the report
            cols_to_show = ['cif_file', 'formula_generated', 'chemsys_generated', 'best_mp_e_above_hull', 'is_novel_by_formula', 'is_stable_on_mp', 'total_score']
            report_df = candidates_df[[col for col in cols_to_show if col in candidates_df.columns]].head(10)
            f.write(report_df.to_markdown(index=False))
            f.write("\n\n")
        else:
            f.write("*No candidate data to display.*\n\n")

        f.write("## 5. Visualization Summary\n")
        if stats["visualization_summary"].get("summary_visualization"):
            # Make path relative to project root for MD file
            vis_sum_path = Path(stats["visualization_summary"]["summary_visualization"])
            if vis_sum_path.is_absolute():
                try:
                    vis_sum_path_rel = vis_sum_path.relative_to(PROJECT_ROOT)
                    f.write(f"![Summary Visualization]({vis_sum_path_rel})\n\n")
                except ValueError: # If not relative, just use name
                     f.write(f"Summary visualization image: {vis_sum_path.name} (check figures/structures/)\n\n")

        for k, v in stats["visualization_summary"].items():
            if k != "summary_visualization" and k!= "visualizations": # Don't print the list of all viz here
                 f.write(f"- **{k.replace('_', ' ').title()}**: {v}\n")
        f.write("\n")
        
        f.write("## 6. Next Steps / Future Work\n")
        f.write("- Perform DFT calculations for promising novel candidates to verify stability and predict properties.\n")
        f.write("- Conduct AIMD simulations for ionic conductivity predictions on stable structures.\n")
        f.write("- Expand chemical space exploration with MatterGen based on initial findings.\n")
        f.write("- Integrate advanced machine learning models (e.g., EquiformerV2, CGCNN) for direct property prediction if more data becomes available.\n")
        f.write("--- End of Report ---\n")
    logger.info(f"Markdown report saved to: {output_path_md}")

def format_report_json(stats, output_path_json):
    """Saves the summary statistics as a JSON file."""
    try:
        with open(output_path_json, 'w') as f:
            json.dump(stats, f, indent=2, cls=NpEncoder) # Use NpEncoder for numpy types
        logger.info(f"JSON report saved to: {output_path_json}")
    except Exception as e:
        logger.error(f"Failed to save JSON report: {e}")

class NpEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main_summarize():
    args = parse_arguments()

    results_base_dir = Path(args.results_dir)
    data_base_dir = Path(args.data_dir)

    # Define specific subdirectories
    tables_dir = results_base_dir / "tables"
    figures_dir = results_base_dir / "figures" # For saving new summary plots
    candidates_dir = results_base_dir / "candidates"
    analyzed_dir_root = data_base_dir / "analyzed" # Root for analysis summaries

    # Ensure output directories for summary report and its plots exist
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load all necessary data
    gen_log = load_generation_log(tables_dir)
    analysis_summaries = load_analysis_summaries(analyzed_dir_root)
    candidates = load_filtered_candidates(candidates_dir)
    vis_summary = load_visualization_summary(figures_dir) # figures_dir contains 'structures' subdir

    # Generate overall statistics
    overall_stats = generate_overall_stats(gen_log, analysis_summaries, candidates, vis_summary)
    
    # Create plots for the summary
    summary_plots_paths = create_plots_for_summary(gen_log, analysis_summaries, candidates, figures_dir)
    overall_stats["summary_plots_paths"] = summary_plots_paths # Add paths to stats for report

    # Generate report in the chosen format
    report_base_name = args.output_report_name
    
    if args.report_format == 'md':
        md_path = tables_dir / f"{report_base_name}.md"
        format_report_md(overall_stats, summary_plots_paths, candidates, md_path)
    elif args.report_format == 'json':
        json_path = tables_dir / f"{report_base_name}.json"
        format_report_json(overall_stats, json_path)
    elif args.report_format == 'txt':
        txt_path = tables_dir / f"{report_base_name}.txt"
        with open(txt_path, 'w') as f:
            f.write(json.dumps(overall_stats, indent=2, cls=NpEncoder)) # Simple TXT as pretty JSON
        logger.info(f"Text report (JSON format) saved to: {txt_path}")
    else:
        logger.error(f"Unsupported report format: {args.report_format}")
        return 1
        
    logger.info("Result summarization completed.")
    return 0

if __name__ == "__main__":
    sys.exit(main_summarize())