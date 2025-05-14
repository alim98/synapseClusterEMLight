"""
Main script for running the synapse analysis pipeline.

This script demonstrates how to use the SynapsePipeline class
to orchestrate the entire synapse analysis workflow.
"""

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import traceback

# Add the parent directory to the path so we can import the synapse package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from synapse import config
from synapse_pipeline import SynapsePipeline
from newdl.dataloader3 import Synapse3DProcessor

# Import vesicle size visualization functions
from vesicle_size_visualizer import (
    visualize_vesicle_size_distribution, 
    create_umap_with_vesicle_sizes,
    analyze_vesicle_sizes_by_cluster,
    count_bboxes_in_clusters,
    plot_bboxes_in_clusters
)


def configure_pipeline_args():
    """Configure pipeline arguments by extending the existing config object"""
    # Create a new parser
    parser = argparse.ArgumentParser(description="Run the synapse analysis pipeline")
    
    # Add pipeline-specific arguments
    parser.add_argument("--only_vesicle_analysis", action="store_true",
                        help="Only run vesicle size analysis on existing results")
    
    # Add feature extraction method arguments
    parser.add_argument("--extraction_method", type=str, choices=['standard', 'stage_specific'],
                       help="Method to extract features ('standard' or 'stage_specific')")
    parser.add_argument("--layer_num", type=int,
                       help="Layer number to extract features from when using stage_specific method")
    
    # Add pooling method argument
    parser.add_argument("--pooling_method", type=str, choices=['avg', 'max', 'concat_avg_max', 'spp'],
                       help="Method to use for pooling ('avg', 'max', 'concat_avg_max', 'spp')")
    
    # Let config parse its arguments first
    config.parse_args()
    
    # Parse our additional arguments
    args, _ = parser.parse_known_args()
    
    # Add our arguments to config
    if args.only_vesicle_analysis:
        config.only_vesicle_analysis = True
    else:
        config.only_vesicle_analysis = False
    
    # Add feature extraction parameters if provided
    if args.extraction_method:
        config.extraction_method = args.extraction_method
    
    if args.layer_num:
        config.layer_num = args.layer_num
    
    # Add pooling method if provided
    if args.pooling_method:
        config.pooling_method = args.pooling_method
    
    return config


def run_vesicle_analysis():
    """
    Run vesicle analysis on existing features.
    This is a stub function that will be implemented with the full vesicle analysis logic.
    """
    pass


def analyze_vesicle_sizes(pipeline, features_df):
    """
    Analyze vesicle sizes using the provided features.
    
    Args:
        pipeline: SynapsePipeline instance
        features_df: DataFrame with extracted features
        
    Returns:
        dict: Analysis results
    """
    return {"status": "success"}


def main():
    """Main function to run the pipeline with vesicle analysis"""
    # Configure pipeline arguments
    config.parse_args()
    
    # Get extraction method parameters
    extraction_method = getattr(config, 'extraction_method', 'standard')
    layer_num = getattr(config, 'layer_num', 20)
    pooling_method = getattr(config, 'pooling_method', 'avg')
    
    # Create timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save original directories
    results_base_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(results_base_dir, exist_ok=True)
    
    # Save original directories
    original_csv_dir = config.csv_output_dir
    original_clustering_dir = config.clustering_output_dir
    original_gifs_dir = config.save_gifs_dir
    
    # Create new directories for this run
    config.csv_output_dir = os.path.join(results_base_dir, "csv_outputs")
    config.clustering_output_dir = os.path.join(results_base_dir, "clustering_results")
    config.save_gifs_dir = os.path.join(results_base_dir, "gifs")
    config.report_output_dir = os.path.join(results_base_dir, "reports")
    
    # Create all directories
    for directory in [config.csv_output_dir, config.clustering_output_dir, config.save_gifs_dir, config.report_output_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Check if only vesicle analysis should be run
    if hasattr(config, 'only_vesicle_analysis') and config.only_vesicle_analysis:
        # Implementation will be added
        pass
    else:
        try:
            # Run the full pipeline
            pipeline = SynapsePipeline(config)
            result = pipeline.run_full_pipeline(
                seg_type=config.segmentation_type,
                alpha=config.alpha,
                extraction_method=extraction_method,
                layer_num=layer_num,
                pooling_method=pooling_method
            )
            
            if result:
                # Run vesicle analysis if available
                pass
        except Exception as e:
            print(f"Error during pipeline execution: {str(e)}")
            import traceback
            print(traceback.format_exc())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        import traceback
        print(traceback.format_exc()) 
