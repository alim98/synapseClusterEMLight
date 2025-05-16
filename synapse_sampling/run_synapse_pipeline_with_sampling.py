"""
cd SynapseClusterEM
python synapse_sampling/run_synapse_pipeline_with_sampling.py --use_connectome --policy dummy --batch_size 5 num_samples 10 --verbose
"""

import os
import sys
import argparse
import torch
import pandas as pd
from pathlib import Path
from glob import glob
import time
import datetime


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from synapse import config
from synapse_pipeline import SynapsePipeline


from newdl.dataloader import Synapse3DProcessor

from synapse_sampling.adapter import SynapseConnectomeAdapter, ConnectomeDataset
from synapse_sampling.inference_patch import patch_extract_features, patch_extract_stage_specific_features


def configure_pipeline_args():
    """Configure pipeline arguments by extending the existing config object"""
    
    parser = argparse.ArgumentParser(description="Run the synapse analysis pipeline with connectome data")
    
    
    parser.add_argument("--only_vesicle_analysis", action="store_true",
                        help="Only run vesicle size analysis on existing results")
    
    
    parser.add_argument("--extraction_method", type=str, choices=['standard', 'stage_specific'],default='standard',
                       help="Method to extract features ('standard' or 'stage_specific')")
    parser.add_argument("--layer_num", type=int,default=20, 
                       help="Layer number to extract features from when using stage_specific method")
    
    
    parser.add_argument("--pooling_method", type=str, choices=['avg', 'max', 'concat_avg_max', 'spp'],default='avg',
                       help="Method to use for pooling ('avg', 'max', 'concat_avg_max', 'spp')")
    
    
    parser.add_argument("--use_connectome", action="store_true",
                        help="Use connectome data instead of local files", default=True)
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Number of samples to load from connectome")
    parser.add_argument("--policy", type=str, choices=['random', 'dummy'], default='dummy',
                        help="Sampling policy for connectome data")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose information during sampling")
    parser.add_argument("--num_samples", type=int, default=10,
                    help="Number of samples to load from connectome")
    
    
    config.parse_args()
    
    
    args, _ = parser.parse_known_args()    
    
    if args.extraction_method:
        config.extraction_method = args.extraction_method
    
    if args.layer_num:
        config.layer_num = args.layer_num
    
    
    if args.pooling_method:
        config.pooling_method = args.pooling_method
        
    
    config.use_connectome = args.use_connectome
    config.connectome_batch_size = args.batch_size
    config.connectome_policy = args.policy
    config.connectome_verbose = args.verbose
    config.connectome_num_samples = args.num_samples
    print(f"config.connectome_num_samples: {config.connectome_num_samples}")
    return config


def run_pipeline_with_connectome(config, timestamp):
    """
    Run the pipeline using connectome data instead of local files.
    
    This is a modified version of the pipeline that uses our adapter
    instead of the original data loading process.
    
    Args:
        config: Configuration object
        timestamp: Timestamp for creating output directories
        
    Returns:
        dict: Pipeline results
    """
    results_base_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(results_base_dir, exist_ok=True)
    
    
    # csv_dir = os.path.join(results_base_dir, "csv_outputs")
    # clustering_dir = os.path.join(results_base_dir, "clustering_results")
    # gifs_dir = os.path.join(results_base_dir, "gifs")
    # visualization_dir = os.path.join(results_base_dir, "visualizations")
    features_dir = os.path.join(results_base_dir, "features")
    
    
    for directory in [ features_dir]:
        os.makedirs(directory, exist_ok=True)
    
    
    pipeline = SynapsePipeline(config)
    pipeline.results_parent_dir = results_base_dir
    
    
    processor = Synapse3DProcessor(size=config.size)
    processor.normalize_volume = True
    
    
    dataset = ConnectomeDataset(
        processor=processor,
        segmentation_type=config.segmentation_type,
        alpha=config.alpha,
        num_samples=config.connectome_num_samples,
        batch_size=config.connectome_batch_size,
        policy=config.connectome_policy,
        verbose=config.connectome_verbose
    )
    
    
    pipeline.dataset = dataset
    pipeline.syn_df = dataset.synapse_df
    
    
    pipeline.load_model()
    
    
    extraction_method = getattr(config, 'extraction_method', 'standard')
    layer_num = getattr(config, 'layer_num', 20)
    pooling_method = getattr(config, 'pooling_method', 'avg')
    
    try:
        if extraction_method == 'stage_specific':
            pipeline.features_df = patch_extract_stage_specific_features(
                pipeline.model, 
                pipeline.dataset, 
                config,
                layer_num=layer_num,
                pooling_method=pooling_method
            )
        else:
            # Standard extraction method
            pipeline.features_df = patch_extract_features(
                pipeline.model,
                pipeline.dataset,
                config,
                pooling_method=pooling_method
            )
            
        
        # Check if features_df is properly defined
        features_path = os.path.join(features_dir, f"features_{timestamp}.csv")
        pipeline.features_df.to_csv(features_path, index=False)
        
        return {
            "features_df": pipeline.features_df,
            "model": pipeline.model,
            "dataset": pipeline.dataset,
            "results_dir": results_base_dir
        }
    except Exception as e:
        print(f"Error during pipeline execution: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # Create empty DataFrame to avoid AttributeError
        pipeline.features_df = pd.DataFrame()
        return None


def main():
    """Main function to run the pipeline"""
    configure_pipeline_args()
    
    extraction_method = getattr(config, 'extraction_method', 'standard')
    layer_num = getattr(config, 'layer_num', 20)
    pooling_method = getattr(config, 'pooling_method', 'avg')
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_base_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(results_base_dir, exist_ok=True)
    
    original_csv_dir = config.csv_output_dir
    original_clustering_dir = config.clustering_output_dir
    original_gifs_dir = config.save_gifs_dir
    
    config.csv_output_dir = os.path.join(results_base_dir, "csv_outputs")
    config.save_gifs_dir = os.path.join(results_base_dir, "gifs")
    
    config.report_output_dir = os.path.join(results_base_dir, "reports")
    
    if hasattr(config, 'use_connectome') and config.use_connectome:
        try:
            result = run_pipeline_with_connectome(config, timestamp)
        except Exception as e:
            print(f"Error during pipeline execution with connectome data: {str(e)}")
            import traceback
            print(traceback.format_exc())
    else:
        try:
            pipeline = SynapsePipeline(config)
            result = pipeline.run_full_pipeline(
                seg_type=config.segmentation_type,
                alpha=config.alpha,
                extraction_method=extraction_method,
                layer_num=layer_num,
                pooling_method=pooling_method
            )
        except Exception as e:
            print(f"Error during standard pipeline execution: {str(e)}")
            import traceback
            print(traceback.format_exc())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        import traceback
        print(traceback.format_exc()) 