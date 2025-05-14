#!/usr/bin/env python
"""
Run VGG Feature Extraction on Connectome Data

This script runs a simplified pipeline to extract features from connectome data
using the VGG3D model and save them to CSV files.
"""

import os
import argparse
import sys
import traceback

# Handle NumPy version issues
try:
    from synapse import config
    from synapse_pipeline import SynapsePipeline
except ImportError as e:
    if "numpy.dtype size changed" in str(e):
        print("WARNING: Dependency conflict detected. Trying to fix NumPy version...")
        try:
            # Attempt to update packages to compatible versions
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "numpy==1.24.3", "--quiet"])
            print("NumPy has been updated. Please restart the script.")
        except Exception as install_error:
            print(f"Failed to fix dependencies automatically: {install_error}")
            print("You may need to manually install compatible versions:")
            print("pip install numpy==1.24.3")
        sys.exit(1)
    else:
        print(f"Import error: {e}")
        traceback.print_exc()
        sys.exit(1)
except Exception as e:
    print(f"Unexpected error during imports: {e}")
    traceback.print_exc()
    sys.exit(1)

def main():
    """
    Main function to parse arguments and run the feature extraction pipeline
    """
    parser = argparse.ArgumentParser(description='Extract features from connectome data using VGG3D')
    
    # Connectome-related arguments
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Number of samples to load from connectome')
    parser.add_argument('--policy', type=str, default='dummy', choices=['dummy', 'random'],
                        help='Sampling policy for connectome data (dummy = synthetic data)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output during connectome sampling')
    
    # Feature extraction arguments
    parser.add_argument('--pooling_method', type=str, default='avg', 
                        choices=['avg', 'max', 'concat_avg_max', 'spp'],
                        help='Pooling method for feature extraction')
    parser.add_argument('--size', type=int, default=80,
                        help='Size of the 3D subvolume for processing')
    parser.add_argument('--segmentation_type', type=int, default=1,
                        choices=range(0, 11),
                        help='Type of segmentation overlay (0-10)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Alpha value (0-1) for overlaying segmentation')
    
    # Output directory arguments
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Base directory to save results')
    
    # Parse arguments and update config
    args = parser.parse_args()
    
    # Set configuration from arguments
    config.connectome_batch_size = args.batch_size
    config.connectome_policy = args.policy
    config.connectome_verbose = args.verbose
    config.pooling_method = args.pooling_method
    config.size = args.size
    config.segmentation_type = args.segmentation_type
    config.alpha = args.alpha
    
    # Set output directories
    config.results_dir = args.output_dir
    config.csv_output_dir = os.path.join(args.output_dir, 'csv_outputs')
    os.makedirs(config.csv_output_dir, exist_ok=True)
    
    print(f"Configuration:")
    print(f"  Connectome batch size: {config.connectome_batch_size}")
    print(f"  Connectome policy: {config.connectome_policy}")
    print(f"  Pooling method: {config.pooling_method}")
    print(f"  Segmentation type: {config.segmentation_type}")
    print(f"  Alpha: {config.alpha}")
    print(f"  Output directory: {config.results_dir}")
    
    # Run the pipeline
    try:
        pipeline = SynapsePipeline(config)
        results = pipeline.run_pipeline(pooling_method=config.pooling_method)
        
        print(f"\nFeatures successfully extracted!")
        print(f"Results saved to: {results['results_dir']}")
        return 0
    except Exception as e:
        if "numpy.dtype size changed" in str(e):
            print("\nERROR: NumPy version conflict detected.")
            print("This is usually caused by a mismatch between NumPy versions.")
            print("\nTo fix this issue, try updating NumPy: pip install numpy==1.24.3")
        else:
            print(f"Error running pipeline: {str(e)}")
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 