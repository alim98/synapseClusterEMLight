"""
Synapse Analysis Pipeline - Feature Extraction

This simplified module provides feature extraction from connectome data using VGG3D.
"""

import os
import datetime
import pandas as pd
from pathlib import Path

from synapse import config
from synapse_sampling.adapter import ConnectomeDataset
from synapse.dl.dataloader import Synapse3DProcessor
from inference import extract_and_save_features

class SynapsePipeline:

    def __init__(self, config_obj=None):
        self.config = config_obj if config_obj else config
        self.model = None
        self.dataset = None
        self.run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.results_parent_dir = None
        
    def create_results_directory(self):
        if hasattr(self.config, 'results_dir') and self.config.results_dir:
            results_base_dir = self.config.results_dir
        else:
            if hasattr(self.config, 'csv_output_dir') and 'csv_outputs' in self.config.csv_output_dir:
                results_base_dir = os.path.dirname(self.config.csv_output_dir)
            else:
                workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
                results_base_dir = os.path.join(workspace_dir, "results")
        
        self.results_parent_dir = os.path.join(results_base_dir, f"run_{self.run_timestamp}")
        os.makedirs(self.results_parent_dir, exist_ok=True)
        
        return self.results_parent_dir
        
    def load_data(self):
        self.processor = Synapse3DProcessor(size=self.config.size)
        self.processor.normalize_volume = False
        num_samples = getattr(self.config, 'connectome_num_samples', 100)
        batch_size = getattr(self.config, 'connectome_batch_size', 10)
        policy = getattr(self.config, 'connectome_policy', 'dummy')
        verbose = getattr(self.config, 'connectome_verbose', False)
        
        self.dataset = ConnectomeDataset(
            processor=self.processor,
            segmentation_type=self.config.segmentation_type,
            alpha=self.config.alpha,
            num_samples=num_samples,
            batch_size=batch_size,
            policy=policy,
            verbose=verbose
        )
        
        return self.dataset
    
    def load_model(self):
        """
        Load VGG3D model
        """
        from inference import VGG3D
        self.model = VGG3D()
        return self.model
    
    def extract_features(self, pooling_method='avg'):
        """
        Extract features using VGG3D model and save results
        
        Args:
            pooling_method: Method to use for pooling ('avg', 'max', 'concat_avg_max', 'spp')
        """
        output_dir = os.path.join(self.results_parent_dir, f"features_{pooling_method}")
        os.makedirs(output_dir, exist_ok=True)
        
        features_path = extract_and_save_features(
            self.model, 
            self.dataset, 
            self.config, 
            self.config.segmentation_type, 
            self.config.alpha, 
            output_dir,
            extraction_method='standard',
            pooling_method=pooling_method
        )
        
        # Manually save the features as a backup
        if isinstance(features_path, pd.DataFrame):
            # If features_path is a DataFrame, save it directly
            backup_path = os.path.join(output_dir, f"backup_features_{pooling_method}.csv")
            features_path.to_csv(backup_path, index=False)
            self.features_df = features_path
        elif isinstance(features_path, str):
            if os.path.exists(features_path):
                self.features_df = pd.read_csv(features_path)
                print(f"Features loaded from: {features_path}")
            else:
                print(f"Features file not found at {features_path}, creating empty DataFrame")
                self.features_df = pd.DataFrame()
        else:
            print("Unexpected type for features_path, creating empty DataFrame")
            self.features_df = pd.DataFrame()
        return self.features_df
    
    def run_pipeline(self, pooling_method='avg'):
        """
        Run the simplified pipeline: load data, load model, extract features
        
        Args:
            pooling_method: Method to use for pooling ('avg', 'max', 'concat_avg_max', 'spp')
        """
        # Create results directory
        self.create_results_directory()
        
        # Create output directory
        os.makedirs(self.config.csv_output_dir, exist_ok=True)
        
        try:
            # Load data using connectome loader
            self.load_data()
            
            # Load VGG3D model
            self.load_model()
            
            # Extract features
            self.extract_features(pooling_method=pooling_method)
            
        except Exception as e:
            print(f"Error during pipeline execution: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return {
            "features_df": self.features_df,
            "model": self.model,
            "dataset": self.dataset,
            "results_dir": self.results_parent_dir
        }


def main():
    """Main function to run the pipeline"""
    
    config.parse_args()
    
    # Create results directory if needed
    if not hasattr(config, 'results_dir') or not config.results_dir:
        workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        config.results_dir = os.path.join(workspace_dir, "results")
    
    # Configure connectome settings
    config.connectome_num_samples = getattr(config, 'connectome_num_samples', 100)
    config.connectome_batch_size = getattr(config, 'connectome_batch_size', 10)
    config.connectome_policy = getattr(config, 'connectome_policy', 'dummy')
    config.connectome_verbose = getattr(config, 'connectome_verbose', False)
    
    # Set other config attributes needed
    config.segmentation_type = getattr(config, 'segmentation_type', 1)
    config.alpha = getattr(config, 'alpha', 1.0)
    config.pooling_method = getattr(config, 'pooling_method', 'avg')
    
    # Run the pipeline
    pipeline = SynapsePipeline(config)
    results = pipeline.run_pipeline(pooling_method=config.pooling_method)


if __name__ == "__main__":
    main() 