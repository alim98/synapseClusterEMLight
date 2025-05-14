"""
Synapse Sampling Adapter for the Synapse Pipeline

This adapter integrates the synapse_sampling module with the existing pipeline
to enable sampling from the connectome.
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, List, Union, Optional
from tqdm import tqdm

from synapse_sampling.synapse_sampling import sample_synapses
from newdl.dataloader3 import SynapseDataLoader, Synapse3DProcessor
from newdl.dataset3 import SynapseDataset

class SynapseSamplingDataset(torch.utils.data.Dataset):
    """
    Dataset class that uses synapse_sampling to generate samples on-the-fly
    """
    
    def __init__(self, 
                 batch_size: int = 100, 
                 policy: str = "dummy",  
                 processor = None,
                 segmentation_type: int = 10,  
                 alpha: float = 0.3,
                 verbose: bool = False):
        """
        Initialize the dataset with synapse_sampling parameters
        
        Args:
            batch_size: Number of samples to generate
            policy: Sampling policy ("dummy" or "random")
            processor: Image processor for the raw data
            segmentation_type: Type of segmentation (default: 10)
            alpha: Alpha value for blending mask with raw data
            verbose: Whether to show progress information
        """
        self.batch_size = batch_size
        self.policy = policy
        self.processor = processor if processor is not None else Synapse3DProcessor()
        self.segmentation_type = segmentation_type
        self.alpha = alpha
        self.verbose = verbose
        
        # Generate samples
        if self.verbose:
            print(f"Generating {batch_size} samples using {policy} policy...")
        self.raw_data, self.masks = sample_synapses(batch_size=batch_size, policy=policy, verbose=verbose)
        
        # Create minimal metadata
        self.metadata = []
        for i in range(batch_size):
            self.metadata.append({
                'bbox_name': f'sample_{i}',
                'Var1': i
            })
        
        if self.verbose:
            print(f"Generated {len(self.raw_data)} samples with shape {self.raw_data.shape}")

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        # Get raw data and mask
        raw = self.raw_data[idx]
        mask = self.masks[idx]
        
        # Blend raw and mask using alpha
        blended = raw * (1 - self.alpha) + mask * self.alpha * 255
        
        # Normalize
        blended = blended / 255.0
        
        # Reshape for processing
        cube = np.transpose(blended, (1, 2, 0, 3))
        
        # Extract frames
        frames = [cube[..., z] for z in range(cube.shape[3])]
        
        # Process frames
        if self.processor is not None:
            inputs = self.processor(frames, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0).float()
        else:
            # Basic processing if no processor is provided
            pixel_values = torch.tensor(np.stack(frames)).float()
        
        # Get metadata
        metadata = self.metadata[idx]
        
        return pixel_values, pd.Series(metadata), metadata['bbox_name']


class SynapseSamplingAdapter:
    """
    Adapter class that integrates synapse_sampling with the synapse pipeline
    """
    
    def __init__(self, config_obj=None):
        """
        Initialize the adapter with the given configuration
        
        Args:
            config_obj: Configuration object
        """
        self.config = config_obj
        
    def create_dataset(self, 
                       batch_size: int = 100, 
                       policy: str = "dummy",
                       segmentation_type: int = 10, 
                       alpha: float = 0.3,
                       processor = None, 
                       verbose: bool = False) -> SynapseSamplingDataset:
        """
        Create a dataset that uses synapse_sampling to generate samples
        
        Args:
            batch_size: Number of samples to generate
            policy: Sampling policy ("dummy" or "random")
            segmentation_type: Type of segmentation
            alpha: Alpha value for blending
            processor: Image processor for the raw data
            verbose: Whether to show progress information
            
        Returns:
            SynapseSamplingDataset: Dataset ready for use in the pipeline
        """
        return SynapseSamplingDataset(
            batch_size=batch_size,
            policy=policy,
            processor=processor,
            segmentation_type=segmentation_type,
            alpha=alpha,
            verbose=verbose
        )
    
    def load_and_prepare_data_for_pipeline(self, 
                                          batch_size: int = 100, 
                                          policy: str = "dummy",
                                          verbose: bool = False) -> Tuple[Dict, pd.DataFrame]:
        """
        Load and prepare data for the synapse pipeline, compatible with the existing interface
        
        Args:
            batch_size: Number of samples to generate
            policy: Sampling policy ("dummy" or "random")
            verbose: Whether to show progress information
            
        Returns:
            tuple: (vol_data_dict, synapse_df) - Compatible with the existing pipeline
        """
        # Get raw data and masks
        raw_data, masks = sample_synapses(batch_size=batch_size, policy=policy, verbose=verbose)
        
        # Prepare data dictionary
        vol_data_dict = {}
        
        # Prepare minimal metadata
        synapse_data = []
        
        for i in range(batch_size):
            bbox_name = f"sample_{i}"
            
            # Store raw data and mask (using mask for both seg_vol and add_mask_vol)
            vol_data_dict[bbox_name] = (np.squeeze(raw_data[i]), np.squeeze(masks[i]), np.squeeze(masks[i]))
            
            # Only add minimal required metadata
            synapse_data.append({
                'bbox_name': bbox_name,
                'Var1': i
            })
        
        # Create DataFrame
        synapse_df = pd.DataFrame(synapse_data)
        
        if verbose:
            print(f"Created vol_data_dict with {len(vol_data_dict)} samples")
            print(f"Created synapse_df with {len(synapse_df)} rows")
        
        return vol_data_dict, synapse_df


if __name__ == "__main__":
    # Test the adapter
    adapter = SynapseSamplingAdapter()
    
    # Test dataset creation
    dataset = adapter.create_dataset(batch_size=10, policy="dummy", verbose=True)
    print(f"Dataset size: {len(dataset)}")
    
    # Test sample access
    pixel_values, metadata, bbox_name = dataset[0]
    print(f"Sample shape: {pixel_values.shape}")
    print(f"Sample bbox_name: {bbox_name}")
    
    # Test pipeline data preparation
    vol_data_dict, synapse_df = adapter.load_and_prepare_data_for_pipeline(batch_size=10, policy="dummy", verbose=True)
    print(f"vol_data_dict size: {len(vol_data_dict)}")
    print(f"synapse_df size: {len(synapse_df)}") 