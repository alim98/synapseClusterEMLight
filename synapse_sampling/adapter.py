import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, List, Any, Optional
import warnings


from synapse_sampling.synapse_sampling import sample_synapses




class SynapseConnectomeAdapter:
    """
    Adapter class to integrate synapse_sampling module with the existing pipeline.
    This replaces the functionality of SynapseDataLoader, providing sampled data from connectome.
    """
    def __init__(self, num_samples=100, batch_size=10, policy="random", verbose=False):
        """
        Initialize the adapter with sampling parameters
        
        Args:
            num_samples: Number of samples to read from connectome
            batch_size: Batch size for model inference
            policy: Sampling policy ("random" or "dummy")
            verbose: Whether to print verbose information
        """
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.policy = policy
        self.verbose = verbose
        
        print(f"DEBUG - SynapseConnectomeAdapter initialized with:")
        print(f"  num_samples: {num_samples}")
        print(f"  batch_size: {batch_size}")
        print(f"  policy: {policy}")
        print(f"  verbose: {verbose}")
        
    def get_batch(self, batch_indices):
        """
        Load a single batch of data dynamically using the synapse_sampling module.
        
        Args:
            batch_indices: Indices of the samples to load in this batch
            
        Returns:
            Tuple[Dict, DataFrame]: A tuple containing:
                - Dictionary mapping sample IDs to (raw_vol, seg_vol, mask_vol) tuples
                - DataFrame with metadata for the batch
        """
        if self.verbose:
            print(f"DEBUG - Loading batch with indices: {batch_indices}")
        
        # Get actual batch size (last batch may be smaller)
        actual_batch_size = len(batch_indices)
        
        # Sample raw volumes and masks for just this batch
        raw_volumes, mask_volumes = sample_synapses(batch_size=actual_batch_size, policy=self.policy, verbose=self.verbose)
        
        if self.verbose:
            print(f"DEBUG - Raw volumes shape: {raw_volumes.shape}")
            print(f"DEBUG - Mask volumes shape: {mask_volumes.shape}")
        
        # Prepare data dictionary
        vol_data_dict = {}
        
        # Prepare minimal metadata
        metadata_list = []
        
        # Process each sample
        for i in range(raw_volumes.shape[0]):
            # Extract volumes
            raw_vol = raw_volumes[i, 0]  
            mask_vol = mask_volumes[i, 0]  
            
            # Use mask for segmentation volume
            seg_vol = mask_vol.copy()
            
            # Create sample ID (using the real index from batch_indices)
            sample_id = f"sample_{batch_indices[i]+1}"
            
            # Store in dictionary
            vol_data_dict[sample_id] = (raw_vol, seg_vol, mask_vol)
            
            # Create minimal metadata
            metadata = {
                'bbox_name': sample_id,
                'Var1': batch_indices[i]
            }
            metadata_list.append(metadata)
        
        # Create DataFrame
        batch_df = pd.DataFrame(metadata_list)
        
        if self.verbose:
            print(f"DEBUG - Created batch_df with {len(batch_df)} rows")
        
        return vol_data_dict, batch_df
    
    def get_metadata(self):
        """
        Generate metadata for all samples without loading the actual data.
        
        Returns:
            DataFrame: Metadata for all samples
        """
        # Generate minimal metadata for all samples
        metadata_list = []
        for i in range(self.num_samples):
            metadata = {
                'bbox_name': f"sample_{i+1}",
                'Var1': i
            }
            metadata_list.append(metadata)
        
        # Create DataFrame
        synapse_df = pd.DataFrame(metadata_list)
        
        print(f"DEBUG - Created metadata for {len(synapse_df)} samples")
        
        return synapse_df


class ConnectomeDataset(Dataset):
    """
    Dataset class for connectome data that works with the existing pipeline.
    Similar to SynapseDataset but uses the adapter for data loading.
    """
    def __init__(self, processor, segmentation_type: int = 10, alpha: float = 1.0, 
                 num_samples: int = 100, batch_size: int = 10, policy: str = "random", verbose: bool = False):
        """
        Initialize the dataset with the adapter
        
        Args:
            processor: Processor for transforming images
            segmentation_type: Segmentation type to use (default 10)
            alpha: Alpha value for blending (default 1.0)
            num_samples: Number of samples to read from connectome
            batch_size: Batch size for model inference
            policy: Sampling policy ("random" or "dummy")
            verbose: Whether to print verbose information
        """
        # Create adapter
        self.adapter = SynapseConnectomeAdapter(num_samples, batch_size, policy, verbose)
        
        # Get metadata only (not loading all data)
        self.synapse_df = self.adapter.get_metadata()
        
        # Cache for batches
        self.cache = {}
        self.cached_batch_indices = set()
        
        # Store parameters
        self.processor = processor
        self.segmentation_type = segmentation_type
        self.alpha = alpha
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Default dimensions
        self.num_frames = 80  
        self.subvol_size = 80
    
    def __len__(self):
        return len(self.synapse_df)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Tuple: (pixel_values, metadata, sample_id)
        """
        # Get sample info
        syn_info = self.synapse_df.iloc[idx]
        sample_id = syn_info['bbox_name']
        
        # Calculate which batch this sample belongs to
        batch_idx = idx // self.batch_size
        batch_start = batch_idx * self.batch_size
        batch_end = min(batch_start + self.batch_size, len(self))
        batch_indices = list(range(batch_start, batch_end))
        
        # Check if we already have this batch cached
        if batch_idx not in self.cached_batch_indices:
            # Clear the cache if it's getting too large
            if len(self.cached_batch_indices) > 3:  # Keep only a few batches in memory
                self.cache = {}
                self.cached_batch_indices = set()
            
            # Load this batch
            vol_data_dict, _ = self.adapter.get_batch(batch_indices)
            
            # Store in cache
            self.cache[batch_idx] = vol_data_dict
            self.cached_batch_indices.add(batch_idx)
            
            if self.verbose:
                print(f"Loaded batch {batch_idx} into RAM")
        
        # Get volumes from cache
        vol_data_dict = self.cache[batch_idx]
        raw_vol, seg_vol, mask_vol = vol_data_dict.get(sample_id, (None, None, None))
        
        if raw_vol is None:
            print(f"Volume data not found for {sample_id}. Returning None.")
            return None
        
        # Handle 4D volumes if needed
        if len(raw_vol.shape) == 4 and raw_vol.shape[0] == 1:
            raw_vol = raw_vol[0]  
        
        if len(mask_vol.shape) == 4 and mask_vol.shape[0] == 1:
            mask_vol = mask_vol[0]  
        
        # Normalize raw volume
        raw_norm = raw_vol.astype(np.float32)
        min_val = raw_norm.min()
        max_val = raw_norm.max()
        if max_val > min_val:
            raw_norm = (raw_norm - min_val) / (max_val - min_val)
        
        # Create overlaid volume by blending raw and mask
        D, H, W = raw_norm.shape
        overlaid = np.zeros((D, H, W), dtype=np.float32)
        for d in range(D):
            slice_raw = raw_norm[d]
            slice_mask = mask_vol[d] > 0
            
            # Blend raw and mask using alpha
            overlaid_slice = slice_raw.copy()
            overlaid_slice[slice_mask] = slice_raw[slice_mask] * (1 - self.alpha) + self.alpha
            overlaid[d] = overlaid_slice
        
        # Convert to uint8 for processing
        overlaid_uint8 = (overlaid * 255).astype(np.uint8)
        
        # Convert to frames
        frames = [overlaid_uint8[d] for d in range(D)]
        
        # Process each frame
        processed_frames = []
        for frame in frames:
            processed_frame = self.processor.transform(frame)
            processed_frames.append(processed_frame)
        
        # Stack frames
        pixel_values = torch.stack(processed_frames)
        
        # Permute dimensions to match expected format
        pixel_values = pixel_values.permute(1, 0, 2, 3)
        
        return pixel_values, syn_info, sample_id 