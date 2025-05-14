"""
Synapse Feature Extraction 

This module provides functions for extracting features from 3D synapse data using VGG3D.
Only essential functions for feature extraction are included.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import argparse

from synapse import (
    Vgg3D, 
    load_model_from_checkpoint,
    config
)

# Import VGG3DStageExtractor for stage-specific feature extraction
from vgg3d_stage_extractor import VGG3DStageExtractor


def extract_features(model, dataset, config, pooling_method='avg'):
    """
    Extract features from the model.
    
    Args:
        model: The VGG3D model
        dataset: The dataset to extract features from
        config: Configuration object
        pooling_method: Method to use for pooling ('avg', 'max', 'concat_avg_max', 'spp')
        
    Returns:
        DataFrame with extracted features and metadata
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=0,
        collate_fn=lambda b: (
            torch.stack([item[0] for item in b if item is not None]) if any(item is not None for item in b) else torch.empty((0, 1, dataset.num_frames, dataset.subvol_size, dataset.subvol_size), device='cpu'),
            [item[1] for item in b if item is not None],
            [item[2] for item in b if item is not None]
        )
    )

    features = []
    metadata = []

    # Default implementation ('avg' pooling)
    if pooling_method == 'avg' or not pooling_method:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features", unit="batch"):
                if len(batch[0]) == 0:
                    continue
                    
                pixels, info, names = batch
                inputs = pixels.permute(0, 2, 1, 3, 4).to(device)

                batch_features = model.features(inputs)
                pooled_features = nn.AdaptiveAvgPool3d((1, 1, 1))(batch_features)

                batch_features_np = pooled_features.cpu().numpy()
                batch_size = batch_features_np.shape[0]
                num_features = np.prod(batch_features_np.shape[1:])
                batch_features_np = batch_features_np.reshape(batch_size, num_features)
                
                features.append(batch_features_np)
                metadata.extend(zip(names, info))

        features = np.concatenate(features, axis=0)

        metadata_df = pd.DataFrame([
            {"bbox": name, **info.to_dict()}
            for name, info in metadata
        ])

        feature_columns = [f'feat_{i+1}' for i in range(features.shape[1])]
        features_df = pd.DataFrame(features, columns=feature_columns)

        combined_df = pd.concat([metadata_df, features_df], axis=1)
    
    # Alternative pooling methods
    else:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features", unit="batch"):
                if len(batch[0]) == 0:
                    continue
                    
                pixels, info, names = batch
                inputs = pixels.permute(0, 2, 1, 3, 4).to(device)

                batch_features = model.features(inputs)
                
                # Different pooling strategies
                if pooling_method == 'max':
                    # Max pooling
                    pooled_features = nn.AdaptiveMaxPool3d((1, 1, 1))(batch_features)
                    batch_features_np = pooled_features.cpu().numpy()
                    batch_size = batch_features_np.shape[0]
                    num_features = np.prod(batch_features_np.shape[1:])
                    batch_features_np = batch_features_np.reshape(batch_size, num_features)
                    
                elif pooling_method == 'concat_avg_max':
                    # Concatenate average and max pooling
                    avg_pooled = nn.AdaptiveAvgPool3d((1, 1, 1))(batch_features)
                    max_pooled = nn.AdaptiveMaxPool3d((1, 1, 1))(batch_features)
                    
                    # Reshape both to 2D tensors and concatenate along feature dimension
                    batch_size = avg_pooled.size(0)
                    avg_features = avg_pooled.reshape(batch_size, -1)
                    max_features = max_pooled.reshape(batch_size, -1)
                    concat_features = torch.cat([avg_features, max_features], dim=1)
                    
                    batch_features_np = concat_features.cpu().numpy()
                    
                elif pooling_method == 'spp':
                    # Spatial Pyramid Pooling
                    # Pool at multiple resolutions (1x1x1, 2x2x2, 4x4x4) and concatenate
                    batch_size = batch_features.size(0)
                    
                    # 1x1x1 pooling
                    pool_1x1x1 = nn.AdaptiveAvgPool3d((1, 1, 1))(batch_features)
                    feat_1x1x1 = pool_1x1x1.reshape(batch_size, -1)
                    
                    # 2x2x2 pooling
                    pool_2x2x2 = nn.AdaptiveAvgPool3d((2, 2, 2))(batch_features)
                    feat_2x2x2 = pool_2x2x2.reshape(batch_size, -1)
                    
                    # 4x4x4 pooling (only if input size is large enough)
                    input_size = min(batch_features.size(2), batch_features.size(3), batch_features.size(4))
                    if input_size >= 4:
                        pool_4x4x4 = nn.AdaptiveAvgPool3d((4, 4, 4))(batch_features)
                        feat_4x4x4 = pool_4x4x4.reshape(batch_size, -1)
                        # Concatenate all pooling results
                        concat_features = torch.cat([feat_1x1x1, feat_2x2x2, feat_4x4x4], dim=1)
                    else:
                        # Only use 1x1x1 and 2x2x2 if input is too small
                        concat_features = torch.cat([feat_1x1x1, feat_2x2x2], dim=1)
                    
                    batch_features_np = concat_features.cpu().numpy()
                
                else:
                    raise ValueError(f"Unknown pooling method: {pooling_method}")
                
                features.append(batch_features_np)
                metadata.extend(zip(names, info))

        features = np.concatenate(features, axis=0)

        metadata_df = pd.DataFrame([
            {"bbox": name, **info.to_dict()}
            for name, info in metadata
        ])

        feature_columns = [f'feat_{i+1}' for i in range(features.shape[1])]
        features_df = pd.DataFrame(features, columns=feature_columns)
        
        # Add pooling method information to the dataframe
        features_df['pooling_method'] = pooling_method

        combined_df = pd.concat([metadata_df, features_df], axis=1)

    return combined_df


def extract_stage_specific_features(model, dataset, config, layer_num=20, pooling_method='avg'):
    """
    Extract features from a specific layer or stage of the VGG3D model.
    
    Args:
        model: The VGG3D model
        dataset: The dataset to extract features from
        config: Configuration object
        layer_num: Layer number to extract features from (default: 20 for best attention)
        pooling_method: Method to use for pooling ('avg', 'max', 'concat_avg_max', 'spp')
        
    Returns:
        DataFrame with extracted features and metadata
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    # Create the stage extractor
    extractor = VGG3DStageExtractor(model)
    
    # Get information about the model stages
    stage_info = extractor.get_stage_info()
    print(f"Extracting features from layer {layer_num}")
    
    # Find which stage contains this layer (for reference)
    for stage_num, info in stage_info.items():
        start_idx, end_idx = info['range']
        if start_idx <= layer_num <= end_idx:
            print(f"Layer {layer_num} is in Stage {stage_num}")
            break
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=0,
        collate_fn=lambda b: (
            torch.stack([item[0] for item in b if item is not None]) if any(item is not None for item in b) else torch.empty((0, 1, dataset.num_frames, dataset.subvol_size, dataset.subvol_size), device='cpu'),
            [item[1] for item in b if item is not None],
            [item[2] for item in b if item is not None]
        )
    )
    
    # Extract features from specified layer
    features = []
    metadata = []
    
    # Default implementation (avg pooling)
    if pooling_method == 'avg' or not pooling_method:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Extracting layer {layer_num} features", unit="batch"):
                if len(batch[0]) == 0:
                    continue
                    
                pixels, info, names = batch
                inputs = pixels.permute(0, 2, 1, 3, 4).to(device)
                
                # Extract features from specified layer
                batch_features = extractor.extract_layer(layer_num, inputs)
                
                # Global average pooling to get a feature vector
                batch_size = batch_features.shape[0]
                num_channels = batch_features.shape[1]
                
                # Reshape to (batch_size, channels, -1) for easier processing
                batch_features_reshaped = batch_features.reshape(batch_size, num_channels, -1)
                
                # Global average pooling across spatial dimensions
                pooled_features = torch.mean(batch_features_reshaped, dim=2)
                
                # Convert to numpy
                features_np = pooled_features.cpu().numpy()
                
                features.append(features_np)
                metadata.extend(zip(names, info))
        
        # Concatenate all features
        features = np.concatenate(features, axis=0)
        
        # Create metadata DataFrame
        metadata_df = pd.DataFrame([
            {"bbox": name, **info.to_dict()}
            for name, info in metadata
        ])
        
        # Create feature DataFrame
        feature_columns = [f'layer{layer_num}_feat_{i+1}' for i in range(features.shape[1])]
        features_df = pd.DataFrame(features, columns=feature_columns)
        
        # Combine metadata and features
        combined_df = pd.concat([metadata_df, features_df], axis=1)
    
    # Alternative pooling methods
    else:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Extracting layer {layer_num} features", unit="batch"):
                if len(batch[0]) == 0:
                    continue
                    
                pixels, info, names = batch
                inputs = pixels.permute(0, 2, 1, 3, 4).to(device)
                
                # Extract features from specified layer
                batch_features = extractor.extract_layer(layer_num, inputs)
                
                # Apply different pooling strategies
                batch_size = batch_features.shape[0]
                
                if pooling_method == 'max':
                    # Global max pooling
                    batch_features_reshaped = batch_features.reshape(batch_size, batch_features.shape[1], -1)
                    pooled_features = torch.max(batch_features_reshaped, dim=2)[0]  # [0] to get values, not indices
                    features_np = pooled_features.cpu().numpy()
                    
                elif pooling_method == 'concat_avg_max':
                    # Concatenate average and max pooling
                    batch_features_reshaped = batch_features.reshape(batch_size, batch_features.shape[1], -1)
                    avg_features = torch.mean(batch_features_reshaped, dim=2)
                    max_features = torch.max(batch_features_reshaped, dim=2)[0]
                    concat_features = torch.cat([avg_features, max_features], dim=1)
                    features_np = concat_features.cpu().numpy()
                    
                elif pooling_method == 'spp':
                    # Spatial Pyramid Pooling with different levels
                    # Get the current spatial dimensions
                    spatial_dims = batch_features.shape[2:]
                    min_dim = min(spatial_dims)
                    
                    # Initialize the list to store different pooling levels
                    pooled_features_list = []
                    
                    # 1x1x1 pooling (global average pooling)
                    pool_1x1x1 = nn.AdaptiveAvgPool3d((1, 1, 1))(batch_features)
                    pooled_features_list.append(pool_1x1x1.view(batch_size, -1))
                    
                    # 2x2x2 pooling (if dimension allows)
                    if min_dim >= 2:
                        pool_2x2x2 = nn.AdaptiveAvgPool3d((2, 2, 2))(batch_features)
                        pooled_features_list.append(pool_2x2x2.view(batch_size, -1))
                    
                    # 4x4x4 pooling (if dimension allows)
                    if min_dim >= 4:
                        pool_4x4x4 = nn.AdaptiveAvgPool3d((4, 4, 4))(batch_features)
                        pooled_features_list.append(pool_4x4x4.view(batch_size, -1))
                    
                    # Concatenate all pooled features
                    concat_features = torch.cat(pooled_features_list, dim=1)
                    features_np = concat_features.cpu().numpy()
                
                else:
                    raise ValueError(f"Unknown pooling method: {pooling_method}")
                
                features.append(features_np)
                metadata.extend(zip(names, info))

        # Concatenate all features
        features = np.concatenate(features, axis=0)
        
        # Create metadata DataFrame
        metadata_df = pd.DataFrame([
            {"bbox": name, **info.to_dict()}
            for name, info in metadata
        ])
        
        # Create feature DataFrame
        feature_columns = [f'layer{layer_num}_feat_{i+1}' for i in range(features.shape[1])]
        features_df = pd.DataFrame(features, columns=feature_columns)
        
        # Add pooling method information
        features_df['pooling_method'] = pooling_method
        
        # Combine metadata and features
        combined_df = pd.concat([metadata_df, features_df], axis=1)
    
    return combined_df


def extract_and_save_features(model, dataset, config, seg_type, alpha, output_dir, extraction_method='standard', layer_num=20, pooling_method='avg'):
    """
    Extract and save features using the specified extraction method.
    
    Args:
        model: The VGG3D model
        dataset: The dataset to extract features from
        config: Configuration object
        seg_type: Segmentation type
        alpha: Alpha value for segmentation
        output_dir: Directory to save features
        extraction_method: Feature extraction method ('standard' or 'stage_specific')
        layer_num: Layer number to extract features from if using stage_specific method
        pooling_method: Method to use for pooling ('avg', 'max', 'concat_avg_max', 'spp')
        
    Returns:
        Path to the saved features CSV file
    """
    # Extract features based on the chosen method
    if extraction_method == 'stage_specific':
        features_df = extract_stage_specific_features(model, dataset, config, layer_num, pooling_method)
        method_suffix = f"_layer{layer_num}"
        feat_prefix = f"layer{layer_num}_feat_"
    else:  # standard method
        features_df = extract_features(model, dataset, config, pooling_method)
        method_suffix = ""
        feat_prefix = "feat_"
    
    # Format alpha value for filename
    alpha_str = str(alpha).replace('.', '_')
    
    # Create filename with method and pooling information
    csv_filename = f"features{method_suffix}_{pooling_method}_seg{seg_type}_alpha{alpha_str}.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Add metadata
    features_df['segmentation_type'] = seg_type
    features_df['alpha'] = alpha
    features_df['extraction_method'] = extraction_method
    if extraction_method == 'stage_specific':
        features_df['layer_num'] = layer_num
    features_df['pooling_method'] = pooling_method
    
    # Save features to CSV
    features_df.to_csv(csv_filepath, index=False)
    
    return csv_filepath


def VGG3D():
    """
    Initialize and load the VGG3D model.
    
    Returns:
        model: Loaded VGG3D model
    """
    checkpoint_url = "https://dl.dropboxusercontent.com/scl/fo/mfejaomhu43aa6oqs6zsf/AKMAAgT7OrUtruR0AQXZBy0/hemibrain_production.checkpoint.20220225?rlkey=6cmwxdvehy4ylztvsbgkfnrfc&dl=0"
    checkpoint_path = 'hemibrain_production.checkpoint'

    if not os.path.exists(checkpoint_path):
        os.system(f"wget -O {checkpoint_path} '{checkpoint_url}'")
    
    model = Vgg3D(input_size=(80, 80, 80), fmaps=24, output_classes=7, input_fmaps=1)
    model = load_model_from_checkpoint(model, checkpoint_path)
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from synapse data')
    parser.add_argument('--segmentation_type', type=int, default=1, 
                       choices=range(0, 11), help='Type of segmentation overlay (0-10)')
    args = parser.parse_args()
