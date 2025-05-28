import os
import glob
import io
from typing import List, Tuple
import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torchvision import transforms
from scipy.ndimage import label
from scipy import ndimage
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import umap
import shutil
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path
import argparse
from sklearn.cluster import KMeans

# Import from newdl module instead of synapse
from newdl.dataloader import SynapseDataLoader, Synapse3DProcessor
from newdl.dataset import SynapseDataset
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


def extract_and_save_features(model, dataset, config, seg_type, alpha, output_dir, extraction_method='standard', layer_num=20, pooling_method='avg', perform_clustering=False):
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
        perform_clustering: Whether to perform clustering analysis (default: False)
        
    Returns:
        Path to the saved features CSV file
    """
    print(f"Extracting features for SegType {seg_type} and Alpha {alpha} using {extraction_method} method with {pooling_method} pooling")
    
    # Extract features based on the chosen method
    if extraction_method == 'stage_specific':
        features_df = extract_stage_specific_features(model, dataset, config, layer_num, pooling_method)
        method_suffix = f"_layer{layer_num}"
        feat_prefix = f"layer{layer_num}_feat_"
    else:  # standard method
        features_df = extract_features(model, dataset, config, pooling_method)
        method_suffix = ""
        feat_prefix = "feat_"
    
    print(f"Features extracted for SegType {seg_type} and Alpha {alpha}")
    
    # Format alpha value for filename
    alpha_str = str(alpha).replace('.', '_')
    
    # Create filename with method and pooling information
    csv_filename = f"features{method_suffix}_{pooling_method}_seg{seg_type}_alpha{alpha_str}.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving features to {csv_filepath}")
    features_df.to_csv(csv_filepath, index=False)
    
    print("Processing features for UMAP")
    feature_cols = [c for c in features_df.columns if c.startswith(feat_prefix)]
    features = features_df[feature_cols].values
    print(f"Features shape: {features.shape}")
    
    features_scaled = StandardScaler().fit_transform(features)
    print("Features scaled")
    
    features_df['segmentation_type'] = seg_type
    features_df['alpha'] = alpha
    features_df['extraction_method'] = extraction_method
    if extraction_method == 'stage_specific':
        features_df['layer_num'] = layer_num
    features_df['pooling_method'] = pooling_method
    
    # Save the updated features DataFrame with UMAP coordinates and metadata
    features_df.to_csv(csv_filepath, index=False)
    print(f"Updated features saved to {csv_filepath}")
    
    return csv_filepath

def run_full_analysis(config, vol_data_dict, syn_df, processor, model):
    output_dir = config.csv_output_dir
    
    segmentation_types = [config.segmentation_type] if isinstance(config.segmentation_type, int) else config.segmentation_type
    alpha_values = [config.alpha] if isinstance(config.alpha, (int, float)) else config.alpha
    
    # Feature extraction method - default to standard if not specified
    extraction_method = getattr(config, 'extraction_method', 'standard')
    
    # Layer number for stage-specific extraction (only used if extraction_method is 'stage_specific')
    layer_num = getattr(config, 'layer_num', 20)
    
    # Pooling method - default to average pooling if not specified
    pooling_method = getattr(config, 'pooling_method', 'avg')
    
    combined_features = []
    
    print(f"Using feature extraction method: {extraction_method}")
    if extraction_method == 'stage_specific':
        print(f"Extracting features from layer {layer_num}")
    print(f"Using pooling method: {pooling_method}")
    
    for seg_type in segmentation_types:
        for alpha in alpha_values:
            print(f"\n{'='*80}\nAnalyzing segmentation type {seg_type} with alpha {alpha}\n{'='*80}")
            
            current_dataset = SynapseDataset(
                vol_data_dict=vol_data_dict,
                synapse_df=syn_df,
                processor=processor,
                segmentation_type=seg_type,
                alpha=alpha
            )
            
            # Step 1: Extract and save features if not skipped
            if not hasattr(config, 'skip_feature_extraction') or not config.skip_feature_extraction:
                features_path = extract_and_save_features(
                    model, 
                    current_dataset, 
                    config, 
                    seg_type, 
                    alpha, 
                    output_dir,
                    extraction_method=extraction_method,
                    layer_num=layer_num,
                    pooling_method=pooling_method,
                    perform_clustering=(not hasattr(config, 'skip_clustering') or not config.skip_clustering)
                )
            else:
                # Load existing features
                alpha_str = str(alpha).replace('.', '_')
                
                # Create appropriate filename based on extraction method and pooling method
                if extraction_method == 'stage_specific':
                    features_path = os.path.join(output_dir, f"features_layer{layer_num}_{pooling_method}_seg{seg_type}_alpha{alpha_str}.csv")
                else:
                    features_path = os.path.join(output_dir, f"features_{pooling_method}_seg{seg_type}_alpha{alpha_str}.csv")
                
                if not os.path.exists(features_path):
                    print(f"Error: Feature file not found at {features_path}")
                    print("Please run without --skip_feature_extraction first")
                    return
                print(f"Loading existing features from {features_path}")
            
            # Step 2: Perform clustering analysis if not skipped
            if not hasattr(config, 'skip_clustering') or not config.skip_clustering:
                try:
                    # Read features
                    features_df = pd.read_csv(features_path)
                    combined_features.append(features_df)
                    
                    # Add segmentation type and alpha as columns for the combined analysis
                    features_df['seg_type'] = seg_type
                    features_df['alpha'] = alpha
                    
                    # Convert alpha to string format for filenames
                    alpha_str = str(alpha).replace('.', '_')
                    
                    # Save sample visualizations
                    
                    # Clustering analysis for this segmentation type and alpha
                    seg_output_dir = os.path.join(output_dir, f"seg{seg_type}_alpha{alpha_str}_{pooling_method}")
                    os.makedirs(seg_output_dir, exist_ok=True)
                    
                    # Perform UMAP and clustering
                    # run_clustering_analysis(features_df, seg_output_dir)
                except Exception as e:
                    print(f"Error during analysis for seg_type={seg_type}, alpha={alpha}: {str(e)}")
                    import traceback
                    traceback.print_exc()
    
    # Perform combined clustering analysis if not skipped and there are features
    if (not hasattr(config, 'skip_clustering') or not config.skip_clustering) and combined_features:
        print("\n\nRunning clustering analysis on combined features from all segmentation types...")
        try:
            # Combine all features
            combined_df = pd.concat(combined_features, ignore_index=True)
            
            # Save combined features
            combined_dir = os.path.join(output_dir, f"combined_analysis_{pooling_method}")
            os.makedirs(combined_dir, exist_ok=True)
            combined_df.to_csv(os.path.join(combined_dir, "combined_features.csv"), index=False)
            
        except Exception as e:
            print(f"Error during combined clustering analysis: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("Main analysis complete!")

    # Check if only feature extraction and clustering should be run
    if hasattr(config, 'only_feature_extraction_and_clustering') and config.only_feature_extraction_and_clustering:
        print("Only running feature extraction and clustering as requested.")
        return

    # Run presynapse analysis if not skipped
    if not hasattr(config, 'skip_presynapse_analysis') or not config.skip_presynapse_analysis:
        print("\n\n" + "="*80)
        print("Starting presynapse analysis to identify synapses with the same presynapse ID")
        print("="*80)
    
    print("All analyses complete!")

    # Generate comprehensive reports if not skipped
    if not hasattr(config, 'skip_report_generation') or not config.skip_report_generation:
        try:
            from report_generator import SynapseReportGenerator
            
            print("\n\n" + "="*80)
            print("Generating comprehensive reports")
            print("="*80)
            
            report_generator = SynapseReportGenerator(
                csv_output_dir=config.csv_output_dir,
                clustering_output_dir=config.clustering_output_dir,
                report_output_dir=config.report_output_dir
            )
            
            # Generate reports
            comprehensive_report = report_generator.generate_complete_report()
            presynapse_summary = report_generator.generate_presynapse_summary()
            
            if comprehensive_report:
                print(f"Comprehensive report generated at: {comprehensive_report}")
            else:
                print("Failed to generate comprehensive report")
                
            if presynapse_summary:
                print(f"Presynapse summary report generated at: {presynapse_summary}")
            else:
                print("Failed to generate presynapse summary report")
        
        except Exception as e:
            print(f"Error generating reports: {e}")
            import traceback
            traceback.print_exc()
    
    print("Pipeline complete!")

def load_and_prepare_data(config):
    data_loader = SynapseDataLoader(
        raw_base_dir=config.raw_base_dir,
        seg_base_dir=config.seg_base_dir,
        add_mask_base_dir=config.add_mask_base_dir
    )
    
    vol_data_dict = {}
    for bbox_name in tqdm(config.bbox_name, desc="Loading volumes"):
        raw_vol, seg_vol, add_mask_vol = data_loader.load_volumes(bbox_name)
        if raw_vol is not None:
            vol_data_dict[bbox_name] = (raw_vol, seg_vol, add_mask_vol)

    syn_df = pd.concat([
        pd.read_excel(os.path.join(config.excel_file, f"{bbox}.xlsx")).assign(bbox_name=bbox)
        for bbox in config.bbox_name if os.path.exists(os.path.join(config.excel_file, f"{bbox}.xlsx"))
    ])
    
    return vol_data_dict, syn_df

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
        print("Downloaded VGG3D checkpoint.")
    else:
        print("VGG3D checkpoint already exists.")

    model = Vgg3D(input_size=(80, 80, 80), fmaps=24, output_classes=7, input_fmaps=1)
    model = load_model_from_checkpoint(model, checkpoint_path)
    print("Model loaded from hemibrain_production.checkpoint")
    
    return model

def main():
    config.parse_args()
    
    # Setup directories
    os.makedirs(config.csv_output_dir, exist_ok=True)
    os.makedirs(config.save_gifs_dir, exist_ok=True)
    
    # Initialize VGG3D model
    model = VGG3D()

    # Load and prepare data
    vol_data_dict, syn_df = load_and_prepare_data(config)

    processor = Synapse3DProcessor(size=config.size)

    # Set flags to skip parts of the analysis
    for flag in ['skip_feature_extraction', 'skip_clustering', 'skip_presynapse_analysis', 
                'skip_report_generation', 'only_feature_extraction_and_clustering']:
        if hasattr(args, flag) and getattr(args, flag):
            setattr(config, flag, True)
            
    # If only_feature_extraction_and_clustering is set, also set skip flags for later steps
    if hasattr(config, 'only_feature_extraction_and_clustering') and config.only_feature_extraction_and_clustering:
        config.skip_presynapse_analysis = True
        config.skip_report_generation = True

    # Run the full analysis pipeline
    run_full_analysis(config, vol_data_dict, syn_df, processor, model)

    print("All analyses complete!")

    # # Generate comprehensive reports if not skipped
    # if not hasattr(config, 'skip_report_generation') or not config.skip_report_generation:
    #     try:
    #         from report_generator import SynapseReportGenerator
            
            # print("\n\n" + "="*80)
            # print("Generating comprehensive reports")
            # print("="*80)
            
            # report_generator = SynapseReportGenerator(
            #     csv_output_dir=config.csv_output_dir,
            #     clustering_output_dir=config.clustering_output_dir,
            #     report_output_dir=config.report_output_dir
            # )
            
            # Generate reports
            # comprehensive_report = report_generator.generate_complete_report()
            # presynapse_summary = report_generator.generate_presynapse_summary()
            
            # if comprehensive_report:
            #     print(f"Comprehensive report generated at: {comprehensive_report}")
            # else:
            #     print("Failed to generate comprehensive report")
                
        #     if presynapse_summary:
        #         print(f"Presynapse summary report generated at: {presynapse_summary}")
        #     else:
        #         print("Failed to generate presynapse summary report")
        
        # except Exception as e:
        #     print(f"Error generating reports: {e}")
        #     import traceback
        #     traceback.print_exc()
    
    print("Pipeline complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the analysis pipeline')
    parser.add_argument('--segmentation_type', type=int, default=config.segmentation_type, 
                       choices=range(0, 11), help='Type of segmentation overlay (0-10)')
    parser.add_argument('--gray_color', type=float, default=config.gray_color,
                       help='Gray color value (0-1) for overlaying segmentation')
    
    # Add flags to skip parts of the analysis (for GUI integration)
    parser.add_argument('--skip_feature_extraction', action='store_true',
                       help='Skip feature extraction and load existing features')
    parser.add_argument('--skip_clustering', action='store_true',
                       help='Skip clustering analysis')
    parser.add_argument('--skip_presynapse_analysis', action='store_true',
                       help='Skip presynapse analysis')
    parser.add_argument('--skip_report_generation', action='store_true',
                       help='Skip report generation')
    
    # Add flag to only run feature extraction and clustering
    parser.add_argument('--only_feature_extraction_and_clustering', action='store_true',
                       help='Only run feature extraction and clustering, skip presynapse analysis and report generation')
    
    args, _ = parser.parse_known_args()

    main()