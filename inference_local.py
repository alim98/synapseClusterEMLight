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


from synapse.dl.dataloader import SynapseDataLoader, Synapse3DProcessor
from synapse.dl.dataset import SynapseDataset
from synapse import (

    Vgg3D, 
    load_model_from_checkpoint,

    config
)

from inference import (
    extract_features as base_extract_features,
    extract_stage_specific_features as base_extract_stage_specific_features,
    VGG3D as base_VGG3D
)

def extract_features(model, dataset, config, pooling_method='avg'):
    
    base_df = base_extract_features(model, dataset, config, pooling_method)
    
    if hasattr(dataset, '__getitem__') and len(dataset) > 0:
        try:
            
            sample = dataset[0]
            if sample and len(sample) >= 3:
                
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

                metadata = []
                for batch in dataloader:
                    if len(batch[0]) == 0:
                        continue
                    _, info, names = batch
                    metadata.extend(zip(names, info))
                
                
                metadata_df = pd.DataFrame([
                    {"bbox": name, **info.to_dict()}
                    for name, info in metadata
                ])
                
                
                feature_columns = [col for col in base_df.columns if col.startswith('feat_') or col.startswith('layer')]
                
                
                if len(metadata_df) == len(base_df):
                    
                    if pooling_method != 'avg':
                        base_df['pooling_method'] = pooling_method
                    
                    combined_df = pd.concat([metadata_df, base_df[feature_columns + (['pooling_method'] if pooling_method != 'avg' else [])]], axis=1)
                    return combined_df
                    
        except Exception as e:
            print(f"Warning: Could not extract detailed metadata, using base implementation: {e}")
    
    
    return base_df

def extract_stage_specific_features(model, dataset, config, layer_num=20, pooling_method='avg'):
    print(f"Extracting features from layer {layer_num}")
    
    
    base_df = base_extract_stage_specific_features(model, dataset, config, layer_num, pooling_method)
    
    
    if hasattr(dataset, '__getitem__') and len(dataset) > 0:
        try:
            
            sample = dataset[0]
            if sample and len(sample) >= 3:
                
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

                metadata = []
                for batch in dataloader:
                    if len(batch[0]) == 0:
                        continue
                    _, info, names = batch
                    metadata.extend(zip(names, info))
                
                
                metadata_df = pd.DataFrame([
                    {"bbox": name, **info.to_dict()}
                    for name, info in metadata
                ])
                
                
                feature_columns = [col for col in base_df.columns if col.startswith(f'layer{layer_num}_feat_')]
                
                
                if len(metadata_df) == len(base_df):
                    combined_df = pd.concat([metadata_df, base_df[feature_columns]], axis=1)
                    return combined_df
                    
        except Exception as e:
            print(f"Warning: Could not extract detailed metadata, using base implementation: {e}")
    
    
    return base_df



def extract_and_save_features(model, dataset, config, seg_type, alpha, output_dir, extraction_method='standard', layer_num=20, pooling_method='avg', perform_clustering=False):
    print(f"Extracting features for SegType {seg_type} and Alpha {alpha} using {extraction_method} method with {pooling_method} pooling")
    
    
    if extraction_method == 'stage_specific':
        features_df = extract_stage_specific_features(model, dataset, config, layer_num, pooling_method)
        method_suffix = f"_layer{layer_num}"
        feat_prefix = f"layer{layer_num}_feat_"
    else:  
        features_df = extract_features(model, dataset, config, pooling_method)
        method_suffix = ""
        feat_prefix = "feat_"
    
    print(f"Features extracted for SegType {seg_type} and Alpha {alpha}")
    
    
    alpha_str = str(alpha).replace('.', '_')
    
    
    csv_filename = f"features{method_suffix}_{pooling_method}_seg{seg_type}_alpha{alpha_str}.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    
    
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
    
    
    features_df.to_csv(csv_filepath, index=False)
    print(f"Updated features saved to {csv_filepath}")
    
    return csv_filepath

def run_full_analysis(config, vol_data_dict, syn_df, processor, model):
    output_dir = config.csv_output_dir
    
    segmentation_types = [config.segmentation_type] if isinstance(config.segmentation_type, int) else config.segmentation_type
    alpha_values = [config.alpha] if isinstance(config.alpha, (int, float)) else config.alpha
    
    
    extraction_method = getattr(config, 'extraction_method', 'standard')
    
    
    layer_num = getattr(config, 'layer_num', 20)
    
    
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
        )
          
    print("All analyses complete!")


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
    print("Initializing VGG3D model...")
    
    
    checkpoint_path = 'hemibrain_production.checkpoint'
    if os.path.exists(checkpoint_path):
        print("VGG3D checkpoint already exists.")
    else:
        print("Downloading VGG3D checkpoint...")
    
    model = base_VGG3D()
    print("Model loaded from hemibrain_production.checkpoint")
    
    return model

def main():
    config.parse_args()
    
    
    os.makedirs(config.csv_output_dir, exist_ok=True)
    os.makedirs(config.save_gifs_dir, exist_ok=True)
    
    
    model = VGG3D()

    
    vol_data_dict, syn_df = load_and_prepare_data(config)

    processor = Synapse3DProcessor(size=config.size)

    
    
    run_full_analysis(config, vol_data_dict, syn_df, processor, model)

    print("All analyses complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the analysis pipeline')
    parser.add_argument('--segmentation_type', type=int, default=config.segmentation_type, 
                       choices=range(0, 11), help='Type of segmentation overlay (0-10)')
    parser.add_argument('--gray_color', type=float, default=config.gray_color,
                       help='Gray color value (0-1) for overlaying segmentation')

    args, _ = parser.parse_known_args()

    main()
