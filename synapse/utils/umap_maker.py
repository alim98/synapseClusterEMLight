import os
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.cluster import KMeans
import umap

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
from tqdm import tqdm  

def merge_feature_datasets(csv1_path, csv2_path):
    """
    Merge two feature CSV files keeping all columns from both datasets
    
    Args:
        csv1_path: Path to first CSV file (SynapseClusterEM/features_20250602_174807.csv)
        csv2_path: Path to second CSV file (SynapseClusterEM/raven_results/features_avg_seg11_alpha1_0.csv)
    
    Returns:
        pd.DataFrame: Merged dataframe with all columns from both files
    """
    print(f"Loading first CSV: {csv1_path}")
    df1 = pd.read_csv(csv1_path)
    print(f"  - Shape: {df1.shape}")
    print(f"  - Columns: {list(df1.columns)}")
    
    print(f"Loading second CSV: {csv2_path}")
    df2 = pd.read_csv(csv2_path)
    print(f"  - Shape: {df2.shape}")
    print(f"  - Columns: {list(df2.columns)}")
    
    # Add a source column to track which dataset each row came from
    df1['source'] = 'dataset1'
    df2['source'] = 'dataset2'
    
    # Find common columns
    common_cols = set(df1.columns) & set(df2.columns)
    df1_only_cols = set(df1.columns) - common_cols
    df2_only_cols = set(df2.columns) - common_cols
    
    print(f"Common columns: {len(common_cols)}")
    print(f"Dataset1 only columns: {len(df1_only_cols)}")
    print(f"Dataset2 only columns: {len(df2_only_cols)}")
    
    # Perform outer join to keep all columns from both datasets
    # First, we need to ensure all columns exist in both dataframes
    for col in df2_only_cols:
        if col != 'source':
            df1[col] = np.nan
    
    for col in df1_only_cols:
        if col != 'source':
            df2[col] = np.nan
    
    # Now concatenate the dataframes
    merged_df = pd.concat([df1, df2], ignore_index=True, sort=False)
    
    print(f"Merged dataframe shape: {merged_df.shape}")
    print(f"Merged dataframe columns: {list(merged_df.columns)}")
    
    return merged_df

def compute_umap(features_scaled, n_components=3, n_neighbors=15, min_dist=0.1, metric='euclidean'):
    """Compute UMAP projection with progress feedback"""
    start_time = time.time()
    
    # Create and fit UMAP
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
        verbose=True  # Enable verbose output
    )
    
    umap_results = reducer.fit_transform(features_scaled)
    
    elapsed_time = time.time() - start_time
    
    return umap_results, reducer

def create_bbox_colored_umap(features_df, output_dir, reuse_umap_results=None, n_dimensions=3, output_format='both'):
    """Create a UMAP visualization specifically colored by bounding box
    
    Args:
        features_df: DataFrame with features and bbox column
        output_dir: Directory to save outputs
        reuse_umap_results: Pre-computed UMAP results to reuse
        n_dimensions: Number of dimensions for UMAP (2 or 3)
        output_format: Output format - 'html', 'png', or 'both'
    """
    if 'bbox' not in features_df.columns:
        print("No bbox column in features data, skipping bbox-colored UMAP")
        return
    
    # Validate parameters
    if n_dimensions not in [2, 3]:
        raise ValueError("n_dimensions must be 2 or 3")
    if output_format not in ['html', 'png', 'both']:
        raise ValueError("output_format must be 'html', 'png', or 'both'")
    
    # Define a consistent color map for bounding boxes bbox1-bbox7
    bbox_colors = {
        'bbox1': '#FF0000',  # Red
        'bbox2': '#00FF00',  # Green
        'bbox3': '#0000FF',  # Blue
        'bbox4': '#FFA500',  # Orange
        'bbox5': '#800080',  # Purple
        'bbox6': '#00FFFF',  # Cyan
        'bbox7': '#FFD700'   # Gold
    }
    
    # Preprocess bbox column to identify bbox1-bbox7 vs others
    features_df = features_df.copy()
    
    # Create a new column for coloring
    def categorize_bbox(bbox_val):
        bbox_str = str(bbox_val).lower()
        if bbox_str in ['bbox1', 'bbox2', 'bbox3', 'bbox4', 'bbox5', 'bbox6', 'bbox7']:
            return bbox_str
        else:
            return 'other'
    
    features_df['bbox_color'] = features_df['bbox'].apply(categorize_bbox)
    
    # Add gray color for 'other' category
    bbox_colors_extended = bbox_colors.copy()
    bbox_colors_extended['other'] = '#808080'  # Gray color for non-bbox1-7
    
    # Print statistics
    bbox_counts = features_df['bbox_color'].value_counts()
    print(f"Bbox distribution:")
    for bbox, count in bbox_counts.items():
        print(f"  {bbox}: {count} samples")
    
    # Simple feature column detection - check if 'feat_' appears anywhere in the column name
    feature_cols = [col for col in features_df.columns if 'feat_' in col]
    
    if not feature_cols:
        raise ValueError("No feature columns found in DataFrame")
    
    print(f"Found {len(feature_cols)} feature columns")
    
    # Extract features
    features = features_df[feature_cols].values
    
    # Scale features
    features_scaled = StandardScaler().fit_transform(features)
    
    # Use provided UMAP results or compute new ones
    if reuse_umap_results is not None:
        umap_results = reuse_umap_results
    else:
        # Compute UMAP with specified dimensions
        umap_results, _ = compute_umap(features_scaled, n_components=n_dimensions)
    
    # Add UMAP coordinates to dataframe
    features_df = features_df.copy()
    features_df['umap_x'] = umap_results[:, 0]
    features_df['umap_y'] = umap_results[:, 1]
    if n_dimensions == 3:
        features_df['umap_z'] = umap_results[:, 2]
    
    # Separate colored and gray samples for plotting order (colored on top)
    colored_mask = features_df['bbox_color'] != 'other'
    gray_df = features_df[~colored_mask].copy()
    colored_df = features_df[colored_mask].copy()
    
    print(f"Gray samples: {len(gray_df)}, Colored samples: {len(colored_df)}")
    
    # Create plotly figure based on dimensions
    if n_dimensions == 3:
        fig = go.Figure()
        
        # Plot gray samples first (so they appear in background)
        if len(gray_df) > 0:
            fig.add_trace(go.Scatter3d(
                x=gray_df['umap_x'],
                y=gray_df['umap_y'],
                z=gray_df['umap_z'],
                mode='markers',
                marker=dict(color='#808080', size=3, opacity=0.5),
                name='other',
                hovertemplate='<b>%{text}</b><br>UMAP1: %{x}<br>UMAP2: %{y}<br>UMAP3: %{z}<extra></extra>',
                text=gray_df.get('Var1', gray_df.index)
            ))
        
        # Plot colored samples on top
        for bbox_cat in ['bbox1', 'bbox2', 'bbox3', 'bbox4', 'bbox5', 'bbox6', 'bbox7']:
            mask = colored_df['bbox_color'] == bbox_cat
            if mask.any():
                subset = colored_df[mask]
                fig.add_trace(go.Scatter3d(
                    x=subset['umap_x'],
                    y=subset['umap_y'],
                    z=subset['umap_z'],
                    mode='markers',
                    marker=dict(color=bbox_colors[bbox_cat], size=4, opacity=0.8),
                    name=bbox_cat,
                    hovertemplate='<b>%{text}</b><br>UMAP1: %{x}<br>UMAP2: %{y}<br>UMAP3: %{z}<extra></extra>',
                    text=subset.get('Var1', subset.index)
                ))
        
        # Update layout for 3D
        fig.update_layout(
            scene=dict(
                xaxis_title='UMAP Dimension 1',
                yaxis_title='UMAP Dimension 2',
                zaxis_title='UMAP Dimension 3'
            ),
            title='3D UMAP Visualization - Merged Features (bbox1-7 colored, others gray)',
            legend=dict(
                title="Bounding Box",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            hovermode='closest'
        )
        
        filename_base = "umap3d_merged_bbox_colored"
        coord_cols = ['bbox', 'bbox_color', 'source', 'Var1', 'umap_x', 'umap_y', 'umap_z']
        
    else:  # 2D
        fig = go.Figure()
        
        # Plot gray samples first (so they appear in background)
        if len(gray_df) > 0:
            fig.add_trace(go.Scatter(
                x=gray_df['umap_x'],
                y=gray_df['umap_y'],
                mode='markers',
                marker=dict(color='#808080', size=5, opacity=0.5),
                name='other',
                hovertemplate='<b>%{text}</b><br>UMAP1: %{x}<br>UMAP2: %{y}<extra></extra>',
                text=gray_df.get('Var1', gray_df.index)
            ))
        
        # Plot colored samples on top
        for bbox_cat in ['bbox1', 'bbox2', 'bbox3', 'bbox4', 'bbox5', 'bbox6', 'bbox7']:
            mask = colored_df['bbox_color'] == bbox_cat
            if mask.any():
                subset = colored_df[mask]
                fig.add_trace(go.Scatter(
                    x=subset['umap_x'],
                    y=subset['umap_y'],
                    mode='markers',
                    marker=dict(color=bbox_colors[bbox_cat], size=6, opacity=0.8),
                    name=bbox_cat,
                    hovertemplate='<b>%{text}</b><br>UMAP1: %{x}<br>UMAP2: %{y}<extra></extra>',
                    text=subset.get('Var1', subset.index)
                ))
        
        # Update layout for 2D
        fig.update_layout(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            title='2D UMAP Visualization - Merged Features (bbox1-7 colored, others gray)',
            legend=dict(
                title="Bounding Box",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            hovermode='closest'
        )
        
        filename_base = "umap2d_merged_bbox_colored"
        coord_cols = ['bbox', 'bbox_color', 'source', 'Var1', 'umap_x', 'umap_y']
    
    # Save outputs based on format preference
    if output_format in ['html', 'both']:
        output_path_html = os.path.join(output_dir, f"{filename_base}.html")
        try:
            fig.write_html(output_path_html)
            print(f"Saved HTML: {output_path_html}")
        except Exception as e:
            print(f"Error saving HTML: {str(e)}")
    
    if output_format in ['png', 'both']:
        try:
            # Create matplotlib figure as alternative
            import matplotlib.pyplot as plt
            
            # Create figure
            if n_dimensions == 3:
                fig_mpl = plt.figure(figsize=(12, 10))
                ax = fig_mpl.add_subplot(111, projection='3d')
                
                # Plot gray samples first (background)
                if len(gray_df) > 0:
                    ax.scatter(
                        gray_df['umap_x'],
                        gray_df['umap_y'], 
                        gray_df['umap_z'],
                        c='#808080', label='other', alpha=0.5, s=10
                    )
                
                # Plot colored samples on top
                for bbox_cat in ['bbox1', 'bbox2', 'bbox3', 'bbox4', 'bbox5', 'bbox6', 'bbox7']:
                    mask = colored_df['bbox_color'] == bbox_cat
                    if mask.any():
                        subset = colored_df[mask]
                        ax.scatter(
                            subset['umap_x'],
                            subset['umap_y'], 
                            subset['umap_z'],
                            c=bbox_colors[bbox_cat], label=bbox_cat, alpha=0.8, s=20
                        )
                
                ax.set_xlabel('UMAP Dimension 1')
                ax.set_ylabel('UMAP Dimension 2')
                ax.set_zlabel('UMAP Dimension 3')
                ax.set_title('3D UMAP - Merged Features (bbox1-7 colored, others gray)')
                
            else:  # 2D
                fig_mpl, ax = plt.subplots(figsize=(12, 10))
                
                # Plot gray samples first (background)
                if len(gray_df) > 0:
                    ax.scatter(
                        gray_df['umap_x'],
                        gray_df['umap_y'],
                        c='#808080', label='other', alpha=0.5, s=15
                    )
                
                # Plot colored samples on top
                for bbox_cat in ['bbox1', 'bbox2', 'bbox3', 'bbox4', 'bbox5', 'bbox6', 'bbox7']:
                    mask = colored_df['bbox_color'] == bbox_cat
                    if mask.any():
                        subset = colored_df[mask]
                        ax.scatter(
                            subset['umap_x'],
                            subset['umap_y'],
                            c=bbox_colors[bbox_cat], label=bbox_cat, alpha=0.8, s=25
                        )
                
                ax.set_xlabel('UMAP Dimension 1')
                ax.set_ylabel('UMAP Dimension 2')
                ax.set_title('2D UMAP - Merged Features (bbox1-7 colored, others gray)')
            
            # Add legend
            ax.legend(title='Bounding Box', loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            
            # Save as PNG
            output_path_png = os.path.join(output_dir, f"{filename_base}.png")
            plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
            print(f"Saved PNG: {output_path_png}")
            
            # Also save as PDF
            output_path_pdf = os.path.join(output_dir, f"{filename_base}.pdf")
            plt.savefig(output_path_pdf, bbox_inches='tight')
            
            plt.close(fig_mpl)
        except Exception as e:
            print(f"Error creating matplotlib plot: {str(e)}")
    
    # Save UMAP coordinates as CSV for easy reuse
    if 'Var1' in features_df.columns:
        coord_cols = [col for col in coord_cols if col in features_df.columns]
    else:
        coord_cols = [col for col in coord_cols if col in features_df.columns and col != 'Var1']
    
    umap_coords_df = features_df[coord_cols]
    umap_coords_path = os.path.join(output_dir, f"{filename_base}_coordinates.csv")
    umap_coords_df.to_csv(umap_coords_path, index=False)
    print(f"Saved coordinates: {umap_coords_path}")
    
    return features_df, umap_results

def sample_points_from_umap(features_df, n_samples=100, method='spatial_grid'):
    """
    Sample points from UMAP space ensuring good spatial distribution
    
    Args:
        features_df: DataFrame with UMAP coordinates (must have 'umap_x', 'umap_y', and optionally 'umap_z')
        n_samples: Number of samples to select (default: 100)
        method: Sampling method - 'spatial_grid', 'random', or 'density_aware'
    
    Returns:
        pd.DataFrame: Subset of original dataframe with sampled points
        np.array: Boolean mask indicating which points were sampled
    """
    if 'umap_x' not in features_df.columns or 'umap_y' not in features_df.columns:
        raise ValueError("DataFrame must contain 'umap_x' and 'umap_y' columns")
    
    # Check if we have 3D coordinates
    has_3d = 'umap_z' in features_df.columns
    
    # Limit n_samples to available data
    n_samples = min(n_samples, len(features_df))
    
    print(f"Sampling {n_samples} points from {len(features_df)} total points using {method} method")
    
    if method == 'random':
        # Simple random sampling
        sampled_indices = np.random.choice(len(features_df), size=n_samples, replace=False)
        
    elif method == 'spatial_grid':
        # Grid-based sampling for better spatial distribution
        if has_3d:
            # 3D grid sampling
            x_coords = features_df['umap_x'].values
            y_coords = features_df['umap_y'].values
            z_coords = features_df['umap_z'].values
            
            # Create 3D grid
            grid_size = int(np.ceil(n_samples ** (1/3)))  # Cube root for 3D
            
            x_bins = np.linspace(x_coords.min(), x_coords.max(), grid_size + 1)
            y_bins = np.linspace(y_coords.min(), y_coords.max(), grid_size + 1)
            z_bins = np.linspace(z_coords.min(), z_coords.max(), grid_size + 1)
            
            sampled_indices = []
            samples_per_cell = max(1, n_samples // (grid_size ** 3))
            
            for i in range(grid_size):
                for j in range(grid_size):
                    for k in range(grid_size):
                        # Find points in this grid cell
                        mask = ((x_coords >= x_bins[i]) & (x_coords < x_bins[i+1]) &
                               (y_coords >= y_bins[j]) & (y_coords < y_bins[j+1]) &
                               (z_coords >= z_bins[k]) & (z_coords < z_bins[k+1]))
                        
                        cell_indices = np.where(mask)[0]
                        if len(cell_indices) > 0:
                            # Sample from this cell
                            n_from_cell = min(samples_per_cell, len(cell_indices))
                            selected = np.random.choice(cell_indices, size=n_from_cell, replace=False)
                            sampled_indices.extend(selected)
        else:
            # 2D grid sampling
            x_coords = features_df['umap_x'].values
            y_coords = features_df['umap_y'].values
            
            # Create 2D grid
            grid_size = int(np.ceil(np.sqrt(n_samples)))  # Square root for 2D
            
            x_bins = np.linspace(x_coords.min(), x_coords.max(), grid_size + 1)
            y_bins = np.linspace(y_coords.min(), y_coords.max(), grid_size + 1)
            
            sampled_indices = []
            samples_per_cell = max(1, n_samples // (grid_size ** 2))
            
            for i in range(grid_size):
                for j in range(grid_size):
                    # Find points in this grid cell
                    mask = ((x_coords >= x_bins[i]) & (x_coords < x_bins[i+1]) &
                           (y_coords >= y_bins[j]) & (y_coords < y_bins[j+1]))
                    
                    cell_indices = np.where(mask)[0]
                    if len(cell_indices) > 0:
                        # Sample from this cell
                        n_from_cell = min(samples_per_cell, len(cell_indices))
                        selected = np.random.choice(cell_indices, size=n_from_cell, replace=False)
                        sampled_indices.extend(selected)
        
        # If we have too few samples, fill with random sampling
        sampled_indices = list(set(sampled_indices))  # Remove duplicates
        if len(sampled_indices) < n_samples:
            remaining_indices = [i for i in range(len(features_df)) if i not in sampled_indices]
            additional_needed = n_samples - len(sampled_indices)
            if len(remaining_indices) >= additional_needed:
                additional = np.random.choice(remaining_indices, size=additional_needed, replace=False)
                sampled_indices.extend(additional)
        
        # If we have too many, randomly select subset
        if len(sampled_indices) > n_samples:
            sampled_indices = np.random.choice(sampled_indices, size=n_samples, replace=False)
            
    elif method == 'density_aware':
        # Sample points with probability inversely proportional to local density
        if has_3d:
            coords = features_df[['umap_x', 'umap_y', 'umap_z']].values
        else:
            coords = features_df[['umap_x', 'umap_y']].values
        
        # Compute local density using k-nearest neighbors
        from sklearn.neighbors import NearestNeighbors
        k = min(20, len(features_df) // 10)  # Adaptive k based on dataset size
        
        nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
        distances, _ = nbrs.kneighbors(coords)
        
        # Use mean distance to k-th neighbor as density measure (inverse)
        density_scores = distances[:, -1]  # Distance to k-th neighbor
        
        # Convert to probabilities (higher distance = lower density = higher probability)
        probabilities = density_scores / density_scores.sum()
        
        # Sample based on these probabilities
        sampled_indices = np.random.choice(
            len(features_df), 
            size=n_samples, 
            replace=False, 
            p=probabilities
        )
    
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    # Create boolean mask
    mask = np.zeros(len(features_df), dtype=bool)
    mask[sampled_indices] = True
    
    # Create sampled dataframe
    sampled_df = features_df.iloc[sampled_indices].copy()
    sampled_df['is_sampled'] = True
    
    print(f"Successfully sampled {len(sampled_df)} points")
    
    return sampled_df, mask

def create_umap_with_sampled_crosses(features_df, sampled_mask, output_dir, 
                                   n_dimensions=2, output_format='both', 
                                   cross_size=100, cross_color='red'):
    """
    Create UMAP visualization with sampled points marked as crosses
    
    Args:
        features_df: DataFrame with UMAP coordinates and bbox information
        sampled_mask: Boolean mask indicating which points are sampled
        output_dir: Directory to save outputs
        n_dimensions: Number of dimensions (2 or 3)
        output_format: Output format - 'html', 'png', or 'both'
        cross_size: Size of the cross markers
        cross_color: Color of the cross markers
    """
    if 'umap_x' not in features_df.columns or 'umap_y' not in features_df.columns:
        raise ValueError("DataFrame must contain UMAP coordinates")
    
    # Validate parameters
    if n_dimensions not in [2, 3]:
        raise ValueError("n_dimensions must be 2 or 3")
    if output_format not in ['html', 'png', 'both']:
        raise ValueError("output_format must be 'html', 'png', or 'both'")
    
    has_3d = 'umap_z' in features_df.columns and n_dimensions == 3
    
    # Define colors for bbox categories
    bbox_colors = {
        'bbox1': '#FF0000',  # Red
        'bbox2': '#00FF00',  # Green
        'bbox3': '#0000FF',  # Blue
        'bbox4': '#FFA500',  # Orange
        'bbox5': '#800080',  # Purple
        'bbox6': '#00FFFF',  # Cyan
        'bbox7': '#FFD700'   # Gold
    }
    
    # Prepare data
    features_df = features_df.copy()
    
    # Create bbox color categories
    def categorize_bbox(bbox_val):
        bbox_str = str(bbox_val).lower()
        if bbox_str in ['bbox1', 'bbox2', 'bbox3', 'bbox4', 'bbox5', 'bbox6', 'bbox7']:
            return bbox_str
        else:
            return 'other'
    
    if 'bbox' in features_df.columns:
        features_df['bbox_color'] = features_df['bbox'].apply(categorize_bbox)
    else:
        features_df['bbox_color'] = 'other'
    
    bbox_colors_extended = bbox_colors.copy()
    bbox_colors_extended['other'] = '#808080'  # Gray
    
    # Separate sampled and non-sampled points
    sampled_df = features_df[sampled_mask].copy()
    non_sampled_df = features_df[~sampled_mask].copy()
    
    print(f"Plotting {len(non_sampled_df)} background points and {len(sampled_df)} sampled crosses")
    
    # Create plotly figure
    if has_3d:
        fig = go.Figure()
        
        # Plot non-sampled points first (background)
        for bbox_cat in ['other'] + ['bbox1', 'bbox2', 'bbox3', 'bbox4', 'bbox5', 'bbox6', 'bbox7']:
            mask = non_sampled_df['bbox_color'] == bbox_cat
            if mask.any():
                subset = non_sampled_df[mask]
                opacity = 0.3 if bbox_cat == 'other' else 0.6
                size = 3 if bbox_cat == 'other' else 4
                fig.add_trace(go.Scatter3d(
                    x=subset['umap_x'],
                    y=subset['umap_y'],
                    z=subset['umap_z'],
                    mode='markers',
                    marker=dict(color=bbox_colors_extended[bbox_cat], size=size, opacity=opacity),
                    name=f'{bbox_cat} (background)',
                    showlegend=True,
                    hovertemplate='<b>%{text}</b><br>UMAP1: %{x}<br>UMAP2: %{y}<br>UMAP3: %{z}<extra></extra>',
                    text=subset.get('Var1', subset.index)
                ))
        
        # Plot sampled points as crosses on top
        if len(sampled_df) > 0:
            fig.add_trace(go.Scatter3d(
                x=sampled_df['umap_x'],
                y=sampled_df['umap_y'],
                z=sampled_df['umap_z'],
                mode='markers',
                marker=dict(
                    color=cross_color,
                    size=cross_size//10,  # Scale for 3D
                    symbol='x',
                    opacity=0.9,
                    line=dict(width=2)
                ),
                name=f'Sampled ({len(sampled_df)} points)',
                showlegend=True,
                hovertemplate='<b>SAMPLED: %{text}</b><br>UMAP1: %{x}<br>UMAP2: %{y}<br>UMAP3: %{z}<extra></extra>',
                text=sampled_df.get('Var1', sampled_df.index)
            ))
        
        # Update layout for 3D
        fig.update_layout(
            scene=dict(
                xaxis_title='UMAP Dimension 1',
                yaxis_title='UMAP Dimension 2',
                zaxis_title='UMAP Dimension 3'
            ),
            title=f'3D UMAP with {len(sampled_df)} Randomly Sampled Points (crosses)',
            legend=dict(title="Categories", yanchor="top", y=0.99, xanchor="right", x=0.99),
            margin=dict(l=0, r=0, b=0, t=50),
            hovermode='closest'
        )
        
        filename_base = f"umap3d_with_sampled_crosses_{len(sampled_df)}"
    
    else:  # 2D
        fig = go.Figure()
        
        # Plot non-sampled points first (background)
        for bbox_cat in ['other'] + ['bbox1', 'bbox2', 'bbox3', 'bbox4', 'bbox5', 'bbox6', 'bbox7']:
            mask = non_sampled_df['bbox_color'] == bbox_cat
            if mask.any():
                subset = non_sampled_df[mask]
                opacity = 0.4 if bbox_cat == 'other' else 0.7
                size = 4 if bbox_cat == 'other' else 5
                fig.add_trace(go.Scatter(
                    x=subset['umap_x'],
                    y=subset['umap_y'],
                    mode='markers',
                    marker=dict(color=bbox_colors_extended[bbox_cat], size=size, opacity=opacity),
                    name=f'{bbox_cat} (background)',
                    showlegend=True,
                    hovertemplate='<b>%{text}</b><br>UMAP1: %{x}<br>UMAP2: %{y}<extra></extra>',
                    text=subset.get('Var1', subset.index)
                ))
        
        # Plot sampled points as crosses on top
        if len(sampled_df) > 0:
            fig.add_trace(go.Scatter(
                x=sampled_df['umap_x'],
                y=sampled_df['umap_y'],
                mode='markers',
                marker=dict(
                    color=cross_color,
                    size=cross_size//5,  # Scale for 2D
                    symbol='x',
                    opacity=0.9,
                    line=dict(width=3)
                ),
                name=f'Sampled ({len(sampled_df)} points)',
                showlegend=True,
                hovertemplate='<b>SAMPLED: %{text}</b><br>UMAP1: %{x}<br>UMAP2: %{y}<extra></extra>',
                text=sampled_df.get('Var1', sampled_df.index)
            ))
        
        # Update layout for 2D
        fig.update_layout(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            title=f'2D UMAP with {len(sampled_df)} Randomly Sampled Points (crosses)',
            legend=dict(title="Categories", yanchor="top", y=0.99, xanchor="right", x=0.99),
            margin=dict(l=0, r=0, b=0, t=50),
            hovermode='closest'
        )
        
        filename_base = f"umap2d_with_sampled_crosses_{len(sampled_df)}"
    
    # Save outputs
    if output_format in ['html', 'both']:
        output_path_html = os.path.join(output_dir, f"{filename_base}.html")
        try:
            fig.write_html(output_path_html)
            print(f"Saved HTML: {output_path_html}")
        except Exception as e:
            print(f"Error saving HTML: {str(e)}")
    
    if output_format in ['png', 'both']:
        try:
            # Create matplotlib figure
            import matplotlib.pyplot as plt
            
            if has_3d:
                fig_mpl = plt.figure(figsize=(14, 10))
                ax = fig_mpl.add_subplot(111, projection='3d')
                
                # Plot background points
                for bbox_cat in ['other'] + ['bbox1', 'bbox2', 'bbox3', 'bbox4', 'bbox5', 'bbox6', 'bbox7']:
                    mask = non_sampled_df['bbox_color'] == bbox_cat
                    if mask.any():
                        subset = non_sampled_df[mask]
                        alpha = 0.3 if bbox_cat == 'other' else 0.6
                        size = 8 if bbox_cat == 'other' else 12
                        ax.scatter(
                            subset['umap_x'], subset['umap_y'], subset['umap_z'],
                            c=bbox_colors_extended[bbox_cat], 
                            label=f'{bbox_cat} (background)',
                            alpha=alpha, s=size
                        )
                
                # Plot sampled points as crosses
                if len(sampled_df) > 0:
                    ax.scatter(
                        sampled_df['umap_x'], sampled_df['umap_y'], sampled_df['umap_z'],
                        c=cross_color, marker='x', s=cross_size, 
                        label=f'Sampled ({len(sampled_df)} points)', 
                        alpha=0.9, linewidths=2
                    )
                
                ax.set_xlabel('UMAP Dimension 1')
                ax.set_ylabel('UMAP Dimension 2')
                ax.set_zlabel('UMAP Dimension 3')
                ax.set_title(f'3D UMAP with {len(sampled_df)} Randomly Sampled Points (crosses)')
                
            else:  # 2D
                fig_mpl, ax = plt.subplots(figsize=(14, 10))
                
                # Plot background points
                for bbox_cat in ['other'] + ['bbox1', 'bbox2', 'bbox3', 'bbox4', 'bbox5', 'bbox6', 'bbox7']:
                    mask = non_sampled_df['bbox_color'] == bbox_cat
                    if mask.any():
                        subset = non_sampled_df[mask]
                        alpha = 0.4 if bbox_cat == 'other' else 0.7
                        size = 15 if bbox_cat == 'other' else 20
                        ax.scatter(
                            subset['umap_x'], subset['umap_y'],
                            c=bbox_colors_extended[bbox_cat], 
                            label=f'{bbox_cat} (background)',
                            alpha=alpha, s=size
                        )
                
                # Plot sampled points as crosses
                if len(sampled_df) > 0:
                    ax.scatter(
                        sampled_df['umap_x'], sampled_df['umap_y'],
                        c=cross_color, marker='x', s=cross_size, 
                        label=f'Sampled ({len(sampled_df)} points)', 
                        alpha=0.9, linewidths=3
                    )
                
                ax.set_xlabel('UMAP Dimension 1')
                ax.set_ylabel('UMAP Dimension 2')
                ax.set_title(f'2D UMAP with {len(sampled_df)} Randomly Sampled Points (crosses)')
            
            # Add legend
            ax.legend(title='Categories', loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            
            # Save as PNG and PDF
            output_path_png = os.path.join(output_dir, f"{filename_base}.png")
            plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
            print(f"Saved PNG: {output_path_png}")
            
            output_path_pdf = os.path.join(output_dir, f"{filename_base}.pdf")
            plt.savefig(output_path_pdf, bbox_inches='tight')
            
            plt.close(fig_mpl)
            
        except Exception as e:
            print(f"Error creating matplotlib plot: {str(e)}")
    
    # Save sampled points information
    sampled_info_path = os.path.join(output_dir, f"sampled_points_{len(sampled_df)}.csv")
    sampled_df.to_csv(sampled_info_path, index=False)
    print(f"Saved sampled points info: {sampled_info_path}")
    
    return sampled_df

def create_umap_heatmaps(features_df, output_dir, n_dimensions=2, 
                        cmaps=['viridis', 'plasma', 'hot', 'coolwarm', 'Blues'],
                        bin_size=50, smooth_sigma=1.0):
    """
    Create heatmap visualizations of UMAP datapoint density with different color maps
    
    Args:
        features_df: DataFrame with UMAP coordinates (must have 'umap_x', 'umap_y', and optionally 'umap_z')
        output_dir: Directory to save heatmap outputs
        n_dimensions: Number of dimensions (2 or 3)
        cmaps: List of matplotlib colormap names to use
        bin_size: Number of bins for histogram (higher = more detailed)
        smooth_sigma: Gaussian smoothing sigma (higher = smoother heatmap)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import gaussian_filter
    import seaborn as sns
    
    if 'umap_x' not in features_df.columns or 'umap_y' not in features_df.columns:
        raise ValueError("DataFrame must contain UMAP coordinates")
    
    print(f"Creating UMAP heatmaps with {len(features_df)} points...")
    
    # Extract coordinates
    x_coords = features_df['umap_x'].values
    y_coords = features_df['umap_y'].values
    
    has_3d = 'umap_z' in features_df.columns and n_dimensions == 3
    if has_3d:
        z_coords = features_df['umap_z'].values
    
    # Create output subdirectory for heatmaps
    heatmap_dir = os.path.join(output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    
    if has_3d:
        # 3D heatmap approach - create multiple 2D slices
        print("Creating 3D heatmap slices...")
        
        # Find coordinate ranges
        x_range = (x_coords.min(), x_coords.max())
        y_range = (y_coords.min(), y_coords.max())
        z_range = (z_coords.min(), z_coords.max())
        
        # Create slices along Z-axis
        n_slices = 5
        z_slice_edges = np.linspace(z_range[0], z_range[1], n_slices + 1)
        
        for cmap_name in cmaps:
            print(f"  Creating 3D slices with {cmap_name} colormap...")
            
            fig, axes = plt.subplots(1, n_slices, figsize=(20, 4))
            if n_slices == 1:
                axes = [axes]
            
            for i in range(n_slices):
                z_min, z_max = z_slice_edges[i], z_slice_edges[i + 1]
                
                # Filter points in this Z slice
                z_mask = (z_coords >= z_min) & (z_coords <= z_max)
                slice_x = x_coords[z_mask]
                slice_y = y_coords[z_mask]
                
                if len(slice_x) > 0:
                    # Create 2D histogram
                    hist, x_edges, y_edges = np.histogram2d(
                        slice_x, slice_y, bins=bin_size,
                        range=[x_range, y_range]
                    )
                    
                    # Apply Gaussian smoothing
                    if smooth_sigma > 0:
                        hist = gaussian_filter(hist, sigma=smooth_sigma)
                    
                    # Plot heatmap
                    im = axes[i].imshow(
                        hist.T, origin='lower', cmap=cmap_name, aspect='auto',
                        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                        interpolation='bilinear'
                    )
                    
                    axes[i].set_title(f'Z slice: [{z_min:.2f}, {z_max:.2f}]\n({len(slice_x)} points)')
                    axes[i].set_xlabel('UMAP Dimension 1')
                    if i == 0:
                        axes[i].set_ylabel('UMAP Dimension 2')
                    
                    # Add colorbar
                    plt.colorbar(im, ax=axes[i], label='Density')
                else:
                    axes[i].text(0.5, 0.5, 'No points\nin slice', 
                               transform=axes[i].transAxes, ha='center', va='center')
                    axes[i].set_title(f'Z slice: [{z_min:.2f}, {z_max:.2f}]')
            
            plt.suptitle(f'3D UMAP Heatmap Slices - {cmap_name.capitalize()} Colormap')
            plt.tight_layout()
            
            # Save plot
            output_path = os.path.join(heatmap_dir, f"heatmap_3d_slices_{cmap_name}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            output_path_pdf = os.path.join(heatmap_dir, f"heatmap_3d_slices_{cmap_name}.pdf")
            plt.savefig(output_path_pdf, bbox_inches='tight')
            
            plt.close()
            print(f"    Saved: heatmap_3d_slices_{cmap_name}.png")
    
    else:
        # 2D heatmaps
        print("Creating 2D heatmaps...")
        
        # Find coordinate ranges
        x_range = (x_coords.min(), x_coords.max())
        y_range = (y_coords.min(), y_coords.max())
        
        # Create 2D histogram
        hist, x_edges, y_edges = np.histogram2d(
            x_coords, y_coords, bins=bin_size,
            range=[x_range, y_range]
        )
        
        # Apply Gaussian smoothing
        if smooth_sigma > 0:
            hist_smooth = gaussian_filter(hist, sigma=smooth_sigma)
        else:
            hist_smooth = hist
        
        # Create heatmaps with different colormaps
        for cmap_name in cmaps:
            print(f"  Creating heatmap with {cmap_name} colormap...")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Raw histogram (left)
            im1 = ax1.imshow(
                hist.T, origin='lower', cmap=cmap_name, aspect='auto',
                extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                interpolation='nearest'
            )
            ax1.set_title(f'Raw Density Heatmap\n({len(features_df)} points)')
            ax1.set_xlabel('UMAP Dimension 1')
            ax1.set_ylabel('UMAP Dimension 2')
            plt.colorbar(im1, ax=ax1, label='Point Count')
            
            # Smoothed histogram (right)
            im2 = ax2.imshow(
                hist_smooth.T, origin='lower', cmap=cmap_name, aspect='auto',
                extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                interpolation='bilinear'
            )
            ax2.set_title(f'Smoothed Density Heatmap\n(σ={smooth_sigma})')
            ax2.set_xlabel('UMAP Dimension 1')
            ax2.set_ylabel('UMAP Dimension 2')
            plt.colorbar(im2, ax=ax2, label='Density')
            
            plt.suptitle(f'2D UMAP Heatmap - {cmap_name.capitalize()} Colormap')
            plt.tight_layout()
            
            # Save plot
            output_path = os.path.join(heatmap_dir, f"heatmap_2d_{cmap_name}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            output_path_pdf = os.path.join(heatmap_dir, f"heatmap_2d_{cmap_name}.pdf")
            plt.savefig(output_path_pdf, bbox_inches='tight')
            
            plt.close()
            print(f"    Saved: heatmap_2d_{cmap_name}.png")
    
    # Create a single plot comparing all colormaps
    print("Creating colormap comparison plot...")
    
    if has_3d:
        # For 3D, use the middle Z slice for comparison
        z_mid = (z_coords.min() + z_coords.max()) / 2
        z_range_width = (z_coords.max() - z_coords.min()) / 5  # Use same slice width as above
        z_mask = (z_coords >= z_mid - z_range_width/2) & (z_coords <= z_mid + z_range_width/2)
        plot_x = x_coords[z_mask]
        plot_y = y_coords[z_mask]
        
        if len(plot_x) > 0:
            plot_range = [(plot_x.min(), plot_x.max()), (plot_y.min(), plot_y.max())]
            hist_compare, _, _ = np.histogram2d(plot_x, plot_y, bins=bin_size//2, range=plot_range)
        else:
            hist_compare = np.zeros((bin_size//2, bin_size//2))
            plot_range = [x_range, y_range]
    else:
        hist_compare = hist
        plot_range = [x_range, y_range]
    
    if smooth_sigma > 0:
        hist_compare = gaussian_filter(hist_compare, sigma=smooth_sigma)
    
    # Create comparison plot
    n_cmaps = len(cmaps)
    cols = min(3, n_cmaps)
    rows = (n_cmaps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, cmap_name in enumerate(cmaps):
        if i < len(axes):
            im = axes[i].imshow(
                hist_compare.T, origin='lower', cmap=cmap_name, aspect='auto',
                extent=[plot_range[0][0], plot_range[0][1], plot_range[1][0], plot_range[1][1]],
                interpolation='bilinear'
            )
            axes[i].set_title(f'{cmap_name.capitalize()}')
            axes[i].set_xlabel('UMAP Dimension 1')
            axes[i].set_ylabel('UMAP Dimension 2')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i], label='Density')
    
    # Hide unused subplots
    for i in range(len(cmaps), len(axes)):
        axes[i].set_visible(False)
    
    dimension_text = "3D (middle slice)" if has_3d else "2D"
    plt.suptitle(f'UMAP Heatmap Colormap Comparison - {dimension_text}')
    plt.tight_layout()
    
    # Save comparison plot
    output_path = os.path.join(heatmap_dir, "heatmap_colormap_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    output_path_pdf = os.path.join(heatmap_dir, "heatmap_colormap_comparison.pdf")
    plt.savefig(output_path_pdf, bbox_inches='tight')
    
    plt.close()
    print(f"Saved colormap comparison: heatmap_colormap_comparison.png")
    
    # Save heatmap data
    if not has_3d:
        heatmap_data = {
            'histogram': hist,
            'histogram_smoothed': hist_smooth,
            'x_edges': x_edges,
            'y_edges': y_edges,
            'x_range': x_range,
            'y_range': y_range
        }
        
        np.savez(os.path.join(heatmap_dir, "heatmap_data.npz"), **heatmap_data)
        print("Saved heatmap data: heatmap_data.npz")
    
    print(f"All heatmaps saved to: {heatmap_dir}")
    return heatmap_dir

if __name__ == "__main__":
    # Configuration
    N_DIMENSIONS = 2  # Choose 2 or 3 for 2D or 3D visualization
    OUTPUT_FORMAT = 'both'  # Choose 'html', 'png', or 'both'
    
    # Sampling configuration
    SAMPLE_POINTS = True  # Set to True to enable point sampling
    N_SAMPLES = 100  # Number of points to sample
    SAMPLING_METHOD = 'spatial_grid'  # Choose 'spatial_grid', 'random', or 'density_aware'

    # Heatmap configuration
    CREATE_HEATMAPS = True  # Set to True to create density heatmaps
    HEATMAP_CMAPS = ['viridis', 'plasma', 'hot', 'coolwarm', 'Blues', 'Reds', 'inferno', 'magma']
    HEATMAP_BIN_SIZE = 50  # Higher = more detailed heatmap
    HEATMAP_SMOOTH_SIGMA = 1.0  # Higher = smoother heatmap

    # Paths to the two CSV files
    csv1_path = "features_20250602_174807.csv"
    csv2_path = "raven_results/features_avg_seg11_alpha1_0.csv"
    
    # Merge the datasets
    print("Merging feature datasets...")
    merged_df = merge_feature_datasets(csv1_path, csv2_path)
    
    # Save merged dataset
    merged_output_path = "merged_features.csv"
    merged_df.to_csv(merged_output_path, index=False)
    print(f"Saved merged dataset: {merged_output_path}")

    # Create output directory
    output_dir = "umap_results"
    os.makedirs(output_dir, exist_ok=True)

    # Create the bounding box colored UMAP visualization
    print("Creating UMAP visualization...")
    df_with_umap, umap_results = create_bbox_colored_umap(
        merged_df, 
        output_dir, 
        n_dimensions=N_DIMENSIONS, 
        output_format=OUTPUT_FORMAT
    )
    
    # Create heatmaps if enabled
    if CREATE_HEATMAPS:
        print(f"\nCreating UMAP density heatmaps...")
        heatmap_dir = create_umap_heatmaps(
            df_with_umap,
            output_dir,
            n_dimensions=N_DIMENSIONS,
            cmaps=HEATMAP_CMAPS,
            bin_size=HEATMAP_BIN_SIZE,
            smooth_sigma=HEATMAP_SMOOTH_SIGMA
        )
        print(f"Heatmaps saved to: {heatmap_dir}")
    
    # Sample points and create visualization with crosses if enabled
    if SAMPLE_POINTS:
        print(f"\nSampling {N_SAMPLES} points from UMAP space...")
        
        # Sample points from the UMAP space
        sampled_df, sampled_mask = sample_points_from_umap(
            df_with_umap, 
            n_samples=N_SAMPLES, 
            method=SAMPLING_METHOD
        )
        
        print(f"Sampled DataFrame shape: {sampled_df.shape}")
        print(f"Sampled points bbox distribution:")
        if 'bbox' in sampled_df.columns:
            print(sampled_df['bbox'].value_counts())
        
        # Create UMAP visualization with sampled points marked as crosses
        print("Creating UMAP with sampled crosses...")
        sampled_crosses_df = create_umap_with_sampled_crosses(
            df_with_umap,
            sampled_mask,
            output_dir,
            n_dimensions=N_DIMENSIONS,
            output_format=OUTPUT_FORMAT,
            cross_size=100,
            cross_color='red'
        )
        
        print(f"\nSampling complete!")
        print(f"- Total points: {len(df_with_umap)}")
        print(f"- Sampled points: {len(sampled_df)}")
        print(f"- Sampling method: {SAMPLING_METHOD}")
        print(f"- Files saved in: {output_dir}")
    
    print("Analysis complete!")

