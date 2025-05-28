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
        features_df: DataFrame with features and bbox_name column
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
    
    # Define a consistent color map for bounding boxes
    bbox_colors = {
        'bbox1': '#FF0000', 'bbox2': '#00FFFF', 'bbox3': '#FFA500',
        'bbox4': '#800080', 'bbox5': '#808080', 'bbox6': '#0000FF', 'bbox7': '#000000'
    }
    
    # Debug: Print column names to understand their structure
    
    # Preprocess bbox_name column to handle invalid formats
    features_df = features_df.copy()
    valid_bbox_pattern = r'^bbox[1-7]$'
    
    # Check which bbox names are valid
    import re
    valid_mask = features_df['bbox'].astype(str).str.match(valid_bbox_pattern, na=False)
    invalid_count = (~valid_mask).sum()
    valid_count = valid_mask.sum()
    
    # Create a new column for coloring - replace invalid bbox names with 'other'
    features_df['bbox_color'] = features_df['bbox'].where(valid_mask, 'other')
    
    # Add gray color for 'other' category
    bbox_colors_extended = bbox_colors.copy()
    bbox_colors_extended['other'] = '#808080'  # Gray color for invalid bbox names
    
    # Print unique bbox values for debugging
    unique_bboxes = features_df['bbox_color'].unique()
    
    # Simple feature column detection - check if 'feat_' appears anywhere in the column name
    feature_cols = [col for col in features_df.columns if 'feat_' in col]
    
    if not feature_cols:
        raise ValueError("No feature columns found in DataFrame")
    
    
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
    
    
    # Create plotly figure based on dimensions
    if n_dimensions == 3:
        fig = px.scatter_3d(
            features_df,
            x='umap_x',
            y='umap_y',
            z='umap_z',
            color='bbox_color',
            color_discrete_map=bbox_colors_extended,
            hover_data=['Var1'],  # Display synapse ID in hover
            title='3D UMAP Visualization Colored by Bounding Box',
            opacity=0.8
        )
        
        # Update layout for 3D
        fig.update_layout(
            scene=dict(
                xaxis_title='UMAP Dimension 1',
                yaxis_title='UMAP Dimension 2',
                zaxis_title='UMAP Dimension 3'
            ),
            legend=dict(
                title="Bounding Box",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            hovermode='closest'
        )
        
        filename_base = "umap3d_bbox_colored"
        coord_cols = ['bbox', 'Var1', 'umap_x', 'umap_y', 'umap_z']
    else:  # 2D
        fig = px.scatter(
            features_df,
            x='umap_x',
            y='umap_y',
            color='bbox_color',
            color_discrete_map=bbox_colors_extended,
            hover_data=['Var1'],  # Display synapse ID in hover
            title='2D UMAP Visualization Colored by Bounding Box',
            opacity=0.8
        )
        
        # Update layout for 2D
        fig.update_layout(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            legend=dict(
                title="Bounding Box",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            hovermode='closest'
        )
        
        filename_base = "umap2d_bbox_colored"
        coord_cols = ['bbox', 'Var1', 'umap_x', 'umap_y']
    
    # Update marker size and appearance
    fig.update_traces(marker=dict(size=1, line=dict(width=0.2, color='black')))
    
    # Save outputs based on format preference
    
    if output_format in ['html', 'both']:
        output_path_html = os.path.join(output_dir, f"{filename_base}.html")
        try:
            fig.write_html(output_path_html)
        except Exception as e:
            print(f"Error saving HTML: {str(e)}")
            
            try:
                # Alternative: Save as a JSON file
                import json
                fig_json = fig.to_json()
                json_path = os.path.join(output_dir, f"{filename_base}.json")
                with open(json_path, 'w') as f:
                    f.write(fig_json)
            except Exception as e2:
                print(f"Error saving JSON: {str(e2)}")
    
    if output_format in ['png', 'both']:
        try:
            # Create matplotlib figure as alternative to kaleido
            import matplotlib.pyplot as plt
            
            # Create figure
            if n_dimensions == 3:
                fig_mpl = plt.figure(figsize=(12, 10))
                ax = fig_mpl.add_subplot(111, projection='3d')
                
                # Plot each bbox category separately to control colors
                for bbox_cat in unique_bboxes:
                    mask = features_df['bbox_color'] == bbox_cat
                    if mask.any():
                        color = bbox_colors_extended.get(bbox_cat, '#808080')
                        ax.scatter(
                            features_df.loc[mask, 'umap_x'],
                            features_df.loc[mask, 'umap_y'], 
                            features_df.loc[mask, 'umap_z'],
                            c=color, label=bbox_cat, alpha=0.8, s=5
                        )
                
                ax.set_xlabel('UMAP Dimension 1')
                ax.set_ylabel('UMAP Dimension 2')
                ax.set_zlabel('UMAP Dimension 3')
                ax.set_title('3D UMAP Visualization Colored by Bounding Box')
                
            else:  # 2D
                fig_mpl, ax = plt.subplots(figsize=(12, 10))
                
                # Plot each bbox category separately to control colors
                for bbox_cat in unique_bboxes:
                    mask = features_df['bbox_color'] == bbox_cat
                    if mask.any():
                        color = bbox_colors_extended.get(bbox_cat, '#808080')
                        ax.scatter(
                            features_df.loc[mask, 'umap_x'],
                            features_df.loc[mask, 'umap_y'],
                            c=color, label=bbox_cat, alpha=0.8, s=15
                        )
                
                ax.set_xlabel('UMAP Dimension 1')
                ax.set_ylabel('UMAP Dimension 2')
                ax.set_title('2D UMAP Visualization Colored by Bounding Box')
            
            # Add legend with better positioning and debugging
            if len(unique_bboxes) > 0:
                legend = ax.legend(title='Bounding Box', loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            
            # Save as PNG
            output_path_png = os.path.join(output_dir, f"{filename_base}.png")
            plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
            
            # Also save as PDF (vector format alternative to SVG)
            output_path_pdf = os.path.join(output_dir, f"{filename_base}.pdf")
            plt.savefig(output_path_pdf, bbox_inches='tight')
            
            plt.close(fig_mpl)
        except Exception as e:
            print(f"Error creating matplotlib plot: {str(e)}")
    
    
    # Save UMAP coordinates as CSV for easy reuse
    umap_coords_df = features_df[coord_cols]
    umap_coords_path = os.path.join(output_dir, f"{filename_base}_coordinates.csv")
    umap_coords_df.to_csv(umap_coords_path, index=False)
    
    return features_df, umap_results



N_DIMENSIONS = 3 # Choose 2 or 3 for 2D or 3D visualization
OUTPUT_FORMAT = 'both'  # Choose 'html', 'png', or 'both'

# Load feature data
df = pd.read_csv(r"merged.csv")

# Create output directory
output_dir = r"run_20250518_200339/umap"
os.makedirs(output_dir, exist_ok=True)

# Create the bounding box colored UMAP
df_with_umap, umap_results = create_bbox_colored_umap(df, output_dir, n_dimensions=N_DIMENSIONS, output_format=OUTPUT_FORMAT)

