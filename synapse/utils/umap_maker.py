import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
from tqdm import tqdm  
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def merge_feature_datasets(csv1_path, csv2_path):
    df1 = pd.read_csv(csv1_path)
    
    df2 = pd.read_csv(csv2_path)
    df1['source'] = 'dataset1'
    df2['source'] = 'dataset2'
    common_cols = set(df1.columns) & set(df2.columns)
    df1_only_cols = set(df1.columns) - common_cols
    df2_only_cols = set(df2.columns) - common_cols    
    for col in df2_only_cols:
        if col != 'source':
            df1[col] = np.nan
    for col in df1_only_cols:
        if col != 'source':
            df2[col] = np.nan
    merged_df = pd.concat([df1, df2], ignore_index=True, sort=False)
    return merged_df

def compute_umap(features_scaled, n_components=3, n_neighbors=15, min_dist=0.1, metric='euclidean'):
    start_time = time.time()
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
        verbose=True  
    )
    umap_results = reducer.fit_transform(features_scaled)
    elapsed_time = time.time() - start_time
    return umap_results, reducer

def create_bbox_colored_umap(features_df, output_dir, reuse_umap_results=None, n_dimensions=3, output_format='both'):
    """Create a 2-D or 3-D UMAP coloured by bounding-box ID and save as HTML/PNG.

    Parameters
    ----------
    features_df : pd.DataFrame
        Must contain a column ``bbox`` and feature columns starting with ``feat_``.
    output_dir : str
        Directory where the plot(s) and coordinate CSV will be written.
    reuse_umap_results : np.ndarray | None
        Pre-computed UMAP coordinates to reuse (shape *(N, n_dimensions)*).
    n_dimensions : int {2,3}
    output_format : {'html','png','both'}
    """
    if n_dimensions not in (2, 3):
        raise ValueError("n_dimensions must be 2 or 3")
    if output_format not in {"html", "png", "both"}:
        raise ValueError("output_format must be 'html', 'png', or 'both'")

    # ------------------------------------------------------------------
    # Prepare dataframe & colours
    # ------------------------------------------------------------------
    df = features_df.copy()
    BBOX_LIST = [f"bbox{i}" for i in range(1, 8)]
    bbox_colors = {
        **{b: c for b, c in zip(BBOX_LIST, ["#FF0000", "#00FF00", "#0000FF", "#FFA500", "#800080", "#00FFFF", "#FFD700"])},
        "other": "#808080",
    }
    df["bbox_color"] = df["bbox"].astype(str).str.lower().where(lambda s: s.isin(BBOX_LIST), "other")

    # ------------------------------------------------------------------
    # UMAP (reuse when provided)
    # ------------------------------------------------------------------
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    if reuse_umap_results is None:
        scaled = StandardScaler().fit_transform(df[feature_cols].values)
        umap_results, _ = compute_umap(scaled, n_components=n_dimensions)
    else:
        umap_results = reuse_umap_results
        if umap_results.shape[1] != n_dimensions:
            raise ValueError("reuse_umap_results has incompatible dimensionality")

    df[["umap_x", "umap_y"]] = umap_results[:, :2]
    if n_dimensions == 3:
        df["umap_z"] = umap_results[:, 2]

    # ------------------------------------------------------------------
    # Plot with Plotly Express – one call handles both 2-D and 3-D cases
    # ------------------------------------------------------------------
    import plotly.express as px

    hover_cols = [c for c in ["bbox", "Var1"] if c in df.columns]
    if n_dimensions == 3:
        fig = px.scatter_3d(
            df,
            x="umap_x",
            y="umap_y",
            z="umap_z",
            color="bbox_color",
            color_discrete_map=bbox_colors,
            opacity=0.8,
            hover_data=hover_cols,
        )
        filename_base = "umap3d_merged_bbox_colored"
        coord_cols = hover_cols + ["bbox_color", "umap_x", "umap_y", "umap_z"]
    else:
        fig = px.scatter(
            df,
            x="umap_x",
            y="umap_y",
            color="bbox_color",
            color_discrete_map=bbox_colors,
            opacity=0.8,
            hover_data=hover_cols,
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)  # keep aspect square
        filename_base = "umap2d_merged_bbox_colored"
        coord_cols = hover_cols + ["bbox_color", "umap_x", "umap_y"]

    fig.update_layout(title=f"{n_dimensions}D UMAP – bbox coloured", legend_title="Bounding Box")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    if output_format in {"html", "both"}:
        html_path = os.path.join(output_dir, f"{filename_base}.html")
        fig.write_html(html_path)
        print(f"Saved HTML to {html_path}")
    if output_format in {"png", "both"}:
        png_path = os.path.join(output_dir, f"{filename_base}.png")
        try:
            fig.write_image(png_path, scale=2)
            print(f"Saved PNG to {png_path}")
        except Exception as e:
            print(f"Could not save PNG via Plotly – {e}")

    # Save coordinates CSV
    coord_cols = [c for c in coord_cols if c in df.columns]
    df[coord_cols].to_csv(os.path.join(output_dir, f"{filename_base}_coordinates.csv"), index=False)

    return df, umap_results

def sample_points_from_umap(features_df, n_samples=100, method='spatial_grid'):
    has_3d = 'umap_z' in features_df.columns 
    n_samples = min(n_samples, len(features_df))
    if method == 'random':
        sampled_indices = np.random.choice(len(features_df), size=n_samples, replace=False)
    elif method == 'spatial_grid':   
        if has_3d:    
            x_coords = features_df['umap_x'].values
            y_coords = features_df['umap_y'].values
            z_coords = features_df['umap_z'].values
            grid_size = int(np.ceil(n_samples ** (1/3)))  
            x_bins = np.linspace(x_coords.min(), x_coords.max(), grid_size + 1)
            y_bins = np.linspace(y_coords.min(), y_coords.max(), grid_size + 1)
            z_bins = np.linspace(z_coords.min(), z_coords.max(), grid_size + 1)
            sampled_indices = []
            samples_per_cell = max(1, n_samples // (grid_size ** 3))
            for i in range(grid_size):
                for j in range(grid_size):
                    for k in range(grid_size):
                        
                        mask = ((x_coords >= x_bins[i]) & (x_coords < x_bins[i+1]) &
                               (y_coords >= y_bins[j]) & (y_coords < y_bins[j+1]) &
                               (z_coords >= z_bins[k]) & (z_coords < z_bins[k+1]))
                        
                        cell_indices = np.where(mask)[0]
                        if len(cell_indices) > 0:
                            
                            n_from_cell = min(samples_per_cell, len(cell_indices))
                            selected = np.random.choice(cell_indices, size=n_from_cell, replace=False)
                            sampled_indices.extend(selected)
        else:
            x_coords = features_df['umap_x'].values
            y_coords = features_df['umap_y'].values
            grid_size = int(np.ceil(np.sqrt(n_samples)))  
            x_bins = np.linspace(x_coords.min(), x_coords.max(), grid_size + 1)
            y_bins = np.linspace(y_coords.min(), y_coords.max(), grid_size + 1)
            sampled_indices = []
            samples_per_cell = max(1, n_samples // (grid_size ** 2))
            for i in range(grid_size):
                for j in range(grid_size):
                    mask = ((x_coords >= x_bins[i]) & (x_coords < x_bins[i+1]) &
                           (y_coords >= y_bins[j]) & (y_coords < y_bins[j+1]))
                    cell_indices = np.where(mask)[0]
                    if len(cell_indices) > 0:
                        n_from_cell = min(samples_per_cell, len(cell_indices))
                        selected = np.random.choice(cell_indices, size=n_from_cell, replace=False)
                        sampled_indices.extend(selected)
        sampled_indices = list(set(sampled_indices))  
        if len(sampled_indices) < n_samples:
            remaining_indices = [i for i in range(len(features_df)) if i not in sampled_indices]
            additional_needed = n_samples - len(sampled_indices)
            if len(remaining_indices) >= additional_needed:
                additional = np.random.choice(remaining_indices, size=additional_needed, replace=False)
                sampled_indices.extend(additional)
        if len(sampled_indices) > n_samples:
            sampled_indices = np.random.choice(sampled_indices, size=n_samples, replace=False)
    elif method == 'density_aware':
        if has_3d:
            coords = features_df[['umap_x', 'umap_y', 'umap_z']].values
        else:
            coords = features_df[['umap_x', 'umap_y']].values
        k = min(20, len(features_df) // 10)  
        nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
        distances, _ = nbrs.kneighbors(coords)
        density_scores = distances[:, -1]  
        probabilities = density_scores / density_scores.sum()
        sampled_indices = np.random.choice(
            len(features_df), 
            size=n_samples, 
            replace=False, 
            p=probabilities
        )
    mask = np.zeros(len(features_df), dtype=bool)
    mask[sampled_indices] = True
    sampled_df = features_df.iloc[sampled_indices].copy()
    sampled_df['is_sampled'] = True
    return sampled_df, mask

def create_umap_with_sampled_crosses(features_df, sampled_mask, output_dir, 
                                   n_dimensions=2, output_format='both', 
                                   cross_size=100, cross_color='red'):
    if 'umap_x' not in features_df.columns or 'umap_y' not in features_df.columns:
        raise ValueError("DataFrame must contain UMAP coordinates")
    
    
    if n_dimensions not in [2, 3]:
        raise ValueError("n_dimensions must be 2 or 3")
    if output_format not in ['html', 'png', 'both']:
        raise ValueError("output_format must be 'html', 'png', or 'both'")
    
    has_3d = 'umap_z' in features_df.columns and n_dimensions == 3
    
    
    bbox_colors = {
        'bbox1': '#FF0000',  
        'bbox2': '#00FF00',  
        'bbox3': '#0000FF',  
        'bbox4': '#FFA500',  
        'bbox5': '#800080',  
        'bbox6': '#00FFFF',  
        'bbox7': '#FFD700'   
    }
    
    
    features_df = features_df.copy()
    
    
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
    bbox_colors_extended['other'] = '#808080'  
    
    
    sampled_df = features_df[sampled_mask].copy()
    non_sampled_df = features_df[~sampled_mask].copy()
    
    print(f"Plotting {len(non_sampled_df)} background points and {len(sampled_df)} sampled crosses")
    
    
    if has_3d:
        fig = go.Figure()
        
        
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
        
        
        if len(sampled_df) > 0:
            fig.add_trace(go.Scatter3d(
                x=sampled_df['umap_x'],
                y=sampled_df['umap_y'],
                z=sampled_df['umap_z'],
                mode='markers',
                marker=dict(
                    color=cross_color,
                    size=cross_size//10,  
                    symbol='x',
                    opacity=0.9,
                    line=dict(width=2)
                ),
                name=f'Sampled ({len(sampled_df)} points)',
                showlegend=True,
                hovertemplate='<b>SAMPLED: %{text}</b><br>UMAP1: %{x}<br>UMAP2: %{y}<br>UMAP3: %{z}<extra></extra>',
                text=sampled_df.get('Var1', sampled_df.index)
            ))
        
        
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
    
    else:  
        fig = go.Figure()
        
        
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
        
        
        if len(sampled_df) > 0:
            fig.add_trace(go.Scatter(
                x=sampled_df['umap_x'],
                y=sampled_df['umap_y'],
                mode='markers',
                marker=dict(
                    color=cross_color,
                    size=cross_size//5,  
                    symbol='x',
                    opacity=0.9,
                    line=dict(width=3)
                ),
                name=f'Sampled ({len(sampled_df)} points)',
                showlegend=True,
                hovertemplate='<b>SAMPLED: %{text}</b><br>UMAP1: %{x}<br>UMAP2: %{y}<extra></extra>',
                text=sampled_df.get('Var1', sampled_df.index)
            ))
        
        
        fig.update_layout(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            title=f'2D UMAP with {len(sampled_df)} Randomly Sampled Points (crosses)',
            legend=dict(title="Categories", yanchor="top", y=0.99, xanchor="right", x=0.99),
            margin=dict(l=0, r=0, b=0, t=50),
            hovermode='closest'
        )
        
        filename_base = f"umap2d_with_sampled_crosses_{len(sampled_df)}"
    
    
    if output_format in ['html', 'both']:
        output_path_html = os.path.join(output_dir, f"{filename_base}.html")
        try:
            fig.write_html(output_path_html)
            print(f"Saved HTML: {output_path_html}")
        except Exception as e:
            print(f"Error saving HTML: {str(e)}")
    
    if output_format in ['png', 'both']:
        try:
            
            import matplotlib.pyplot as plt
            
            if has_3d:
                fig_mpl = plt.figure(figsize=(14, 10))
                ax = fig_mpl.add_subplot(111, projection='3d')
                
                
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
                
            else:  
                fig_mpl, ax = plt.subplots(figsize=(14, 10))
                
                
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
            
            
            ax.legend(title='Categories', loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            
            
            output_path_png = os.path.join(output_dir, f"{filename_base}.png")
            plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
            print(f"Saved PNG: {output_path_png}")
            
            output_path_pdf = os.path.join(output_dir, f"{filename_base}.pdf")
            plt.savefig(output_path_pdf, bbox_inches='tight')
            
            plt.close(fig_mpl)
            
        except Exception as e:
            print(f"Error creating matplotlib plot: {str(e)}")
    
    
    sampled_info_path = os.path.join(output_dir, f"sampled_points_{len(sampled_df)}.csv")
    sampled_df.to_csv(sampled_info_path, index=False)
    print(f"Saved sampled points info: {sampled_info_path}")
    
    return sampled_df

def create_umap_heatmaps(features_df, output_dir, n_dimensions=2, 
                        cmaps=['viridis', 'plasma', 'hot', 'coolwarm', 'Blues'],
                        bin_size=50, smooth_sigma=1.0):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import gaussian_filter
    import seaborn as sns
    
    if 'umap_x' not in features_df.columns or 'umap_y' not in features_df.columns:
        raise ValueError("DataFrame must contain UMAP coordinates")
    
    print(f"Creating UMAP heatmaps with {len(features_df)} points...")
    
    
    x_coords = features_df['umap_x'].values
    y_coords = features_df['umap_y'].values
    
    has_3d = 'umap_z' in features_df.columns and n_dimensions == 3
    if has_3d:
        z_coords = features_df['umap_z'].values
    
    
    heatmap_dir = os.path.join(output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    
    if has_3d:
        
        print("Creating 3D heatmap slices...")
        
        
        x_range = (x_coords.min(), x_coords.max())
        y_range = (y_coords.min(), y_coords.max())
        z_range = (z_coords.min(), z_coords.max())
        
        
        n_slices = 5
        z_slice_edges = np.linspace(z_range[0], z_range[1], n_slices + 1)
        
        for cmap_name in cmaps:
            print(f"  Creating 3D slices with {cmap_name} colormap...")
            
            fig, axes = plt.subplots(1, n_slices, figsize=(20, 4))
            if n_slices == 1:
                axes = [axes]
            
            for i in range(n_slices):
                z_min, z_max = z_slice_edges[i], z_slice_edges[i + 1]
                
                
                z_mask = (z_coords >= z_min) & (z_coords <= z_max)
                slice_x = x_coords[z_mask]
                slice_y = y_coords[z_mask]
                
                if len(slice_x) > 0:
                    
                    hist, x_edges, y_edges = np.histogram2d(
                        slice_x, slice_y, bins=bin_size,
                        range=[x_range, y_range]
                    )
                    
                    
                    if smooth_sigma > 0:
                        hist = gaussian_filter(hist, sigma=smooth_sigma)
                    
                    
                    im = axes[i].imshow(
                        hist.T, origin='lower', cmap=cmap_name, aspect='auto',
                        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                        interpolation='bilinear'
                    )
                    
                    axes[i].set_title(f'Z slice: [{z_min:.2f}, {z_max:.2f}]\n({len(slice_x)} points)')
                    axes[i].set_xlabel('UMAP Dimension 1')
                    if i == 0:
                        axes[i].set_ylabel('UMAP Dimension 2')
                    
                    
                    plt.colorbar(im, ax=axes[i], label='Density')
                else:
                    axes[i].text(0.5, 0.5, 'No points\nin slice', 
                               transform=axes[i].transAxes, ha='center', va='center')
                    axes[i].set_title(f'Z slice: [{z_min:.2f}, {z_max:.2f}]')
            
            plt.suptitle(f'3D UMAP Heatmap Slices - {cmap_name.capitalize()} Colormap')
            plt.tight_layout()
            
            
            output_path = os.path.join(heatmap_dir, f"heatmap_3d_slices_{cmap_name}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            output_path_pdf = os.path.join(heatmap_dir, f"heatmap_3d_slices_{cmap_name}.pdf")
            plt.savefig(output_path_pdf, bbox_inches='tight')
            
            plt.close()
            print(f"    Saved: heatmap_3d_slices_{cmap_name}.png")
    
    else:
        
        print("Creating 2D heatmaps...")
        
        
        x_range = (x_coords.min(), x_coords.max())
        y_range = (y_coords.min(), y_coords.max())
        
        
        hist, x_edges, y_edges = np.histogram2d(
            x_coords, y_coords, bins=bin_size,
            range=[x_range, y_range]
        )
        
        
        if smooth_sigma > 0:
            hist_smooth = gaussian_filter(hist, sigma=smooth_sigma)
        else:
            hist_smooth = hist
        
        
        for cmap_name in cmaps:
            print(f"  Creating heatmap with {cmap_name} colormap...")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            
            im1 = ax1.imshow(
                hist.T, origin='lower', cmap=cmap_name, aspect='auto',
                extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                interpolation='nearest'
            )
            ax1.set_title(f'Raw Density Heatmap\n({len(features_df)} points)')
            ax1.set_xlabel('UMAP Dimension 1')
            ax1.set_ylabel('UMAP Dimension 2')
            plt.colorbar(im1, ax=ax1, label='Point Count')
            
            
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
            
            
            output_path = os.path.join(heatmap_dir, f"heatmap_2d_{cmap_name}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            output_path_pdf = os.path.join(heatmap_dir, f"heatmap_2d_{cmap_name}.pdf")
            plt.savefig(output_path_pdf, bbox_inches='tight')
            
            plt.close()
            print(f"    Saved: heatmap_2d_{cmap_name}.png")
    
    
    print("Creating colormap comparison plot...")
    
    if has_3d:
        
        z_mid = (z_coords.min() + z_coords.max()) / 2
        z_range_width = (z_coords.max() - z_coords.min()) / 5  
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
            
            
            plt.colorbar(im, ax=axes[i], label='Density')
    
    
    for i in range(len(cmaps), len(axes)):
        axes[i].set_visible(False)
    
    dimension_text = "3D (middle slice)" if has_3d else "2D"
    plt.suptitle(f'UMAP Heatmap Colormap Comparison - {dimension_text}')
    plt.tight_layout()
    
    
    output_path = os.path.join(heatmap_dir, "heatmap_colormap_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    output_path_pdf = os.path.join(heatmap_dir, "heatmap_colormap_comparison.pdf")
    plt.savefig(output_path_pdf, bbox_inches='tight')
    
    plt.close()
    print(f"Saved colormap comparison: heatmap_colormap_comparison.png")
    
    
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
    
    N_DIMENSIONS = 2  
    OUTPUT_FORMAT = 'both'  
    
    
    SAMPLE_POINTS = True  
    N_SAMPLES = 100  
    SAMPLING_METHOD = 'spatial_grid'  

    
    CREATE_HEATMAPS = True  
    HEATMAP_CMAPS = ['viridis', 'plasma', 'hot', 'coolwarm', 'Blues', 'Reds', 'inferno', 'magma']
    HEATMAP_BIN_SIZE = 50  
    HEATMAP_SMOOTH_SIGMA = 1.0  

    
    csv1_path = "features_20250602_174807.csv"
    csv2_path = "raven_results/features_avg_seg11_alpha1_0.csv"
    
    
    merged_df = merge_feature_datasets(csv1_path, csv2_path)
    
    
    merged_output_path = "merged_features.csv"
    merged_df.to_csv(merged_output_path, index=False)

    
    os.makedirs(output_dir, exist_ok=True)
    df_with_umap, umap_results = create_bbox_colored_umap(
        merged_df, 
        output_dir, 
        n_dimensions=N_DIMENSIONS, 
        output_format=OUTPUT_FORMAT
    )
    
    
    if CREATE_HEATMAPS:
        heatmap_dir = create_umap_heatmaps(
            df_with_umap,
            output_dir,
            n_dimensions=N_DIMENSIONS,
            cmaps=HEATMAP_CMAPS,
            bin_size=HEATMAP_BIN_SIZE,
            smooth_sigma=HEATMAP_SMOOTH_SIGMA
        )
    
    
    if SAMPLE_POINTS:
        
        sampled_df, sampled_mask = sample_points_from_umap(
            df_with_umap, 
            n_samples=N_SAMPLES, 
            method=SAMPLING_METHOD
        )
        
        
        sampled_crosses_df = create_umap_with_sampled_crosses(
            df_with_umap,
            sampled_mask,
            output_dir,
            n_dimensions=N_DIMENSIONS,
            output_format=OUTPUT_FORMAT,
            cross_size=100,
            cross_color='red'
        )
