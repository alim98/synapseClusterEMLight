"""
Mask processing utilities for synapse analysis.

This module contains utility functions for processing and analyzing 
3D masks in the synapse analysis pipeline.
"""

import numpy as np
from scipy.ndimage import label


def get_closest_component_mask(full_mask, z_start, z_end, y_start, y_end, x_start, x_end, target_coord):
    """
    Find the closest connected component in a mask to a target coordinate.
    
    This function extracts a sub-region from a full mask, identifies connected 
    components within that region, and returns a mask containing only the 
    component that is closest to the specified target coordinate.
    
    Args:
        full_mask (np.ndarray): The full 3D binary mask to process
        z_start (int): Starting Z coordinate of the sub-region
        z_end (int): Ending Z coordinate of the sub-region  
        y_start (int): Starting Y coordinate of the sub-region
        y_end (int): Ending Y coordinate of the sub-region
        x_start (int): Starting X coordinate of the sub-region
        x_end (int): Ending X coordinate of the sub-region
        target_coord (tuple): Target coordinate (x, y, z) to find closest component to
        
    Returns:
        np.ndarray: Binary mask with same shape as full_mask, containing only 
                   the closest connected component within the specified region.
                   Returns zeros if no components found.
    """
    # Extract the sub-region of interest
    sub_mask = full_mask[z_start:z_end, y_start:y_end, x_start:x_end]
    
    # Label connected components in the sub-region
    labeled_sub_mask, num_features = label(sub_mask)
    
    # Return empty mask if no components found
    if num_features == 0:
        return np.zeros_like(full_mask, dtype=bool)
    
    # Find the closest component to the target coordinate
    cx, cy, cz = target_coord
    min_distance = float('inf')
    closest_label = None

    for label_num in range(1, num_features + 1):
        # Get all coordinates of pixels belonging to this component
        vesicle_coords = np.column_stack(np.where(labeled_sub_mask == label_num))

        # Calculate distances from each pixel to the target coordinate
        # Note: vesicle_coords are in (z, y, x) order from np.where
        distances = np.sqrt(
            (vesicle_coords[:, 0] + z_start - cz) ** 2 +
            (vesicle_coords[:, 1] + y_start - cy) ** 2 +
            (vesicle_coords[:, 2] + x_start - cx) ** 2
        )

        # Find the minimum distance for this component
        min_dist_for_vesicle = np.min(distances)
        
        # Update closest component if this one is closer
        if min_dist_for_vesicle < min_distance:
            min_distance = min_dist_for_vesicle
            closest_label = label_num

    # Create the output mask with only the closest component
    if closest_label is not None:
        # Create a mask for just the closest component in the sub-region
        filtered_sub_mask = (labeled_sub_mask == closest_label)
        
        # Create the full-size output mask
        combined_mask = np.zeros_like(full_mask, dtype=bool)
        combined_mask[z_start:z_end, y_start:y_end, x_start:x_end] = filtered_sub_mask
        
        return combined_mask
    else:
        # Fallback: return empty mask
        return np.zeros_like(full_mask, dtype=bool)


def get_component_center_of_mass(mask):
    """
    Calculate the center of mass of a binary mask.
    
    Args:
        mask (np.ndarray): Binary 3D mask
        
    Returns:
        tuple: Center of mass coordinates (z, y, x), or None if mask is empty
    """
    if not np.any(mask):
        return None
    
    coords = np.array(np.where(mask)).T
    center_of_mass = np.mean(coords, axis=0)
    return tuple(center_of_mass)


def filter_components_by_size(mask, min_size=None, max_size=None):
    """
    Filter connected components in a mask by size.
    
    Args:
        mask (np.ndarray): Binary 3D mask
        min_size (int, optional): Minimum component size in voxels
        max_size (int, optional): Maximum component size in voxels
        
    Returns:
        np.ndarray: Filtered binary mask with same shape as input
    """
    labeled_mask, num_features = label(mask)
    
    if num_features == 0:
        return mask.copy()
    
    filtered_mask = np.zeros_like(mask, dtype=bool)
    
    for label_num in range(1, num_features + 1):
        component_mask = (labeled_mask == label_num)
        component_size = np.sum(component_mask)
        
        # Check size constraints
        if min_size is not None and component_size < min_size:
            continue
        if max_size is not None and component_size > max_size:
            continue
            
        # Add this component to the filtered mask
        filtered_mask |= component_mask
    
    return filtered_mask


def get_largest_component_mask(mask):
    """
    Get a mask containing only the largest connected component.
    
    Args:
        mask (np.ndarray): Binary 3D mask
        
    Returns:
        np.ndarray: Binary mask with same shape as input, containing only 
                   the largest connected component
    """
    labeled_mask, num_features = label(mask)
    
    if num_features == 0:
        return np.zeros_like(mask, dtype=bool)
    
    # Find the largest component
    largest_size = 0
    largest_label = None
    
    for label_num in range(1, num_features + 1):
        component_size = np.sum(labeled_mask == label_num)
        if component_size > largest_size:
            largest_size = component_size
            largest_label = label_num
    
    # Return mask with only the largest component
    if largest_label is not None:
        return labeled_mask == largest_label
    else:
        return np.zeros_like(mask, dtype=bool) 