"""
Segmentation type processor for the SynapseDataLoader.

This module contains the logic for processing different segmentation types
that was extracted from the main dataloader for better modularity.
"""

import numpy as np
from synapse.dl.mask_utils import get_closest_component_mask


def process_segmentation_type(
    segmentation_type: int,
    add_mask_vol: np.ndarray,
    mask_1_full: np.ndarray, 
    mask_2_full: np.ndarray,
    presynapse_side: int,
    presynapse_mask_full: np.ndarray,
    z_start: int, z_end: int,
    y_start: int, y_end: int, 
    x_start: int, x_end: int,
    cx: int, cy: int, cz: int,
    vesicle_label: int,
    cleft_label: int,
    cleft_label2: int,
    mito_label: int,

):
    """
    Process different segmentation types and return the appropriate combined mask.
    
    Returns:
        tuple: (combined_mask_full, additional_data)
               where additional_data can contain special processing results for certain types
    """
    
    if segmentation_type == 0:
        combined_mask_full = np.ones_like(add_mask_vol, dtype=bool)
    elif segmentation_type == 1:
        combined_mask_full = mask_1_full if presynapse_side == 1 else mask_2_full
    elif segmentation_type == 2:
        combined_mask_full = mask_2_full if presynapse_side == 1 else mask_1_full
    elif segmentation_type == 3:
        combined_mask_full = np.logical_or(mask_1_full, mask_2_full)
    elif segmentation_type == 4:
        vesicle_closest = get_closest_component_mask(
            (add_mask_vol == vesicle_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
        cleft_closest = get_closest_component_mask(
            ((add_mask_vol == cleft_label)), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
        cleft_closest2 = get_closest_component_mask(
            ((add_mask_vol == cleft_label2)), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
        combined_mask_full = np.logical_or(vesicle_closest, np.logical_or(cleft_closest,cleft_closest2))
    elif segmentation_type == 5:
        vesicle_closest = get_closest_component_mask(
            (add_mask_vol == vesicle_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
        cleft_closest = get_closest_component_mask(
            (add_mask_vol == cleft_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
        combined_mask_extra = np.logical_or(vesicle_closest, cleft_closest)
        combined_mask_full = np.logical_or(mask_1_full, np.logical_or(mask_2_full, combined_mask_extra))
    elif segmentation_type == 6:
        combined_mask_full = get_closest_component_mask(
            (add_mask_vol == vesicle_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
    elif segmentation_type == 7:
        cleft_closest = get_closest_component_mask(
            ((add_mask_vol == cleft_label)), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
        cleft_closest2 = get_closest_component_mask(
            ((add_mask_vol == cleft_label2)), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
        combined_mask_full =  np.logical_or(cleft_closest,cleft_closest2)
    elif segmentation_type == 8:
        combined_mask_full = get_closest_component_mask(
            (add_mask_vol == mito_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
    elif segmentation_type == 9:
        vesicle_closest = get_closest_component_mask(
            (add_mask_vol == vesicle_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
        cleft_closest = get_closest_component_mask(
            (add_mask_vol == cleft_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
        combined_mask_full = np.logical_or(cleft_closest,vesicle_closest)
    elif segmentation_type == 10:
        cleft_closest = get_closest_component_mask(
            (add_mask_vol == cleft_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
        pre_mask_full = mask_1_full if presynapse_side == 1 else mask_2_full
        combined_mask_full = np.logical_or(cleft_closest,pre_mask_full)
    elif segmentation_type == 11: # segtype 10 without mito
        # Get cleft mask
        cleft_closest = get_closest_component_mask(
            (add_mask_vol == cleft_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz)
        )
        pre_mask_full = mask_1_full if presynapse_side == 1 else mask_2_full
        all_mito_mask = (add_mask_vol == mito_label)
        
        # Dilate the mitochondria mask by 2 voxels to create a safety margin
        from scipy import ndimage
        dilated_mito_mask = ndimage.binary_dilation(all_mito_mask, iterations=2)
        
        combined_temp = np.logical_or(cleft_closest, pre_mask_full)
        # Exclude dilated mitochondria mask from the combined mask
        combined_mask_full = np.logical_and(combined_temp, np.logical_not(dilated_mito_mask))
    return combined_mask_full