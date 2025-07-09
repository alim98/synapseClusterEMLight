
import numpy as np
from scipy.ndimage import label


def get_closest_component_mask(full_mask, z_start, z_end, y_start, y_end, x_start, x_end, target_coord):
    sub_mask = full_mask[z_start:z_end, y_start:y_end, x_start:x_end]
    
    labeled_sub_mask, num_features = label(sub_mask)
    cx, cy, cz = target_coord
    min_distance = float('inf')
    closest_label = None
    for label_num in range(1, num_features + 1):   
        vesicle_coords = np.column_stack(np.where(labeled_sub_mask == label_num))
        distances = np.sqrt(
            (vesicle_coords[:, 0] + z_start - cz) ** 2 +
            (vesicle_coords[:, 1] + y_start - cy) ** 2 +
            (vesicle_coords[:, 2] + x_start - cx) ** 2
        )
        min_dist_for_vesicle = np.min(distances)
        if min_dist_for_vesicle < min_distance:
            min_distance = min_dist_for_vesicle
            closest_label = label_num
    if closest_label is not None:
        filtered_sub_mask = (labeled_sub_mask == closest_label)
        combined_mask = np.zeros_like(full_mask, dtype=bool)
        combined_mask[z_start:z_end, y_start:y_end, x_start:x_end] = filtered_sub_mask
        
        return combined_mask
    else:
        
        return np.zeros_like(full_mask, dtype=bool)
