import pandas as pd
import os
import glob
import sys
import numpy as np
from pathlib import Path
from tifffile import imread
from collections import Counter
from scipy.ndimage import label
from typing import Tuple, Optional

# Add the synapse_sampling directory to the Python path
sys.path.append(str(Path(__file__).parent))
from wk_convert import calculate_bbox_coordinates

# cache for loaded volumes so that each bbox is read only once
_vol_cache = {}

def get_closest_component_mask(full_mask, z_start, z_end, y_start, y_end, x_start, x_end, target_coord):
    """
    Get the closest component mask to the target coordinate.
    
    Args:
        full_mask (numpy.ndarray): The full mask
        z_start, z_end, y_start, y_end, x_start, x_end (int): Bounding box coordinates
        target_coord (tuple): Target coordinate (cx, cy, cz)
        
    Returns:
        numpy.ndarray: The closest component mask
    """
    sub_mask = full_mask[z_start:z_end, y_start:y_end, x_start:x_end]
    labeled_sub_mask, num_features = label(sub_mask)
    
    if num_features == 0:
        return np.zeros_like(full_mask, dtype=bool)
    else:
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

def get_bbox_labels(bbox_num):
    """
    Get the label IDs for the given bounding box number.
    
    Args:
        bbox_num (int): Bounding box number
        
    Returns:
        tuple: (mito_label, vesicle_label, cleft_label, cleft_label2)
    """
    bbox_str = str(bbox_num)
    
    if bbox_str in {'2', '5'}:
        mito_label = 1
        vesicle_label = 3
        cleft_label2 = 4
        cleft_label = 2
    elif bbox_str == '7':
        mito_label = 1
        vesicle_label = 2
        cleft_label2 = 3
        cleft_label = 4
    elif bbox_str == '4':
        mito_label = 3
        vesicle_label = 2
        cleft_label2 = 4
        cleft_label = 1
    elif bbox_str == '3':
        mito_label = 6
        vesicle_label = 7
        cleft_label2 = 8
        cleft_label = 9
    else:
        mito_label = 5
        vesicle_label = 6
        cleft_label = 7
        cleft_label2 = 7
    
    return mito_label, vesicle_label, cleft_label, cleft_label2

def _find_slice_file(slice_dir: str, z_val: int) -> Tuple[str, int]:
    """Return the path to a slice_{z}.tif inside slice_dir.
    If the exact slice does not exist the slice with the smallest distance
    to *z_val* is returned. The function also returns the slice number that
    was finally chosen.
    """
    candidate = os.path.join(slice_dir, f"slice_{z_val}.tif")
    if os.path.exists(candidate):
        return candidate, z_val

    # search for closest
    slice_nums = []
    for fname in os.listdir(slice_dir):
        if fname.startswith("slice_") and fname.endswith(".tif"):
            try:
                num = int(fname.replace("slice_", "").replace(".tif", ""))
                slice_nums.append(num)
            except ValueError:
                pass

    if not slice_nums:
        raise FileNotFoundError(f"No slice_*.tif found in {slice_dir}")

    closest = min(slice_nums, key=lambda x: abs(x - z_val))
    return os.path.join(slice_dir, f"slice_{closest}.tif"), closest

def identify_presynaptic_id(rel_x, rel_y, rel_z,
                           side1_coords: Tuple[int, int, int],
                           side2_coords: Tuple[int, int, int],
                           bbox_num: int,
                           seg_dir: str,
                           add_mask_root: str) -> Optional[int]:
    """
    Identify the presynaptic agglomeration ID by finding the side with more overlap with vesicle mask.
    
    Args:
        rel_x, rel_y, rel_z (int): Relative coordinates
        bbox_num (int): Bounding box number
        seg_dir (str): Directory for segmentation data
        add_mask_root (str): Directory for additional mask data
        
    Returns:
        int: Presynaptic agglomeration ID or None if not found
    """
    try:
        # directories for slices
        seg_slice_dir = os.path.join(seg_dir, f"bbox{bbox_num}")
        add_mask_dir = os.path.join(add_mask_root, f"bbox_{bbox_num}")  # note underscore in folder name

        # Use central z to pick a working slice (closest existing)
        seg_slice_path, seg_used_z = _find_slice_file(seg_slice_dir, int(rel_z))
        add_mask_path, _ = _find_slice_file(add_mask_dir, seg_used_z)
        
        # Read the files
        seg_vol = imread(seg_slice_path)
        add_mask_vol = imread(add_mask_path)
        
        # Retrieve label ids for this bbox (needed for vesicle_label)
        _, vesicle_label, _, _ = get_bbox_labels(bbox_num)
        
        # Convert 2D slices to 3D volumes (1 slice)
        if len(seg_vol.shape) == 2:
            seg_vol = seg_vol[np.newaxis, :, :]
        if len(add_mask_vol.shape) == 2:
            add_mask_vol = add_mask_vol[np.newaxis, :, :]
        
        # Define the region of interest
        z_start = 0
        z_end = seg_vol.shape[0]
        y_start = max(0, int(rel_y) - 40)
        y_end = min(seg_vol.shape[1], int(rel_y) + 40)
        x_start = max(0, int(rel_x) - 40)
        x_end = min(seg_vol.shape[2], int(rel_x) + 40)
        
        # IDs at side1 and side2 coords (note z index 0 since single slice)
        x1, y1, z1 = side1_coords
        x2, y2, z2 = side2_coords

        seg_id_side1 = seg_vol[0, int(y1), int(x1)]
        seg_id_side2 = seg_vol[0, int(y2), int(x2)]

        mask_side1 = (seg_vol == seg_id_side1)
        mask_side2 = (seg_vol == seg_id_side2)
        
        # Find vesicles in the region
        vesicle_full_mask = (add_mask_vol == vesicle_label)
        vesicle_mask = get_closest_component_mask(
            vesicle_full_mask,
            z_start, z_end, 
            y_start, y_end, 
            x_start, x_end,
            (rel_x, rel_y, 0)  # Z-coordinate is 0 since we're working with a single slice
        )
        
        # Check overlap with side1
        overlap_side1 = int(np.sum(mask_side1 & vesicle_mask))
        
        # Check overlap with side2
        overlap_side2 = int(np.sum(mask_side2 & vesicle_mask))
        
        if overlap_side1 == 0 and overlap_side2 == 0:
            # Fall back: choose the side closer to vesicle component centre
            return None

        return int(seg_id_side1) if overlap_side1 >= overlap_side2 else int(seg_id_side2)
    
    except Exception as e:
        print(f"Error identifying presynaptic ID: {e}")
        return None

def extract_agglo_id_from_masks(bbox_num, rel_x, rel_y, rel_z, seg_base_dir="data/data/seg"):
    """
    Extract agglomeration ID from segmentation mask files at the specified position.
    
    Args:
        bbox_num (int): Bounding box number
        rel_x (int): Relative x coordinate
        rel_y (int): Relative y coordinate
        rel_z (int): Relative z coordinate
        seg_base_dir (str): Base directory for segmentation data
        
    Returns:
        tuple: (agglo_id, frequency) - most common agglomeration ID and its frequency
    """
    # Calculate which slice corresponds to the z position
    z_slice = int(rel_z)  # Convert to int, removing fraction if present
    
    # Construct the path to the segmentation mask
    seg_dir = os.path.join(seg_base_dir, f"bbox{bbox_num}")
    mask_file = os.path.join(seg_dir, f"slice_{z_slice}.tif")
    
    if not os.path.exists(mask_file):
        # Try looking for the closest slice
        slices = []
        for f in os.listdir(seg_dir):
            if f.startswith("slice_") and f.endswith(".tif"):
                try:
                    slice_num = int(f.replace("slice_", "").replace(".tif", ""))
                    slices.append(slice_num)
                except:
                    pass
        
        if not slices:
            return None, 0
            
        # Find closest slice
        closest_slice = min(slices, key=lambda x: abs(x - z_slice))
        mask_file = os.path.join(seg_dir, f"slice_{closest_slice}.tif")
        print(f"Slice {z_slice} not found, using closest slice {closest_slice}")
    
    try:
        # Read the mask file
        mask = imread(mask_file)
        
        # Window around the central coordinate. Index order in 2-D slice is [Y, X]
        window_size = 5
        y_min = max(0, int(rel_y) - window_size)
        y_max = min(mask.shape[0], int(rel_y) + window_size + 1)
        x_min = max(0, int(rel_x) - window_size)
        x_max = min(mask.shape[1], int(rel_x) + window_size + 1)

        region = mask[y_min:y_max, x_min:x_max]
        
        # Find the most common ID in the region (excluding 0 which is typically background)
        ids = region.flatten()
        ids = ids[ids > 0]  # Remove background (0) values
        
        if len(ids) == 0:
            return None, 0
            
        # Count occurrences of each ID
        id_counter = Counter(ids)
        most_common_id, frequency = id_counter.most_common(1)[0]
        
        return int(most_common_id), frequency
        
    except Exception as e:
        print(f"Error reading mask file {mask_file}: {e}")
        return None, 0

def _load_volumes_for_bbox(bbox_num: int, seg_root: str, mask_root: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load full 3-D segmentation and additional-mask volumes for the given bbox.
    The function uses a simple cache so every bbox is read only once.
    """
    if bbox_num in _vol_cache:
        return _vol_cache[bbox_num]

    seg_dir = os.path.join(seg_root, f"bbox{bbox_num}")
    mask_dir = os.path.join(mask_root, f"bbox_{bbox_num}")

    seg_files = sorted(glob.glob(os.path.join(seg_dir, "slice_*.tif")), key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "slice_*.tif")), key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))

    if len(seg_files) == 0 or len(seg_files) != len(mask_files):
        raise RuntimeError(f"Mismatch or no slices for bbox{bbox_num}: seg {len(seg_files)} mask {len(mask_files)}")

    seg_slices = [imread(f).astype(np.uint32) for f in seg_files]
    mask_slices = [imread(f).astype(np.uint32) for f in mask_files]

    seg_vol = np.stack(seg_slices, axis=0)
    add_mask_vol = np.stack(mask_slices, axis=0)

    _vol_cache[bbox_num] = (seg_vol, add_mask_vol)
    return seg_vol, add_mask_vol

def main():
    # Path to the data directory containing Excel files
    data_dir = Path("data/data")
    seg_dir = Path("data/data/seg")
    add_mask_root = Path("data/data/vesicle_cloud__syn_interface__mitochondria_annotation")
    
    # Find all Excel files matching the pattern bbox*.xlsx
    excel_files = glob.glob(str(data_dir / "bbox*.xlsx"))
    
    # Initialize an empty list to store all data
    all_data = []
    
    # Process each Excel file
    for excel_file in excel_files:
        # Extract bbox number from filename
        bbox_num = int(Path(excel_file).stem.replace("bbox", ""))
        print(f"Processing {excel_file} (bbox {bbox_num})")
        
        try:
            # Read the Excel file
            df = pd.read_excel(excel_file)
            
            # Check if the required columns exist
            required_cols = ["central_coord_1", "central_coord_2", "central_coord_3"]
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: Required columns not found in {excel_file}. Skipping.")
                continue
            
            # Process each row in the Excel file
            for idx, row in df.iterrows():
                rel_x = row["central_coord_1"]
                rel_y = row["central_coord_2"]
                rel_z = row["central_coord_3"]
                
                # Calculate absolute coordinates
                abs_x, abs_y, abs_z = calculate_bbox_coordinates(rel_x, rel_y, rel_z, bbox_num)
                
                # Load (or fetch from cache) full volumes for this bbox
                try:
                    seg_vol, add_mask_vol = _load_volumes_for_bbox(bbox_num, str(seg_dir), str(add_mask_root))
                except Exception as e:
                    print(f"Failed loading volumes for bbox{bbox_num}: {e}")
                    continue

                # prepare side coordinates
                side1 = (
                    int(row["side_1_coord_1"]),
                    int(row["side_1_coord_2"]),
                    int(row["side_1_coord_3"]),
                )
                side2 = (
                    int(row["side_2_coord_1"]),
                    int(row["side_2_coord_2"]),
                    int(row["side_2_coord_3"]),
                )

                # ---------------- determine presynaptic id -----------------
                _, vesicle_label, _, _ = get_bbox_labels(bbox_num)

                # Build vesicle mask (full volume)
                vesicle_full_mask = (add_mask_vol == vesicle_label)

                # Region of interest around cleft centre (same logic as dataloader)
                half_size = 80  # voxels
                cx, cy, cz = int(rel_x), int(rel_y), int(rel_z)
                z_start = max(cz - half_size, 0)
                z_end   = min(cz + half_size, seg_vol.shape[0])
                y_start = max(cy - half_size, 0)
                y_end   = min(cy + half_size, seg_vol.shape[1])
                x_start = max(cx - half_size, 0)
                x_end   = min(cx + half_size, seg_vol.shape[2])

                vesicle_mask = get_closest_component_mask(
                    vesicle_full_mask,
                    z_start, z_end,
                    y_start, y_end,
                    x_start, x_end,
                    (cz, cy, cx)  # note order (Z,Y,X)
                )

                # Segment ids at side coords
                x1, y1, z1 = side1
                x2, y2, z2 = side2
                if z1 >= seg_vol.shape[0] or y1 >= seg_vol.shape[1] or x1 >= seg_vol.shape[2]:
                    seg_id_side1 = 0
                else:
                    seg_id_side1 = int(seg_vol[z1, y1, x1])
                if z2 >= seg_vol.shape[0] or y2 >= seg_vol.shape[1] or x2 >= seg_vol.shape[2]:
                    seg_id_side2 = 0
                else:
                    seg_id_side2 = int(seg_vol[z2, y2, x2])

                mask_side1 = (seg_vol == seg_id_side1) if seg_id_side1 != 0 else np.zeros_like(seg_vol, dtype=bool)
                mask_side2 = (seg_vol == seg_id_side2) if seg_id_side2 != 0 else np.zeros_like(seg_vol, dtype=bool)

                overlap1 = int(np.sum(mask_side1 & vesicle_mask))
                overlap2 = int(np.sum(mask_side2 & vesicle_mask))

                if overlap1 == 0 and overlap2 == 0:
                    presynaptic_id = None
                else:
                    presynaptic_id = seg_id_side1 if overlap1 >= overlap2 else seg_id_side2

                # Create a dictionary for this data point
                data_point = {
                    "bboxnumber": int(bbox_num),
                    "abspos1": int(abs_x),
                    "abspos2": int(abs_y),
                    "abspos3": int(abs_z),
                    "agglo_id": int(presynaptic_id) if presynaptic_id is not None else np.nan,
                    "relpos1": int(rel_x),
                    "relpos2": int(rel_y),
                    "relpos3": int(rel_z)
                }
                
                if presynaptic_id is not None:
                    data_point["is_presynaptic"] = 1
                else:
                    data_point["is_presynaptic"] = 0
                
                # Use existing Var1 if it exists, otherwise use row index + 1
                if "Var1" in row:
                    data_point["Var1"] = row["Var1"]
                else:
                    data_point["Var1"] = idx + 1
                
                all_data.append(data_point)
                
        except Exception as e:
            print(f"Error processing {excel_file}: {e}")
    
    # Create a DataFrame from all collected data
    result_df = pd.DataFrame(all_data)
    
    # Ensure Var1 is the first column
    if "Var1" in result_df.columns:
        cols = ["Var1"] + [col for col in result_df.columns if col != "Var1"]
        result_df = result_df[cols]
    
    # Save the DataFrame to a CSV file
    output_path = Path("data/absolute_positions_with_agglo.csv")
    result_df.to_csv(output_path, index=False)
    
    print(f"Conversion complete. Saved {len(result_df)} positions to {output_path}")
    print(f"Data sample:\n{result_df.head()}")
    
    # Report on agglomeration ID finding
    if "agglo_id" in result_df.columns:
        print(f"Found agglomeration IDs for {result_df['agglo_id'].notna().sum()} out of {len(result_df)} positions")
    
    if "presynaptic_id" in result_df.columns:
        print(f"Found presynaptic IDs for {result_df['presynaptic_id'].notna().sum()} out of {len(result_df)} positions")

if __name__ == "__main__":
    main() 