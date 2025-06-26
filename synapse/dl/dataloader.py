import os
import glob
import numpy as np
import torch
from typing import Tuple
import imageio.v3 as iio
from scipy.ndimage import label
from torchvision import transforms
import matplotlib.pyplot as plt

# Import the config
from synapse.utils.config import config
# Import mask utilities
from synapse.dl.mask_utils import get_closest_component_mask
# Import segmentation type processor
from synapse.dl.segtype_processor import process_segmentation_type
# Import mask label configuration
from synapse.dl.mask_labels import get_mask_labels

class Synapse3DProcessor:
    def __init__(self, size=(80, 80), mean=(0.485,), std=(0.229,)):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            # Explicitly convert to grayscale with one output channel
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        self.mean = mean
        self.std = std
        self.normalize_volume = True  # New flag to control volume-wide normalization

    def __call__(self, frames, return_tensors=None):
        processed_frames = []
        for frame in frames:
            # Check if input is RGB (3 channels) or has unexpected shape
            if len(frame.shape) > 2 and frame.shape[2] > 1:
                if frame.shape[2] > 3:  # More than 3 channels
                    frame = frame[:, :, :3]  # Take first 3 channels
                # Will be converted to grayscale by the transform
            
            processed_frame = self.transform(frame)
            processed_frames.append(processed_frame)
            
        pixel_values = torch.stack(processed_frames)
        
        # Ensure we have a single channel
        if pixel_values.shape[1] != 1:
            # This should not happen due to transforms.Grayscale, but just in case
            pixel_values = pixel_values.mean(dim=1, keepdim=True)
        
        # Apply volume-wide normalization to ensure consistent grayscale values across slices
        if self.normalize_volume:
            # Method 1: Min-max normalization across the entire volume
            min_val = pixel_values.min()
            max_val = pixel_values.max()
            if max_val > min_val:  # Avoid division by zero
                pixel_values = (pixel_values - min_val) / (max_val - min_val)
            
        if return_tensors == "pt":
            return {"pixel_values": pixel_values}
        else:
            return pixel_values

class SynapseDataLoader:
    def __init__(self, raw_base_dir, seg_base_dir, add_mask_base_dir, gray_color=None):
        self.raw_base_dir = raw_base_dir
        self.seg_base_dir = seg_base_dir
        self.add_mask_base_dir = add_mask_base_dir
        # Use provided gray_color or get from config
        self.gray_color = gray_color if gray_color is not None else config.gray_color

    def load_volumes(self, bbox_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw_dir = os.path.join(self.raw_base_dir, bbox_name)
        seg_dir = os.path.join(self.seg_base_dir, bbox_name)
        
        if bbox_name.startswith("bbox"):
            bbox_num = bbox_name.replace("bbox", "")
            add_mask_dir = os.path.join(self.add_mask_base_dir, f"bbox_{bbox_num}")
        else:
            add_mask_dir = os.path.join(self.add_mask_base_dir, bbox_name)
        
        raw_tif_files = sorted(glob.glob(os.path.join(raw_dir, 'slice_*.tif')))
        seg_tif_files = sorted(glob.glob(os.path.join(seg_dir, 'slice_*.tif')))
        add_mask_tif_files = sorted(glob.glob(os.path.join(add_mask_dir, 'slice_*.tif')))
        
        if not (len(raw_tif_files) == len(seg_tif_files) == len(add_mask_tif_files)):
            return None, None, None
        
        try:
            # Load raw volume and convert to grayscale if needed
            raw_slices = []
            multi_channel_detected = False
            for f in raw_tif_files:
                img = iio.imread(f)
                # Check if the image has multiple channels (RGB)
                if len(img.shape) > 2 and img.shape[2] > 1:
                    multi_channel_detected = True
                    # Convert RGB to grayscale using luminosity method
                    img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
                raw_slices.append(img)
            raw_vol = np.stack(raw_slices, axis=0)
            
            # Load segmentation volume and ensure it's single channel
            seg_slices = []
            for f in seg_tif_files:
                img = iio.imread(f)
                if len(img.shape) > 2 and img.shape[2] > 1:
                    multi_channel_detected = True
                    # For segmentation, take first channel (labels should be consistent)
                    img = img[..., 0]
                seg_slices.append(img.astype(np.uint32))
            seg_vol = np.stack(seg_slices, axis=0)
            
            # Load additional mask volume and ensure it's single channel
            add_mask_slices = []
            for f in add_mask_tif_files:
                img = iio.imread(f)
                if len(img.shape) > 2 and img.shape[2] > 1:
                    multi_channel_detected = True
                    # For masks, take first channel
                    img = img[..., 0]
                add_mask_slices.append(img.astype(np.uint32))
            add_mask_vol = np.stack(add_mask_slices, axis=0)
            
            if multi_channel_detected:
                print(f"WARNING: Multi-channel images detected in {bbox_name} and converted to single-channel")
            
            return raw_vol, seg_vol, add_mask_vol
        except Exception as e:
            print(f"Error loading volumes for {bbox_name}: {e}")
            return None, None, None

    def create_segmented_cube(
        self,
        raw_vol: np.ndarray,
        seg_vol: np.ndarray,
        add_mask_vol: np.ndarray,
        central_coord: Tuple[int, int, int],
        side1_coord: Tuple[int, int, int],
        side2_coord: Tuple[int, int, int],
        segmentation_type: int,
        subvolume_size: int = 80,
        alpha: float = 0.3,
        bbox_name: str = "",
        normalize_across_volume: bool = True,

    ) -> np.ndarray:
        bbox_num = bbox_name.replace("bbox", "").strip()
        
        # Get mask labels for this bounding box
        labels = get_mask_labels(bbox_num)
        mito_label = labels['mito_label']
        vesicle_label = labels['vesicle_label']
        cleft_label = labels['cleft_label']
        cleft_label2 = labels['cleft_label2']

        # Original coordinates 
        cx, cy, cz = central_coord

        # Define a large temporary region to find presynapse components
        # This region should be larger than the final bounding box to allow for shifting
        temp_half_size = subvolume_size  # Double the size for initial analysis
        temp_x_start = max(cx - temp_half_size, 0)
        temp_x_end = min(cx + temp_half_size, raw_vol.shape[2])
        temp_y_start = max(cy - temp_half_size, 0)
        temp_y_end = min(cy + temp_half_size, raw_vol.shape[1])
        temp_z_start = max(cz - temp_half_size, 0)
        temp_z_end = min(cz + temp_half_size, raw_vol.shape[0])

        # Find vesicles in the expanded region
        vesicle_full_mask = (add_mask_vol == vesicle_label)
        temp_vesicle_mask = get_closest_component_mask(
            vesicle_full_mask,
            temp_z_start, temp_z_end,
            temp_y_start, temp_y_end,
            temp_x_start, temp_x_end,
            (cx, cy, cz)
        )

        def create_segment_masks(segmentation_volume, s1_coord, s2_coord):
            x1, y1, z1 = s1_coord
            x2, y2, z2 = s2_coord
            seg_id_1 = segmentation_volume[z1, y1, x1]
            seg_id_2 = segmentation_volume[z2, y2, x2]
            mask_1 = (segmentation_volume == seg_id_1) if seg_id_1 != 0 else np.zeros_like(segmentation_volume, dtype=bool)
            mask_2 = (segmentation_volume == seg_id_2) if seg_id_2 != 0 else np.zeros_like(segmentation_volume, dtype=bool)
            return mask_1, mask_2

        mask_1_full, mask_2_full = create_segment_masks(seg_vol, side1_coord, side2_coord)

        # Determine which side is the presynapse by checking overlap with vesicle mask
        overlap_side1 = np.sum(np.logical_and(mask_1_full, temp_vesicle_mask))
        overlap_side2 = np.sum(np.logical_and(mask_2_full, temp_vesicle_mask))
        presynapse_side = 1 if overlap_side1 > overlap_side2 else 2
        
        # Get presynapse mask
        presynapse_mask_full = mask_1_full if presynapse_side == 1 else mask_2_full
        
        # Calculate the final bounding box with possibly adjusted center
        half_size = subvolume_size // 2
        x_start = max(cx - half_size, 0)
        x_end = min(cx + half_size, raw_vol.shape[2])
        y_start = max(cy - half_size, 0)
        y_end = min(cy + half_size, raw_vol.shape[1])
        z_start = max(cz - half_size, 0)
        z_end = min(cz + half_size, raw_vol.shape[0])

        # Get vesicle mask for the final bounding box region
        vesicle_mask = get_closest_component_mask(
            vesicle_full_mask,
            z_start, z_end,
            y_start, y_end,
            x_start, x_end,
            (cx, cy, cz)
        )

        # Process segmentation type using the extracted processor function
        combined_mask_full = process_segmentation_type(
            segmentation_type=segmentation_type,
            add_mask_vol=add_mask_vol,
            mask_1_full=mask_1_full,
            mask_2_full=mask_2_full,
            presynapse_side=presynapse_side,
            presynapse_mask_full=presynapse_mask_full,
            z_start=z_start, z_end=z_end,
            y_start=y_start, y_end=y_end,
            x_start=x_start, x_end=x_end,
            cx=cx, cy=cy, cz=cz,
            vesicle_label=vesicle_label,
            cleft_label=cleft_label,
            cleft_label2=cleft_label2,
            mito_label=mito_label,

        )
        
        # Handle special case where segmentation processor returns None (sample should be discarded)
        if combined_mask_full is None:
                return None

        # Regular processing for other segmentation types
        sub_raw = raw_vol[z_start:z_end, y_start:y_end, x_start:x_end]
        sub_combined_mask = combined_mask_full[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Extract the presynapse mask for the final bounding box
        sub_presynapse_mask = presynapse_mask_full[z_start:z_end, y_start:y_end, x_start:x_end]
        

        # Apply padding if needed
        pad_z = subvolume_size - sub_raw.shape[0]
        pad_y = subvolume_size - sub_raw.shape[1]
        pad_x = subvolume_size - sub_raw.shape[2]
        if pad_z > 0 or pad_y > 0 or pad_x > 0:
            sub_raw = np.pad(sub_raw, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
            sub_combined_mask = np.pad(sub_combined_mask, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=False)

        sub_raw = sub_raw[:subvolume_size, :subvolume_size, :subvolume_size]
        sub_combined_mask = sub_combined_mask[:subvolume_size, :subvolume_size, :subvolume_size]

        sub_raw = sub_raw.astype(np.float32)
        
        # Apply normalization across the entire volume or per slice
        if normalize_across_volume:
            # Global normalization across the entire volume
            min_val = np.min(sub_raw)
            max_val = np.max(sub_raw)
            range_val = max_val - min_val if max_val > min_val else 1.0
            normalized = (sub_raw - min_val) / range_val
            
            # Print for debugging
            # print(f"Global normalization: min={min_val:.4f}, max={max_val:.4f}, range={range_val:.4f}")
        else:
            # Original per-slice normalization
            mins = np.min(sub_raw, axis=(1, 2), keepdims=True)
            maxs = np.max(sub_raw, axis=(1, 2), keepdims=True)
            ranges = np.where(maxs > mins, maxs - mins, 1.0)
            normalized = (sub_raw - mins) / ranges
            
            # Print for debugging
            # print(f"Per-slice normalization: shape of mins={mins.shape}, maxs={maxs.shape}")

        # Convert to RGB here ONLY for visualization purposes
        # The data processing pipeline uses grayscale (1-channel) format
        raw_rgb = np.repeat(normalized[..., np.newaxis], 3, axis=-1)
        mask_factor = sub_combined_mask[..., np.newaxis]

        if alpha < 1:
            blended_part = alpha * self.gray_color + (1 - alpha) * raw_rgb
        else:
            blended_part = self.gray_color * (1 - mask_factor) + raw_rgb * mask_factor

        overlaid_image = raw_rgb * mask_factor + (1 - mask_factor) * blended_part

        overlaid_cube = np.transpose(overlaid_image, (1, 2, 3, 0))

        return overlaid_cube
