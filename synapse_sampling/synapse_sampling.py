import numpy as np
import h5py
import pandas as pd
from typing import Tuple, Iterable
from tqdm import tqdm
import os

from .src.bbox_reader import BboxReader
from .src.utils import filter_non_center_components, get_dataset_layer_mag, read_mag_bbox_data, get_point_mask_in_boxes, SynapseSamplingException


CONNECTOME_PATH = "/nexus/posix0/MBR-neuralsystems/vx/artifacts/alik-Merlin-6285_24-03-01-Sample-1A-Mar2023-full_v1/create_connectome/connectome__34fa6f477d-v1/connectome/connectome.hdf5"
SYNAPSE_PREDICTION_DATASET_PATH = path = "/nexus/posix0/MBR-neuralsystems/vx/artifacts/alik-Merlin-6285_24-03-01-Sample-1A-Mar2023-full_v1/predict/merlin_synapse_model_v8a_synapse_prediction_v1-v1/prediction"
AGGLO_PATH = '/nexus/posix0/MBR-neuralsystems/data/aligned/alik-Merlin-6285_24-03-01-Sample-1A-Mar2023-full_v2_nuclei_exclusion_with_meshes_full_dataset/segmentation/agglomerates/agglomerate_view_60.hdf5'
RAW_PATH = '/nexus/posix0/MBR-neuralsystems/data/aligned/alik-Merlin-6285_24-03-01-Sample-1A-Mar2023-full_extended_v2_nuclei_exclusion_with_meshes/'
SEG_PATH = '/nexus/posix0/MBR-neuralsystems/data/aligned/alik-Merlin-6285_24-03-01-Sample-1A-Mar2023-full_v2_nuclei_exclusion_with_meshes_full_dataset/'

# Local CSV containing curated synapse positions and presynaptic ids
LOCAL_CSV_PATH = 'SynapseClusterEM/data/absolute_positions_with_agglo_filtered.csv'

# Hardcoded plexiform bboxes from Dom
plexiform_bboxes = [
    [11775, 0,   6662,  7114, 12790, 2476], #for 8nm
    [11588, 0,   8966,  6780, 12699, 2684],
    [11339, 65,  11372, 6638, 12608, 2684],
    [10584, 179, 13828, 6814, 12494, 2684],
    [10033, 179, 16295, 7148, 12494, 2619],
    [11977, 65,  4129,  7577, 12484, 2670],
    [12500, 65,  1570,  7097, 12484, 2670],
    [12500, 65,  570,   7356, 12484, 2670]
]

# Global variables so we load them only once (256 MB total)
positions = None
agglo_ids = None

# cache for local csv
local_positions = None
local_agglo_ids = None

rng = np.random.default_rng(seed=37)

def sample_connectome(batch_size=1, policy="random", connectome_path=CONNECTOME_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample positions for synapse data.
    Args:
        batch_size (int): The number of positions to sample.
        policy (str): The sampling policy. Can be "random" 
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the sampled positions and agglomeration IDs (shapes: (batch_size, 3) and (batch_size,)).
    """

    global positions, agglo_ids, local_positions, local_agglo_ids

    if policy == "dummy":
        return np.zeros((batch_size, 3), dtype=np.uint32)

    # load only once    
    if positions is None:
        with h5py.File(connectome_path, 'r') as f:
            positions = f['synapse_positions'][:]
            agglo_ids = f['synapse_to_src_agglomerate'][:]
        
        mask = get_point_mask_in_boxes(positions, plexiform_bboxes)
        positions = positions[mask]
        agglo_ids = agglo_ids[mask]

    if policy == "random":
        # Sample random indices
        indices = rng.choice(positions.shape[0], size=batch_size, replace=False)
        
        # Get the sampled positions and agglomeration IDs
        return positions[indices], agglo_ids[indices]

    elif policy == "sequential": raise NotImplementedError("Sequential sampling is not implemented yet.")
    elif policy == "local":
        # lazy load csv once
        if local_positions is None:
            if not os.path.exists(LOCAL_CSV_PATH):
                raise FileNotFoundError(f"Local CSV not found at {LOCAL_CSV_PATH}")
            df = pd.read_csv(LOCAL_CSV_PATH)
            required_cols = {"abspos1", "abspos2", "abspos3", "agglo_id"}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"CSV must contain columns {required_cols}")
            local_positions = df[["abspos1", "abspos2", "abspos3"]].to_numpy(dtype=np.uint32)
            local_agglo_ids = df["agglo_id"].to_numpy(dtype=np.uint64)

        indices = rng.choice(local_positions.shape[0], size=batch_size, replace=False)
        return local_positions[indices], local_agglo_ids[indices]
    elif policy ==  "mixed":
        # ensure both sources loaded
        # load connectome positions if not yet
        if positions is None:
            with h5py.File(connectome_path, 'r') as f:
                positions = f['synapse_positions'][:]
                agglo_ids = f['synapse_to_src_agglomerate'][:]
            mask = get_point_mask_in_boxes(positions, plexiform_bboxes)
            positions = positions[mask]
            agglo_ids = agglo_ids[mask]

        # load local csv
        if local_positions is None:
            if not os.path.exists(LOCAL_CSV_PATH):
                raise FileNotFoundError(f"Local CSV not found at {LOCAL_CSV_PATH}")
            df = pd.read_csv(LOCAL_CSV_PATH)
            required_cols = {"abspos1", "abspos2", "abspos3", "agglo_id"}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"CSV must contain columns {required_cols}")
            local_positions = df[["abspos1", "abspos2", "abspos3"]].to_numpy(dtype=np.uint32)
            local_agglo_ids = df["agglo_id"].to_numpy(dtype=np.uint64)

        # concatenate both pools
        combined_positions = np.concatenate([positions, local_positions], axis=0)
        combined_agglos = np.concatenate([agglo_ids, local_agglo_ids], axis=0)

        indices = rng.choice(combined_positions.shape[0], size=batch_size, replace=False)
        return combined_positions[indices], combined_agglos[indices]
    else: raise ValueError(f"Unknown sampling policy: {policy}")


def get_synapse_data(position: Iterable, 
                     src_agglo_id: int, 
                     synapse_prediction_dataset_path = SYNAPSE_PREDICTION_DATASET_PATH,
                     raw_path = RAW_PATH,
                     seg_path = SEG_PATH,
                     agglo_path = AGGLO_PATH
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get synapse data for a given position and source agglomeration ID.
    Args:
        position (Iterable): The position in the format [x, y, z].
        src_agglo_id (int): The pre-synaptic agglomeration ID.
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the raw data and the mask of the pre-synaptic agglomeration and the cleft bot with shape (1, 80, 80, 80).
    """
    # get dataset layer
    layer_name = "segment_types_task$non-spine-head-synapse"
    mag = get_dataset_layer_mag(synapse_prediction_dataset_path, layer_name, 1)

    # get synapse mask
    bbox_size = np.array([80, 80, 80])
    center = position
    synapse = read_mag_bbox_data(mag, center, bbox_size)
    synapse_threshold = 100
    synapse_mask = synapse > synapse_threshold
    
    if synapse_mask.sum() == 0: raise SynapseSamplingException(f"No synapses are predicted above synapse threshold value: Threshold: {synapse_threshold}, Max value: {synapse.max()}. Maybe try a lower threshold value.")

    # get pre-synapse agglomeration mask
    reader = BboxReader(bbox_size=np.array([80, 80, 80]), bbox_unit_scale='vox', raw_path=raw_path, seg_path=seg_path, agglo_path=agglo_path)
    raw, agglo = reader.read_bbox_agglo_data(position, 1)
    pre_agglo_mask = (agglo == src_agglo_id).astype(np.uint8)

    
    # merge synapse mask with agglomeration
    full_mask = pre_agglo_mask | filter_non_center_components(synapse_mask.squeeze())[None, ...]

    return raw, full_mask

def sample_synapses(batch_size=1, policy="random", verbose=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample synapse raw em data and pre-synaptic agglomeration + cleft mask for a given batch size and policy.
    """
    if policy == "dummy":
        return np.zeros((batch_size, 1, 80, 80, 80), dtype=np.uint8), np.zeros((batch_size, 1, 80, 80, 80), dtype=np.uint8), np.zeros((batch_size, 3), dtype=np.uint32), np.zeros((batch_size, 1), dtype=np.uint64) 
    else:
        positions, agglo_ids = sample_connectome(batch_size, policy)
        raw = []
        mask = []
        if verbose: print(f"Reading and Masking {batch_size} synapse boxes...")
        wrapper = lambda x: tqdm(x, total=batch_size) if verbose else x
        for position, src_agglo_id in wrapper(zip(positions, agglo_ids)):
            budget = 100
            while budget > 0:
                try:
                    r, m = get_synapse_data(position, src_agglo_id)
                    budget = 0
                except SynapseSamplingException as e:
                    new_position, new_src_agglo_id = sample_connectome(1, policy)
                    position = new_position[0]
                    src_agglo_id = new_src_agglo_id[0]
                    budget -= 1
                    if budget == 0:
                        raise SynapseSamplingException(f"Failed to get synapse data after multiple repeated attempts. Last error: {e}")
            raw.append(r)
            mask.append(m)
        raw = np.stack(raw, axis=0)
        mask = np.stack(mask, axis=0)
        return raw, mask, positions, agglo_ids

if __name__ == "__main__":
    # Test the functions
    raw, mask = sample_synapses(batch_size=10, policy="random", verbose=True)
    print("Raw shape:", raw.shape)
    print("Mask shape:", mask.shape)