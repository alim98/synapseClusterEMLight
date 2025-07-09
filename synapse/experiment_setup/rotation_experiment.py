# rotation_experiment.py
"""Rotation Embedding Drift Experiment

Run:
python -m synapse.experiment_setup.rotation_experiment \
       --angles "0,15,30,45,60" \
       --samples_per_bbox 5 \
       --save_gifs"""
import os
import sys
import argparse
import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
import h5py
from tqdm import tqdm
from scipy import ndimage
from sklearn.metrics import pairwise_distances
import umap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import torch
from sklearn.decomposition import PCA

from synapse.utils.config import config  
from synapse.dl.dataloader import SynapseDataLoader, Synapse3DProcessor
from synapse.dl.dataset import SynapseDataset

from inference_local import (
    VGG3D as load_vgg3d,
    extract_features,
    extract_stage_specific_features,
)

# ----------------------------------------------------------------------------
# ----------------------------- Helper functions -----------------------------
# ----------------------------------------------------------------------------

def setup_logger(log_path: str):
    """Setup python logging to both file and stdout."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    # stderr
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # file
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)




def create_gif_from_volume(
    vol: np.ndarray,
    gif_path: str,
    plane: str = "z",
    sample_frames: int = 60,
):
    """Create a lightweight GIF from *vol*.

    Instead of dumping every slice, we pick *sample_frames* equally-spaced slices along *plane*.
    Assumes *vol* shape (Z, Y, X)."""
    import imageio.v3 as iio

    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    plane = plane.lower()
    if plane not in {"x", "y", "z"}:
        raise ValueError("plane must be one of 'x', 'y', 'z'")

    # Determine axis and slice indices to sample
    axis = {"z": 0, "y": 1, "x": 2}[plane]
    total_slices = vol.shape[axis]
    sample_frames = min(total_slices, sample_frames)
    indices = np.linspace(0, total_slices - 1, sample_frames, dtype=int)

    frames = []
    for idx in indices:
        if axis == 0:  # z
            frame = vol[idx]
        elif axis == 1:  # y
            frame = vol[:, idx, :]
        else:  # x
            frame = vol[:, :, idx]
        frames.append(frame)

    frames = [(f - vol.min()) / (vol.max() - vol.min() + 1e-5) for f in frames]
    frames_uint8 = [np.uint8(frame * 255) for frame in frames]
    iio.imwrite(gif_path, frames_uint8, format="GIF", duration=0.02)


# ----------------------------------------------------------------------------
# ---------------------- Volume caching utilities ---------------------------
# ----------------------------------------------------------------------------


def cache_bbox_path(cache_dir: str, bbox: str) -> str:
    """Return path to HDF5 cache file for given bbox."""
    return os.path.join(cache_dir, f"{bbox}.h5")


def write_bbox_h5(cache_path: str, raw: np.ndarray, seg: np.ndarray, add: np.ndarray):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with h5py.File(cache_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("seg", data=seg, compression="gzip")
        f.create_dataset("add_mask", data=add, compression="gzip")


def read_bbox_h5(cache_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(cache_path, "r") as f:
        raw = f["raw"][:]
        seg = f["seg"][:]
        add = f["add_mask"][:]
    return raw, seg, add


def load_volumes_with_cache(
    bboxes: List[str],
    data_loader: SynapseDataLoader,
    cache_dir: str,
    force_refresh: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load volumes, using HDF5 cache when available."""
    vol_dict = {}
    for bbox in tqdm(bboxes, desc="Volume load (with cache)"):
        h5_path = cache_bbox_path(cache_dir, bbox)
        if os.path.exists(h5_path) and not force_refresh:
            try:
                raw_v, seg_v, add_v = read_bbox_h5(h5_path)
                vol_dict[bbox] = (raw_v, seg_v, add_v)
                continue
            except Exception as e:
                logging.warning(f"Failed reading cache for {bbox}: {e}. Re-loading from source…")

        # Load from original source (tif stack) via existing loader
        rv, sv, av = data_loader.load_volumes(bbox)
        if rv is None:
            logging.warning(f"Skipping {bbox} – failed to load volumes")
            continue
        vol_dict[bbox] = (rv, sv, av)
        try:
            write_bbox_h5(h5_path, rv, sv, av)
        except Exception as e:
            logging.warning(f"Could not write cache for {bbox}: {e}")
    return vol_dict


# -------------------------------------------------------------------------
# -------------------- Synapse DataFrame caching --------------------------
# -------------------------------------------------------------------------


def load_synapse_df_with_cache(
    bbox_names: List[str],
    excel_dir: str,
    cache_path: str,
    force_refresh: bool = False,
):
    """Load concatenated synapse Excel sheets with optional HDF5 caching."""
    if os.path.exists(cache_path) and not force_refresh:
        try:
            logging.info(f"Loading synapse dataframe from cache {cache_path}")
            cached_df = pd.read_hdf(cache_path, key="synapse")
            # Verify that the cache contains **all** requested bboxes.
            cached_bboxes = set(cached_df.get("bbox_name", []) if "bbox_name" in cached_df.columns else [])
            requested_bboxes = set(bbox_names)
            if requested_bboxes.issubset(cached_bboxes):
                return cached_df
            else:
                missing = requested_bboxes - cached_bboxes
                logging.info(
                    f"Synapse cache missing bboxes {sorted(missing)} – rebuilding cache …"
                )
        except Exception as e:
            logging.warning(f"Failed reading synapse cache: {e}. Re-reading excels…")

    dfs = []
    for bbox in bbox_names:
        xlsx_path = os.path.join(excel_dir, f"{bbox}.xlsx")
        if os.path.exists(xlsx_path):
            dfs.append(pd.read_excel(xlsx_path).assign(bbox_name=bbox))
        else:
            logging.warning(f"Excel metadata not found for {bbox} at {xlsx_path}")

    if not dfs:
        raise RuntimeError("No synapse Excel files were found; cannot proceed.")

    syn_df = pd.concat(dfs, ignore_index=True)

    # Write cache
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        syn_df.to_hdf(cache_path, key="synapse", mode="w", complevel=5, complib="zlib")
        logging.info(f"Synapse dataframe cached to {cache_path} (shape {syn_df.shape})")
    except Exception as e:
        logging.warning(f"Could not write synapse cache: {e}")

    return syn_df


# ----------------------------------------------------------------------------
# -------------------------- Embedding helper --------------------------------
# ----------------------------------------------------------------------------


def compute_2d_embedding(features: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Return a (N,2) 2-D UMAP embedding.

    The previous implementation switched to PCA for tiny datasets, but
    we now want to *always* apply UMAP.  UMAP requires
    ``n_neighbors >= 2`` and ``n_neighbors < n_samples`` so we clip the
    value accordingly.
    """
    n_samples = features.shape[0]
    if n_samples < 2:
        raise ValueError("UMAP embedding needs at least 2 samples – got " + str(n_samples))

    # Ensure n_neighbors is within valid range [2, n_samples-1]
    n_neighbors = max(2, min(15, n_samples - 1))
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=random_state)
    return reducer.fit_transform(features)


# ----------------------------------------------------------------------------
# --------------------- Cube-level rotation dataset -------------------------
# ----------------------------------------------------------------------------


class RotatedCubeDataset(torch.utils.data.Dataset):
    """Wrap a SynapseDataset and rotate each 3-D cube after cropping.

    The wrapped dataset must return a tuple ``(pixel_values, syn_info, bbox_name)``
    where ``pixel_values`` is a tensor with shape (1, D, H, W) – exactly what
    ``SynapseDataset`` currently outputs.  We rotate every XY slice in the cube
    by *angle_deg* around the Z-axis.
    """

    def __init__(self, base_ds: torch.utils.data.Dataset, angle_deg: float, fill_val: float):
        self.base = base_ds
        self.angle = angle_deg
        self.fill_val = fill_val

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        pix, syn_info, bbox_name = self.base[idx]

        # Convert to numpy for scipy rotation. pix shape: (num_frames, 1, H, W)
        vol = pix.cpu().numpy()

        # Rotate in-plane (H, W) for every frame & channel at once using ndimage.rotate
        rot_vol_big = ndimage.rotate(
            vol,
            self.angle,
            axes=(-2, -1),  # rotate over H and W axes
            reshape=True,   # allow expansion so nothing is lost
            order=1,
            mode="constant",
            cval=self.fill_val,
        )

        # Center-crop (or pad) back to original H×W
        orig_h, orig_w = vol.shape[-2:]
        big_h, big_w = rot_vol_big.shape[-2:]

        # Compute crop bounds
        top = max((big_h - orig_h) // 2, 0)
        left = max((big_w - orig_w) // 2, 0)
        bottom = top + orig_h
        right = left + orig_w

        rot_cropped = rot_vol_big[..., top:bottom, left:right]

        # If rotated volume is still smaller (rare), pad with fill_val
        pad_h = orig_h - rot_cropped.shape[-2]
        pad_w = orig_w - rot_cropped.shape[-1]
        if pad_h > 0 or pad_w > 0:
            pad_before_h = pad_h // 2
            pad_after_h = pad_h - pad_before_h
            pad_before_w = pad_w // 2
            pad_after_w = pad_w - pad_before_w
            rot_cropped = np.pad(
                rot_cropped,
                ((0, 0), (0, 0), (pad_before_h, pad_after_h), (pad_before_w, pad_after_w)),
                mode="constant",
                constant_values=self.fill_val,
            )

        rot_pix = torch.from_numpy(rot_cropped).type_as(pix)
        return rot_pix, syn_info, bbox_name


# ----------------------------------------------------------------------------
# ------------------------------ Core pipeline -------------------------------
# ----------------------------------------------------------------------------


def main(args):

    original_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0]]
        config.parse_args()  
    finally:
        sys.argv = original_argv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_results_dir = os.path.join(config.results_dir, "rotation_experiment")
    os.makedirs(base_results_dir, exist_ok=True)
    setup_logger(os.path.join(base_results_dir, "experiment.log"))
    logging.info(f"Running rotation experiment on device: {device}")

    # Angles
    angles: List[float] = [float(a) for a in args.angles.split(",")] if isinstance(args.angles, str) else args.angles
    angles = sorted(set(angles))  # unique & sorted
    logging.info(f"Angles to evaluate: {angles}")

    # ------------------------------------------------------------------
    # Load data (575³ volumes + meta dataframe) -------------------------
    # ------------------------------------------------------------------
    logging.info("Loading raw / seg volumes and synapse metadata… (with HDF5 cache)")
    data_loader = SynapseDataLoader(
        raw_base_dir=config.raw_base_dir,
        seg_base_dir=config.seg_base_dir,
        add_mask_base_dir=config.add_mask_base_dir,
        gray_color=config.gray_color,
    )

    cache_dir = os.path.join(config.results_dir, "h5_cache")
    vol_data_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = load_volumes_with_cache(
        bboxes=config.bbox_name,
        data_loader=data_loader,
        cache_dir=cache_dir,
        force_refresh=args.refresh_cache,
    )

    synapse_cache_path = os.path.join(cache_dir, "synapse_df.h5")
    syn_df = load_synapse_df_with_cache(
        bbox_names=config.bbox_name,
        excel_dir=config.excel_file,
        cache_path=synapse_cache_path,
        force_refresh=args.refresh_cache,
    )
    # Optional per-bbox subsampling ------------------------------------
    if args.samples_per_bbox is not None and args.samples_per_bbox > 0:
        logging.info(f"Subsampling up to {args.samples_per_bbox} synapses per bbox …")
        syn_df = (
            syn_df.groupby("bbox_name", group_keys=False)
            .apply(lambda df: df.sample(n=min(len(df), args.samples_per_bbox), random_state=42))
            .reset_index(drop=True)
        )
        logging.info(f"Synapse dataframe after subsampling: {syn_df.shape}")
    logging.info(f"Synapse dataframe ready with shape {syn_df.shape}")

    # ------------------------------------------------------------------
    # Model & processor -------------------------------------------------
    # ------------------------------------------------------------------
    model = load_vgg3d().to(device).eval()
    processor = Synapse3DProcessor(size=config.size)

    # ------------------------------------------------------------------
    # Iterate over rotation angles -------------------------------------
    # ------------------------------------------------------------------
    per_angle_feature_paths: Dict[float, str] = {}
    combined_features: List[pd.DataFrame] = []
    baseline_feat_df: Optional[pd.DataFrame] = None  # stores features for 0°

    for angle in angles:
        logging.info(f"\n==== Processing angle = {angle}° ====")

        # --------------------------------------
        # Build dataset (rotate cubes, not volumes) -------------------
        # --------------------------------------
        base_dataset = SynapseDataset(
            vol_data_dict=vol_data_dict,  # unrotated volumes
            synapse_df=syn_df,
            processor=processor,
            segmentation_type=config.segmentation_type,
            subvol_size=config.subvol_size,
            num_frames=config.num_frames,
            alpha=config.alpha,
        )

        dataset = base_dataset if angle == 0 else RotatedCubeDataset(base_dataset, angle, config.gray_color)

        if config.extraction_method == "stage_specific":
            feat_df = extract_stage_specific_features(
                model,
                dataset,
                config,
                layer_num=config.layer_num,
                pooling_method="avg",
            )
        else:
            feat_df = extract_features(
                model,
                dataset,
                config,
                pooling_method="avg",
            )

        # Add metadata
        feat_df = feat_df.copy()
        feat_df["angle"] = angle
        # ----------------------------------------------------------------
        # Unique patch id: strictly bbox + Var1 (synapse identifier)
        # ----------------------------------------------------------------
        if {"bbox", "Var1"}.issubset(feat_df.columns):
            feat_df["patch_id"] = (
                feat_df["bbox"].astype(str).str.strip()
                + "_"
                + feat_df["Var1"].astype(str).str.strip()
            )
        else:
            missing = {"bbox", "Var1"} - set(feat_df.columns)
            raise ValueError(
                f"Required columns for patch_id generation missing: {missing}. "
                "Ensure synapse dataframe contains both 'bbox' and 'Var1'."
            )

        # Persist per-angle CSV
        angle_csv = os.path.join(base_results_dir, f"features_angle{int(angle)}.csv")
        feat_df.to_csv(angle_csv, index=False)
        per_angle_feature_paths[angle] = angle_csv
        combined_features.append(feat_df)
        logging.info(f"Saved features to {angle_csv}")

        # ------------------------------------------------------------
        # Per-angle UMAP                                               
        # ------------------------------------------------------------
        angle_feature_cols = [c for c in feat_df.columns if c.startswith("feat_") or c.startswith("layer")]
        angle_features_mat = feat_df[angle_feature_cols].values
        angle_emb = compute_2d_embedding(angle_features_mat, random_state=42)
        angle_umap_path = os.path.join(base_results_dir, f"umap_angle_{int(angle)}.png")
        plt.figure(figsize=(6, 5))
        plt.scatter(angle_emb[:, 0], angle_emb[:, 1], s=5, alpha=0.8, c="orange")
        plt.title(f"UMAP – angle {angle}°")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.tight_layout()
        plt.savefig(angle_umap_path, dpi=300)
        plt.close()
        logging.info(f"Per-angle UMAP saved to {angle_umap_path}")

        # ------------------------------------------------------------
        # Comparison UMAP vs baseline (0°)                            
        # ------------------------------------------------------------
        if angle == 0:
            baseline_feat_df = feat_df.copy()
        elif baseline_feat_df is not None:
            comp_df = pd.concat([baseline_feat_df, feat_df], ignore_index=True)
            comp_features_mat = comp_df[angle_feature_cols].values
            comp_emb = compute_2d_embedding(comp_features_mat, random_state=42)
            comp_df["umap_x"], comp_df["umap_y"] = comp_emb[:, 0], comp_emb[:, 1]

            comp_plot_path = os.path.join(base_results_dir, f"umap_compare_angle{int(angle)}_vs0.png")
            plt.figure(figsize=(6, 5))
            # Scatter baseline (blue) and rotated angle (red)
            colors = comp_df["angle"].apply(lambda a: "red" if a == angle else "blue")
            plt.scatter(comp_df["umap_x"], comp_df["umap_y"], c=colors, s=5, alpha=0.8)

            # Draw grey lines linking corresponding patch IDs
            baseline_pts = comp_df[comp_df["angle"] == 0].set_index("patch_id")
            angle_pts = comp_df[comp_df["angle"] == angle].set_index("patch_id")
            common_pids = baseline_pts.index.intersection(angle_pts.index)
            for pid in common_pids:
                x_coords = [baseline_pts.loc[pid, "umap_x"], angle_pts.loc[pid, "umap_x"]]
                y_coords = [baseline_pts.loc[pid, "umap_y"], angle_pts.loc[pid, "umap_y"]]
                plt.plot(x_coords, y_coords, c="grey", alpha=0.3, linewidth=0.7)

            blue_patch = Line2D([], [], marker='o', color='w', label='0°', markerfacecolor='blue', markersize=6)
            red_patch = Line2D([], [], marker='o', color='w', label=f'{angle}°', markerfacecolor='red', markersize=6)
            plt.legend(handles=[blue_patch, red_patch], loc='best')
            plt.xlabel("UMAP-1")
            plt.ylabel("UMAP-2")
            plt.title(f"UMAP comparison: 0° vs {angle}°")
            plt.tight_layout()
            plt.savefig(comp_plot_path, dpi=300)
            plt.close()
            logging.info(f"Comparison UMAP saved to {comp_plot_path}")

        # -----------------------------
        # Optional GIF of first cube --
        # -----------------------------
        if args.save_gifs and len(dataset) > 0:
            try:
                sample_pix, _, _ = dataset[0]
                cube_np = sample_pix[:, 0].cpu().numpy()  # (D, H, W)
                gif_dir = os.path.join(config.save_gifs_dir, f"angle_{int(angle)}")
                os.makedirs(gif_dir, exist_ok=True)
                gif_path = os.path.join(gif_dir, "sample_cube.gif")
                create_gif_from_volume(cube_np, gif_path, plane="z", sample_frames=args.gif_frames)
                logging.info(f"Subvolume GIF written to {gif_path}")
            except Exception as e:
                logging.warning(f"Could not save subvolume GIF for angle {angle}: {e}")

    # Combine & Analyse 
    logging.info("Combining feature dataframes …")
    combined_df = pd.concat(combined_features, ignore_index=True)
    combined_csv = os.path.join(base_results_dir, "features_all_angles.csv")
    combined_df.to_csv(combined_csv, index=False)
    logging.info(f"Combined feature CSV written to {combined_csv} (shape {combined_df.shape})")

    # Feature matrix
    feature_cols = [c for c in combined_df.columns if c.startswith("feat_") or c.startswith("layer")]
    features_mat = combined_df[feature_cols].values

    # -----------------------------------
    # UMAP projection -------------------
    # -----------------------------------
    logging.info("Computing UMAP embedding …")
    umap_emb = compute_2d_embedding(features_mat, random_state=42)
    combined_df["umap_x"], combined_df["umap_y"] = umap_emb[:, 0], umap_emb[:, 1]

    # Plot
    logging.info("Saving UMAP scatter plot …")
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        combined_df["umap_x"],
        combined_df["umap_y"],
        c=combined_df["angle"],
        cmap="hsv",
        s=5,
        alpha=0.8,
    )
    plt.colorbar(scatter, label="Rotation angle (°)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("Embedding drift across rotation angles")
    umap_plot_path = os.path.join(base_results_dir, "umap_angles.png")
    plt.tight_layout()
    plt.savefig(umap_plot_path, dpi=300)
    plt.close()
    logging.info(f"UMAP plot saved to {umap_plot_path}")

    # -----------------------------------
    # Pairwise distance stats -----------
    # -----------------------------------
    logging.info("Computing pairwise distances relative to 0° baseline …")
    baseline_df = combined_df[combined_df["angle"] == 0].set_index("patch_id")
    dist_records = []

    for angle in angles:
        if angle == 0:
            continue
        angle_df = combined_df[combined_df["angle"] == angle].set_index("patch_id")
        common_ids = baseline_df.index.intersection(angle_df.index)
        if len(common_ids) == 0:
            logging.warning(f"No overlapping patch IDs between baseline and angle {angle}")
            continue
        base_feats = baseline_df.loc[common_ids, feature_cols].values
        rot_feats = angle_df.loc[common_ids, feature_cols].values

        # Cosine & Euclidean distances
        cos_dists = pairwise_distances(base_feats, rot_feats, metric="cosine").diagonal()
        euc_dists = np.linalg.norm(base_feats - rot_feats, axis=1)

        summary = {
            "angle": angle,
            "n_pairs": len(cos_dists),
            "cosine_mean": float(np.mean(cos_dists)),
            "cosine_std": float(np.std(cos_dists)),
            "euclidean_mean": float(np.mean(euc_dists)),
            "euclidean_std": float(np.std(euc_dists)),
        }

        # -------------------------------------------------
        # Save per-angle distance summary  ---------------
        # -------------------------------------------------
        per_angle_csv = os.path.join(
            base_results_dir,
            f"pairwise_distance_summary_angle{int(angle)}.cse",
        )
        pd.DataFrame([summary]).to_csv(per_angle_csv, index=False)
        logging.info(f"Pairwise distance summary for angle {angle} saved to {per_angle_csv}")

        dist_records.append(summary)

    dist_df = pd.DataFrame(dist_records)
    dist_csv = os.path.join(base_results_dir, "pairwise_distance_summary_all.cse")
    dist_df.to_csv(dist_csv, index=False)
    logging.info(f"Pairwise distance summary saved to {dist_csv}\n{dist_df}")

    logging.info("Rotation experiment completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-D rotation embedding drift experiment", allow_abbrev=False)
    parser.add_argument(
        "--angles",
        type=str,
        default="0,45,90, 120, 180, 270, 360",
        help="Comma separated list of rotation angles in degrees (e.g. \"0,15,30\")",
    )
    parser.add_argument(
        "--save_rotated_volumes",
        action="store_true",
        help="If set, rotated full volumes will be persisted to HDF5 for debugging.",
    )
    parser.add_argument(
        "--save_gifs",
        action="store_true",
        help="If set, creates quick GIF previews of the first bbox per angle.",
    )
    parser.add_argument(
        "--save_gifs_dir",
        type=str,
        default="synapse/experiment_setup/rotation_experiment/gifs",
        help="Directory to save GIFs.",
    )
    parser.add_argument(
        "--refresh_cache",
        action="store_true",
        help="Force re-loading raw data and overwrite existing HDF5 cache.",
    )
    parser.add_argument(
        "--samples_per_bbox",
        type=int,
        default=None,
        help="If set, randomly sample this many synapse entries per bbox for feature extraction (e.g. 5).",
    )
    parser.add_argument(
        "--gif_frames",
        type=int,
        default=60,
        help="Number of frames to sample for GIF creation.",
    )
    
    parsed_args = parser.parse_args()
    main(parsed_args) 