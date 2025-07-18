import argparse
import os

class SynapseConfig:
    def __init__(self):
        self.raw_base_dir = '/teamspace/studios/this_studio/SynapseClusterEM/data/raw'
        self.seg_base_dir = '/teamspace/studios/this_studio/SynapseClusterEM/data/seg'
        self.add_mask_base_dir = '/teamspace/studios/this_studio/SynapseClusterEM/data/vesicle_cloud__syn_interface__mitochondria_annotation'
        self.bbox_name = ['bbox1','bbox2','bbox3','bbox4','bbox5','bbox6','bbox7']
        # self.bbox_name = ['bbox1']
        self.excel_file = '/teamspace/studios/this_studio/SynapseClusterEM/data/'
        self.csv_output_dir = 'results/csv_outputs'
        self.size = (80, 80)
        self.subvol_size = 80
        self.num_frames = 80
        self.save_gifs_dir = 'results/gifs'
        self.alpha = 1.0
        self.segmentation_type = 11
        # self.segmentation_type = [12, 10,11]  
        self.gray_color = 0.6
        
        self.clustering_output_dir = 'results/clustering_results_final'
        self.report_output_dir = 'results/comprehensive_reports'
        
        # Connectome dataset parameters
        self.connectome_num_samples = 10  # Total number of samples to read from connectome
        self.connectome_batch_size = 10  # Batch size for model inference
        
        # Clustering parameters
        self.clustering_algorithm = 'KMeans'  # Default clustering algorithm
        self.n_clusters = 10 # Default number of clusters for KMeans
        self.dbscan_eps = 0.5  # Default epsilon parameter for DBSCAN
        self.dbscan_min_samples = 5  # Default min_samples parameter for DBSCAN
        self.results_dir='results'
        self.model_path='hemibrain_production.checkpoint'
        
        # Feature extraction parameters
        self.extraction_method = "stage_specific"  # Options: "standard" or "stage_specific"
        self.layer_num = 20  # Layer to extract features from when using stage_specific method
        self.preprocessing = 'normal'  # Options: 'normal' or 'intelligent_cropping'
        self.preprocessing_weights = 0.7 # it has opitons like 0.3 0.5 and 0.7
    def parse_args(self):   
        parser = argparse.ArgumentParser(description="Synapse Dataset Configuration")
        parser.add_argument('--raw_base_dir', type=str, default=self.raw_base_dir)
        parser.add_argument('--seg_base_dir', type=str, default=self.seg_base_dir)
        parser.add_argument('--add_mask_base_dir', type=str, default=self.add_mask_base_dir)
        parser.add_argument('--bbox_name', type=str, default=self.bbox_name, nargs='+')
        parser.add_argument('--excel_file', type=str, default=self.excel_file)
        parser.add_argument('--csv_output_dir', type=str, default=self.csv_output_dir)
        parser.add_argument('--size', type=tuple, default=self.size)
        parser.add_argument('--subvol_size', type=int, default=self.subvol_size)
        parser.add_argument('--num_frames', type=int, default=self.num_frames)
        parser.add_argument('--save_gifs_dir', type=str, default=self.save_gifs_dir)
        parser.add_argument('--alpha', type=float, default=self.alpha)
        parser.add_argument('--segmentation_type', type=int, default=self.segmentation_type, 
                           choices=range(0, 13), help='Type of segmentation overlay')
        parser.add_argument('--gray_color', type=float, default=self.gray_color,
                           help='Gray color value (0-1) for overlaying segmentation')
        parser.add_argument('--clustering_output_dir', type=str, default=self.clustering_output_dir)
        parser.add_argument('--report_output_dir', type=str, default=self.report_output_dir)
        
        # Connectome dataset parameters
        parser.add_argument('--connectome_num_samples', type=int, default=self.connectome_num_samples,
                           help='Total number of samples to read from connectome dataset')
        parser.add_argument('--connectome_batch_size', type=int, default=self.connectome_batch_size,
                           help='Batch size for model inference (how many samples to process at once)')
        
        # Clustering parameters
        parser.add_argument('--clustering_algorithm', type=str, default=self.clustering_algorithm,
                           choices=['KMeans', 'DBSCAN'], help='Clustering algorithm to use')
        parser.add_argument('--n_clusters', type=int, default=self.n_clusters,
                           help='Number of clusters for KMeans')
        parser.add_argument('--dbscan_eps', type=float, default=self.dbscan_eps,
                           help='Epsilon parameter for DBSCAN')
        parser.add_argument('--dbscan_min_samples', type=int, default=self.dbscan_min_samples,
                           help='Minimum samples parameter for DBSCAN')
        
        # Feature extraction parameters
        parser.add_argument('--extraction_method', type=str, default=self.extraction_method,
                           choices=['standard', 'stage_specific'], 
                           help='Method to extract features ("standard" or "stage_specific")')
        parser.add_argument('--layer_num', type=int, default=self.layer_num,
                           help='Layer number to extract features from when using stage_specific method')
        
        args, _ = parser.parse_known_args()
        
        for key, value in vars(args).items():
            setattr(self, key, value)
        
        return self
    
    
config = SynapseConfig()