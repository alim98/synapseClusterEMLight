"""
Synapse package for synapse analysis and visualization.
"""

__version__ = '0.1.0'

# Import commonly used components
from synapse.models import Vgg3D, load_model_from_checkpoint
from synapse.dl.dataset import SynapseDataset
from synapse.dl.dataloader import Synapse3DProcessor, SynapseDataLoader
from synapse.utils import config

# Export the most commonly used components
__all__ = [
    'Vgg3D', 
    'load_model_from_checkpoint',
    'SynapseDataset',
    'Synapse3DProcessor',
    'SynapseDataLoader',
    'config',
    'create_gif_from_volume',
    
] 