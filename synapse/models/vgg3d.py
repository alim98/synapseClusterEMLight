import logging
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# NOTE: Default parameters have been modified from original implementation to match 
# our existing hemibrain_production.checkpoint. The checkpoint was trained with:
# https://github.com/funkelab/funlib.learn.torch/blob/master/funlib/learn/torch/models/vgg3d.py
# - fmaps=24 (modified from original)
# - n_convolutions=(4,2,2,2) (modified from original)
# - output_classes=7 (modified from original)
# These defaults ensure compatibility with the pretrained model while maintaining the new file structure.


class Vgg3D(torch.nn.Module):
    def __init__(
        self,
        input_size=(80, 80, 80),
        fmaps=24,
        downsample_factors=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
        fmap_inc=(2, 2, 2, 2),
        n_convolutions=(4, 2, 2, 2),
        output_classes=7,
        input_fmaps=1,
    ):
        if len(downsample_factors) != len(fmap_inc):
            raise ValueError("fmap_inc needs to have same length as downsample factors")
        if len(n_convolutions) != len(fmap_inc):
            raise ValueError(
                "n_convolutions needs to have the same length as downsample factors"
            )
        if np.any(np.array(n_convolutions) < 1):
            raise ValueError("Each layer must have at least one convolution")

        super(Vgg3D, self).__init__()

        current_fmaps = input_fmaps
        current_size = tuple(input_size)

        features = []
        for i in range(len(downsample_factors)):
            features += [
                torch.nn.Conv3d(current_fmaps, fmaps, kernel_size=3, padding=1),
                torch.nn.BatchNorm3d(fmaps),
                torch.nn.ReLU(inplace=True),
            ]

            for n in range(n_convolutions[i] - 1):
                features += [
                    torch.nn.Conv3d(fmaps, fmaps, kernel_size=3, padding=1),
                    torch.nn.BatchNorm3d(fmaps),
                    torch.nn.ReLU(inplace=True),
                ]

            features += [torch.nn.MaxPool3d(downsample_factors[i])]

            current_fmaps = fmaps
            fmaps *= fmap_inc[i]

            size = tuple(
                int(c / d) for c, d in zip(current_size, downsample_factors[i])
            )
            check = (
                s * d == c for s, d, c in zip(size, downsample_factors[i], current_size)
            )
            assert all(check), "Can not downsample %s by chosen downsample factor" % (
                current_size,
            )
            current_size = size

            logger.info("VGG level %d: (%s), %d fmaps", i, current_size, current_fmaps)

        self.features = torch.nn.Sequential(*features)

        classifier = [
            torch.nn.Linear(
                current_size[0] * current_size[1] * current_size[2] * current_fmaps,
                4096,
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, output_classes),
        ]

        self.classifier = torch.nn.Sequential(*classifier)

    def forward(self, raw):
        """
        expects raw to have both a batch and a channel dimension
        """

        f = self.features(raw)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)

        return y

def load_model_from_checkpoint(model, checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        print(f"Model loaded from {checkpoint_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return model