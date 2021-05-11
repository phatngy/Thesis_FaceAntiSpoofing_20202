import numpy as np
from torch.utils.data import Dataset
from .data_fusion import FDDataset

class Domain(FDDataset):
    def __init__(self, mode, modality, fold_index, image_size, augment, balance, ROI):
        super().__init__(mode, modality=modality, fold_index=fold_index, image_size=image_size, augment=augment, balance=balance, ROI=ROI)

