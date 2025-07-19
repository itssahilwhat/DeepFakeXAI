import os
from src.dataset import DeepfakeDataset

def test_dataset_loads():
    ds = DeepfakeDataset('train', img_size=224, aug_strong=False)
    assert len(ds) > 0, 'Dataset is empty!'
    img, mask, label = ds[0]
    assert img.shape[1:] == (224, 224), 'Image shape mismatch'
    assert mask.shape[1:] == (224, 224), 'Mask shape mismatch'
    assert label in [0, 1], 'Label not 0 or 1' 