import torch
from src.model import MultiTaskDeepfakeModel

def test_model_forward():
    model = MultiTaskDeepfakeModel(backbone_name='mobilenet_v3_small', num_classes=2, segmentation=True, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    cls_logits, seg_logits = model(x)
    assert cls_logits.shape == (2, 2), 'Classification logits shape mismatch'
    assert seg_logits.shape[0] == 2 and seg_logits.shape[2:] == (224, 224), 'Segmentation logits shape mismatch (should be upsampled to input size)' 