import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn.utils.prune as prune
from src.config import Config
from src.model import MultiTaskDeepfakeModel

def main():
    model = MultiTaskDeepfakeModel(
        backbone_name=Config.BACKBONE,
        num_classes=Config.NUM_CLASSES,
        pretrained=False,
        dropout=Config.DROPOUT,
        segmentation=Config.SEGMENTATION,
        attention=Config.ATTENTION
    )
    model.load_state_dict(torch.load(os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth'), map_location='cpu'))
    model.eval()

    # Apply global unstructured pruning to all Conv2d and Linear layers
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.5  # Prune 50% of weights globally
    )
    # Remove pruning re-parametrization so the model can be saved/exported
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')

    pruned_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model_pruned.pth')
    torch.save(model.state_dict(), pruned_path)
    print(f'Pruned model saved to {pruned_path}')

if __name__ == "__main__":
    main() 