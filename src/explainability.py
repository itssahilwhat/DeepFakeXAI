# src/explainability.py

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries


class GradCAM:
    """
    Implements Grad-CAM for visualizing where the model is "looking".
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = self._find_target_layer(target_layer)
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _find_target_layer(self, layer_name):
        # Find the target layer by its name
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer {layer_name} not found in model.")

    def _save_gradient(self, grad):
        self.gradients = grad

    def _save_activation(self, module, input, output):
        self.activations = output

    def _register_hooks(self):
        # Register hooks to capture the activations and gradients
        handle_act = self.target_layer.register_forward_hook(self._save_activation)
        handle_grad = self.target_layer.register_full_backward_hook(self._save_gradient)
        self.hook_handles.extend([handle_act, handle_grad])

    def __call__(self, x, class_idx=1):
        self.model.eval()
        # Forward pass
        cls_logits, _ = self.model(x)

        # Backward pass
        self.model.zero_grad()
        score = cls_logits[:, class_idx].sum()
        score.backward()

        # Generate CAM
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured.")

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()

        # Weight the channels by the gradients
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1.0

        return cv2.resize(heatmap, (x.shape[2], x.shape[3]))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()


class LimeExplainer:
    """
    Wrapper for LIME to generate superpixel-based explanations.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.explainer = lime_image.LimeImageExplainer()

    def _predict_fn(self, images):
        # LIME provides numpy arrays, we need to convert them to tensors
        self.model.eval()
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(self.device)
        with torch.no_grad():
            logits, _ = self.model(images)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def explain(self, image_tensor, num_features=10, num_samples=100):
        # Convert single tensor back to numpy for LIME
        image_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

        explanation = self.explainer.explain_instance(
            image_np,
            self._predict_fn,
            top_labels=1,
            hide_color=0,
            num_features=num_features,
            num_samples=num_samples
        )
        return explanation