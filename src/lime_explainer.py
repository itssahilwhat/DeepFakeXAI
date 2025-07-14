import numpy as np
from lime import lime_image
from PIL import Image
import torch
from src.config import Config


class LIMEDeepfakeExplainer:
    def __init__(self, model, preprocess_fn):
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.explainer = lime_image.LimeImageExplainer()

    def batch_predict(self, images):
        """Batch prediction function for LIME"""
        self.model.eval()
        batch = torch.stack([self.preprocess_fn(Image.fromarray(img)) for img in images])
        with torch.no_grad():
            _, logits = self.model(batch.to(Config.DEVICE))
        return torch.sigmoid(logits).cpu().numpy()

    def explain(self, image, top_labels=1, num_samples=1000):
        """Generate LIME explanation for an image"""
        explanation = self.explainer.explain_instance(
            np.array(image),
            self.batch_predict,
            top_labels=top_labels,
            num_samples=num_samples,
            hide_color=0
        )

        # Get explanation visualization
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=False,
            num_features=5,
            hide_rest=False
        )
        return Image.fromarray(temp)
