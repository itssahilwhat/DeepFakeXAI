# src/text_explainer.py
import torch
import clip
from config import Config
from PIL import Image
import numpy as np


class ArtifactTextExplainer:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if Config.USE_CLIP_EXPLAINER:
            # CLIP-based explanation setup
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.artifact_descriptions = {
                "blur": ["blurry face edges", "unfocused facial features",
                         "hazy texture around nose and mouth"],
                "warp": ["unnatural face contours", "distorted facial proportions",
                         "misaligned facial landmarks"],
                "color": ["inconsistent skin tones", "unnatural lighting patterns",
                          "mismatched color gradients"],
                "texture": ["repetitive skin patterns", "artificial skin pores",
                            "synthetic hair texture"]
            }
        else:
            # Original DistilBERT approach
            from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
            self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
            if model_path:
                self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
            else:
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased", num_labels=2
                )
            self.model.to(self.device)
            self.model.eval()

    def explain(self, artifacts, image=None):
        if Config.USE_CLIP_EXPLAINER and image is not None:
            # CLIP-based explanation
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            phrases = []
            for artifact in artifacts:
                if artifact in self.artifact_descriptions:
                    phrases.extend(self.artifact_descriptions[artifact])

            if not phrases:
                return "No artifacts detected"

            text_inputs = clip.tokenize(phrases).to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_inputs)

                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # Calculate similarity
                similarity = (image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(min(2, len(phrases)))

            # Generate explanation
            explanation = "Detected artifacts: "
            for i, idx in enumerate(indices):
                explanation += f"{phrases[idx]} ({values[i].item():.2f})"
                if i < len(indices) - 1:
                    explanation += ", "
            return explanation
        else:
            # Template-based explanation
            explanations = []
            for art in artifacts:
                if "blur" in art:
                    explanations.append(f"The image contains a blur artifact in region: {art}.")
                elif "warp" in art or "distort" in art:
                    explanations.append(f"The image contains a warp/distortion artifact: {art}.")
                else:
                    explanations.append(f"Detected artifact: {art}.")
            return " ".join(explanations) if explanations else "No artifacts detected"