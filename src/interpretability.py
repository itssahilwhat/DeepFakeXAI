
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import clip
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from src.config import Config


class InterpretabilityTools:
    """Comprehensive interpretability tools for deepfake detection"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.clip_model = None
        self.clip_preprocess = None
        
        # Initialize CLIP if enabled
        if Config.USE_CLIP_EXPLAINER:
            self._init_clip()
    
    def _init_clip(self):
        """Initialize CLIP model"""
        try:
            self.clip_model, self.clip_preprocess = clip.load(Config.CLIP_MODEL_NAME, device=self.device)
            self.clip_model.eval()
        except Exception as e:
            print(f"Warning: Could not load CLIP model: {e}")
            self.clip_model = None
            self.clip_preprocess = None
    
    def generate_eigencam(self, image, target_layer=None):
        """Generate EigenCAM visualization (works with any architecture)"""
        self.model.eval()
        
        if target_layer is None:
            # Find the target layer
            target_layer = self._find_target_layer()
        
        # Check if target layer was found
        if target_layer is None:
            print("Warning: No suitable target layer found for EigenCAM")
            return None
        
        try:
            cam = EigenCAM(
                model=self.model,
                target_layers=[target_layer]
            )
            
            # Generate CAM
            grayscale_cam = cam(image.cpu().numpy(), targets=[SemanticSegmentationTarget(0, None)])[0]
            
            # Convert to heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            return heatmap
        except Exception as e:
            print(f"EigenCAM generation failed: {e}")
            return None
    
    def generate_gradcam(self, image, target_layer=None):
        """Generate GradCAM visualization with fallback for EfficientNet"""
        self.model.eval()
        
        if target_layer is None:
            target_layer = self._find_target_layer()
        
        if target_layer is None:
            print("Warning: No suitable target layer found for GradCAM")
            return None
        
        try:
            # Use standard GradCAM
            cam = GradCAM(
                model=self.model,
                target_layers=[target_layer]
            )
            
            # Generate CAM
            grayscale_cam = cam(image.cpu().numpy(), targets=[SemanticSegmentationTarget(0, None)])[0]
            
            # Convert to heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            return heatmap
        except Exception as e:
            print(f"GradCAM failed, using fallback: {e}")
            # Fallback: use prediction-based attention map
            return self._generate_fallback_attention(image)
    
    def _generate_fallback_attention(self, image):
        """Fallback attention map based on model predictions"""
        try:
            with torch.no_grad():
                predictions = self.model(image)
                if isinstance(predictions, tuple):
                    _, seg_logits = predictions
                else:
                    seg_logits = predictions
                pred_mask = torch.sigmoid(seg_logits)
            
            attention_map = pred_mask[0, 0].cpu().numpy()
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
            
            heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            return heatmap
        except Exception as e:
            print(f"Fallback attention generation failed: {e}")
            return None
    
    def generate_shap_explanation(self, image, num_samples=100):
        """Generate SHAP explanation for the model (memory-efficient CPU version)"""
        self.model.eval()
        
        try:
            # Convert image to the format SHAP expects
            img_np = image[0].cpu().permute(1, 2, 0).numpy()
            
            # Use a simpler SHAP approach with KernelExplainer on CPU
            def predict_fn(x):
                # Move to CPU to avoid GPU memory issues
                x_reshaped = x.reshape(-1, 3, 256, 256)
                x_tensor = torch.from_numpy(x_reshaped).float().cpu()  # Use CPU
                
                with torch.no_grad():
                    # Move model to CPU temporarily
                    model_cpu = self.model.cpu()
                    predictions = model_cpu(x_tensor)
                    if isinstance(predictions, tuple):
                        _, seg_logits = predictions
                    else:
                        seg_logits = predictions
                    pred_mask = torch.sigmoid(seg_logits)
                    # Return mean prediction for each sample
                    result = pred_mask.mean(dim=[2, 3]).numpy()
                    # Move model back to GPU
                    self.model.to(self.device)
                    return result
            
            # Create smaller background data
            background = np.random.rand(3, img_np.size)  # Reduced background size
            
            # Create SHAP explainer
            explainer = shap.KernelExplainer(predict_fn, background)
            
            # Generate SHAP values with fewer samples
            shap_values = explainer.shap_values(img_np.reshape(1, -1), nsamples=50)  # Reduced samples
            
            # Convert to image format
            shap_image = np.array(shap_values).reshape(img_np.shape)
            shap_image = np.abs(shap_image)  # Take absolute values
            shap_image = (shap_image - shap_image.min()) / (shap_image.max() - shap_image.min() + 1e-8)
            
            # Convert to heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * shap_image), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            return heatmap
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return None
    
    def _shap_predict_fn(self, x):
        """Helper function for SHAP predictions"""
        try:
            # Reshape input
            x_reshaped = x.reshape(-1, 3, 256, 256)
            x_tensor = torch.from_numpy(x_reshaped).float().to(self.device)
            
            with torch.no_grad():
                predictions = self.model(x_tensor)
                if isinstance(predictions, tuple):
                    _, seg_logits = predictions
                else:
                    seg_logits = predictions
                pred_mask = torch.sigmoid(seg_logits)
                # Return mean prediction for each sample
                return pred_mask.mean(dim=[2, 3]).cpu().numpy()
        except Exception as e:
            print(f"SHAP prediction failed: {e}")
            return np.zeros(x.shape[0])
    
    def generate_lime_explanation(self, image, num_samples=500):
        """Generate LIME explanation"""
        self.model.eval()
        
        # Convert tensor to numpy
        img_np = (image[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        def batch_predict(images):
            """Batch prediction function for LIME"""
            tensors = torch.stack([
                torch.from_numpy(img.astype(np.float32)/255.).permute(2,0,1).to(self.device)
                for img in images
            ])
            with torch.no_grad():
                predictions = self.model(tensors)
                if isinstance(predictions, tuple):
                    _, seg_logits = predictions
                else:
                    seg_logits = predictions
                probs = torch.sigmoid(seg_logits).cpu().numpy()
                # Reshape to 2D: (batch, features) where features = height * width * channels
                batch_size = probs.shape[0]
                probs_2d = probs.reshape(batch_size, -1)
            return probs_2d
        
        # Generate LIME explanation
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img_np,
            batch_predict,
            top_labels=1,
            hide_color=Config.LIME_HIDE_COLOR,
            num_samples=num_samples
        )
        
        # Get explanation visualization
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        
        overlay = mark_boundaries(temp, mask)
        return (overlay * 255).astype(np.uint8)
    
    def generate_clip_explanation(self, image):
        """Generate CLIP-based explanation"""
        if self.clip_model is None:
            return None
        
        self.model.eval()
        
        # Preprocess image for CLIP
        pil_image = Image.fromarray((image[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        clip_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # Define artifact descriptions
        artifact_descriptions = {
            "blur": ["blurry face edges", "unfocused facial features", "hazy texture around nose and mouth"],
            "warp": ["unnatural face contours", "distorted facial proportions", "misaligned facial landmarks"],
            "color": ["inconsistent skin tones", "unnatural lighting patterns", "mismatched color gradients"],
            "texture": ["repetitive skin patterns", "artificial skin pores", "synthetic hair texture"],
            "boundary": ["sharp face boundaries", "unnatural face edges", "artificial face borders"]
        }
        
        # Get model prediction
        with torch.no_grad():
            predictions = self.model(image)
            if isinstance(predictions, tuple):
                _, seg_logits = predictions
            else:
                seg_logits = predictions
            pred_mask = torch.sigmoid(seg_logits)
            
            # Encode image with CLIP
            image_features = self.clip_model.encode_image(clip_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Test different artifact descriptions
            artifact_scores = {}
            for artifact_type, descriptions in artifact_descriptions.items():
                text_inputs = clip.tokenize(descriptions).to(self.device)
                text_features = self.clip_model.encode_text(text_inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity
                similarity = (image_features @ text_features.T).softmax(dim=-1)
                artifact_scores[artifact_type] = similarity[0].max().item()
        
        return artifact_scores
    
    def generate_attention_maps(self, image):
        """Generate attention maps from the model"""
        self.model.eval()
        
        # Register hooks to capture attention maps
        attention_maps = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                attention_maps[name] = output.detach()
            return hook
        
        # Register hooks on attention layers
        hooks = []
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            predictions = self.model(image)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_maps
    
    def generate_saliency_maps(self, image):
        """Generate saliency maps using gradient-based methods"""
        self.model.eval()
        image.requires_grad_(True)
        
        # Forward pass
        predictions = self.model(image)
        if isinstance(predictions, tuple):
            _, seg_logits = predictions
        else:
            seg_logits = predictions
        
        # Backward pass
        seg_logits.sum().backward()
        
        # Get gradients
        gradients = image.grad
        
        # Generate saliency map
        saliency_map = torch.abs(gradients).sum(dim=1, keepdim=True)
        saliency_map = F.interpolate(saliency_map, size=image.shape[2:], mode='bilinear', align_corners=False)
        
        # Normalize
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        
        return saliency_map
    
    def generate_occlusion_maps(self, image, patch_size=16):
        """Generate occlusion sensitivity maps"""
        self.model.eval()
        
        B, C, H, W = image.shape
        occlusion_map = torch.zeros(1, 1, H, W).to(self.device)
        
        # Slide window over the image
        for i in range(0, H - patch_size + 1, patch_size // 2):
            for j in range(0, W - patch_size + 1, patch_size // 2):
                # Create occluded image
                occluded_image = image.clone()
                occluded_image[:, :, i:i+patch_size, j:j+patch_size] = 0
                
                # Get prediction
                with torch.no_grad():
                    predictions = self.model(occluded_image)
                    if isinstance(predictions, tuple):
                        _, seg_logits = predictions
                    else:
                        seg_logits = predictions
                    pred = torch.sigmoid(seg_logits).mean()
                
                # Update occlusion map
                occlusion_map[:, :, i:i+patch_size, j:j+patch_size] = pred
        
        return occlusion_map
    
    def _find_target_layer(self):
        """Find the target layer for GradCAM"""
        target_layer_name = Config.GRADCAM_TARGET_LAYER
        
        # First try to find the exact target layer
        for name, module in self.model.named_modules():
            if target_layer_name in name:
                return module
        
        # Look for encoder layers specifically
        for name, module in self.model.named_modules():
            if 'encoder' in name and isinstance(module, torch.nn.Conv2d):
                return module
        
        # Look for any Conv2d layer in the model
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                return module
        
        # If still no layer found, return the encoder itself
        if hasattr(self.model, 'encoder'):
            return self.model.encoder
        
        # Final fallback: return the model itself
        return self.model
    
    def comprehensive_explanation(self, image):
        """Generate comprehensive explanation with multiple methods"""
        explanations = {}
        
        # GradCAM
        try:
            explanations['gradcam'] = self.generate_gradcam(image)
        except Exception as e:
            print(f"GradCAM failed: {e}")
            explanations['gradcam'] = None
        
        # LIME
        try:
            explanations['lime'] = self.generate_lime_explanation(image)
        except Exception as e:
            print(f"LIME failed: {e}")
            explanations['lime'] = None
        
        # CLIP
        try:
            explanations['clip'] = self.generate_clip_explanation(image)
        except Exception as e:
            print(f"CLIP failed: {e}")
            explanations['clip'] = None
        
        # Attention maps
        try:
            explanations['attention'] = self.generate_attention_maps(image)
        except Exception as e:
            print(f"Attention maps failed: {e}")
            explanations['attention'] = None
        
        # Saliency maps
        try:
            explanations['saliency'] = self.generate_saliency_maps(image)
        except Exception as e:
            print(f"Saliency maps failed: {e}")
            explanations['saliency'] = None
        
        # SHAP
        try:
            explanations['shap'] = self.generate_shap_explanation(image)
        except Exception as e:
            print(f"SHAP failed: {e}")
            explanations['shap'] = None
        
        return explanations
    
    def save_explanations(self, explanations, save_path):
        """Save explanation visualizations"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Original image
        img_np = (explanations.get('original', torch.zeros(1, 3, 256, 256))[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # GradCAM
        if explanations.get('gradcam') is not None:
            axes[1].imshow(explanations['gradcam'])
            axes[1].set_title('GradCAM')
            axes[1].axis('off')
        
        # LIME
        if explanations.get('lime') is not None:
            axes[2].imshow(explanations['lime'])
            axes[2].set_title('LIME')
            axes[2].axis('off')
        
        # Saliency
        if explanations.get('saliency') is not None:
            saliency_np = explanations['saliency'][0, 0].cpu().numpy()
            axes[3].imshow(saliency_np, cmap='hot')
            axes[3].set_title('Saliency Map')
            axes[3].axis('off')
        
        # CLIP scores
        if explanations.get('clip') is not None:
            clip_scores = explanations['clip']
            artifact_types = list(clip_scores.keys())
            scores = list(clip_scores.values())
            axes[4].bar(artifact_types, scores)
            axes[4].set_title('CLIP Artifact Scores')
            axes[4].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() 