import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from lime import lime_image
import contextlib, io
from sklearn.metrics import roc_auc_score
import cv2
from PIL import Image

# ============================================================================
# GRAD-CAM EXPLANATION
# ============================================================================

def explain_gradcam(det_model, img_tensor, cls):
    """Generate Grad-CAM explanation for a given class"""
    det_model.eval()

    # Handle regular detection models (VanillaCNN)
    if hasattr(det_model, 'encoder'):
        # Use last conv layer in encoder (VanillaCNN)
        conv_layers = [m for m in det_model.encoder.modules() if isinstance(m, torch.nn.Conv2d)]
        if not conv_layers:
            raise AttributeError('No Conv2d layers found in model.encoder for Grad-CAM')
        target_layer = conv_layers[-1]
    else:
        raise AttributeError('Unsupported model type for Grad-CAM: expected encoder attribute')
    
    cam = GradCAM(model=det_model, target_layers=[target_layer])

    # Generate Grad-CAM heatmap
    grayscale_cam = cam(input_tensor=img_tensor.unsqueeze(0), targets=[ClassifierOutputTarget(cls)])[0]
    
    # Normalize to [0, 1] range
    return (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)

# ============================================================================
# LIME EXPLANATION
# ============================================================================

def explain_lime(
    predict_fn,
    img_np,
    num_samples: int = 1000,
    num_features: int = 5,
    positive_only: bool = True,
    hide_rest: bool = True,
    silent: bool = True,
):
    """Generate LIME explanation using superpixel segmentation"""
    explainer = lime_image.LimeImageExplainer()
    
    # Normalize image to [0, 1] range for LIME
    img_normalized = img_np.astype(np.float32) / 255.0
    
    # Generate explanation (with optional silent mode)
    if silent:
        fnull = io.StringIO()
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            exp = explainer.explain_instance(img_normalized, predict_fn, num_samples=num_samples)
    else:
        exp = explainer.explain_instance(img_normalized, predict_fn, num_samples=num_samples)
    
    # Extract mask for top predicted class
    _, mask = exp.get_image_and_mask(
        exp.top_labels[0],
        positive_only=positive_only,
        hide_rest=hide_rest,
        num_features=num_features,
    )
    
    # Convert to binary mask
    return (mask > 0).astype(np.float32)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def overlay_numpy(img, mask, alpha=0.5):
    """Overlay heatmap on image"""
    # Normalize image to [0, 1] if needed
    img = img.astype(np.float32) / 255 if img.max() > 1 else img.astype(np.float32)
    
    # Stack mask to 3 channels
    mask = np.stack([mask]*3, -1)
    
    # Blend image and mask
    return np.clip(img * (1 - alpha) + mask * alpha, 0, 1)

# ============================================================================
# FIDELITY METRICS
# ============================================================================

def compute_fidelity_deletion(model, img_tensor, heatmap, target_class, steps=10):
    """
    Compute deletion fidelity: how much confidence drops when removing top-k% of heatmap
    Returns AUC of confidence vs percentage of heatmap removed
    """
    # Regular detection model (VanillaCNN)
    original_conf = torch.softmax(model(img_tensor.unsqueeze(0)), dim=1)[0, target_class].item()
    
    # Sort heatmap values (highest to lowest)
    flat_heatmap = heatmap.flatten()
    sorted_indices = np.argsort(flat_heatmap)[::-1]
    
    confidences = []
    percentages = np.linspace(0, 1, steps)
    device = img_tensor.device
    
    # Test different percentages of heatmap removal
    for pct in percentages:
        # Create mask for current percentage
        mask = np.ones_like(heatmap)
        n_pixels_to_mask = int(pct * len(flat_heatmap))
        
        if n_pixels_to_mask > 0:
            indices_to_mask = sorted_indices[:n_pixels_to_mask]
            mask_flat = np.ones(len(flat_heatmap))
            mask_flat[indices_to_mask] = 0
            mask = mask_flat.reshape(heatmap.shape)
        
        # Apply mask to image
        masked_img = img_tensor.clone()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)
        masked_img = masked_img * mask_tensor
        
        # Get confidence for masked image
        with torch.no_grad():
            conf = torch.softmax(model(masked_img.unsqueeze(0)), dim=1)[0, target_class].item()
        
        confidences.append(conf)
    
    # Compute AUC (area under confidence vs percentage curve)
    # Higher AUC means confidence drops quickly when removing important regions
    auc = np.trapz(confidences, percentages)
    
    return auc, confidences, percentages

def compute_sensitivity_n(model, img_tensor, heatmap, target_class, n_percent=10):
    """
    Compute sensitivity-n: how much prediction changes when perturbing top-n% regions
    """
    # Regular detection model (VanillaCNN)
    original_conf = torch.softmax(model(img_tensor.unsqueeze(0)), dim=1)[0, target_class].item()
    
    # Get top n% of heatmap
    threshold = np.percentile(heatmap, 100 - n_percent)
    mask = (heatmap > threshold).astype(np.float32)
    
    # Perturb top regions (set to mean value)
    perturbed_img = img_tensor.clone()
    mean_val = img_tensor.mean()
    device = img_tensor.device
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)
    perturbed_img = perturbed_img * (1 - mask_tensor) + mean_val * mask_tensor
    
    # Get new confidence
    with torch.no_grad():
        new_conf = torch.softmax(model(perturbed_img.unsqueeze(0)), dim=1)[0, target_class].item()
    
    # Sensitivity is the absolute change in confidence
    sensitivity = abs(original_conf - new_conf)
    return sensitivity

# ============================================================================
# LOCALIZATION METRICS
# ============================================================================

def compute_localization_agreement(heatmap, ground_truth_mask, threshold=0.5):
    """Compute IoU between thresholded heatmap and ground truth segmentation mask"""
    # Threshold heatmap to binary
    heatmap_binary = (heatmap > threshold).astype(np.float32)
    
    # Ensure same shape
    if heatmap_binary.shape != ground_truth_mask.shape:
        # Resize ground truth to match heatmap
        from PIL import Image
        gt_resized = np.array(Image.fromarray(ground_truth_mask).resize(heatmap_binary.shape[::-1]))
        ground_truth_mask = gt_resized
    
    # Compute IoU
    intersection = (heatmap_binary * ground_truth_mask).sum()
    union = (heatmap_binary + ground_truth_mask).clip(0, 1).sum()
    
    iou = intersection / (union + 1e-8)
    return iou

# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================

def evaluate_explainability(model, img_tensor, img_np, target_class, ground_truth_mask=None):
    """
    Comprehensive explainability evaluation
    Returns dictionary with all metrics
    """
    # Generate explanations
    gradcam = explain_gradcam(model, img_tensor, target_class)
    
    # Regular detection model prediction function for LIME
    def lime_predict_fn(images):
        batch = []
        for img in images:
            # Preprocess image
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            img_tensor = (img_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            batch.append(img_tensor)
        
        batch_tensor = torch.stack(batch)
        with torch.no_grad():
            return model(batch_tensor).softmax(1).cpu().numpy()
    
    # Generate LIME explanation
    lime_mask = explain_lime(lime_predict_fn, img_np)
    
    # Compute metrics
    metrics = {}
    
    # Grad-CAM metrics
    deletion_auc, confidences, percentages = compute_fidelity_deletion(model, img_tensor, gradcam, target_class)
    sensitivity_10 = compute_sensitivity_n(model, img_tensor, gradcam, target_class, n_percent=10)
    
    metrics['gradcam_deletion_auc'] = deletion_auc
    metrics['gradcam_sensitivity_10'] = sensitivity_10
    
    # LIME metrics
    lime_deletion_auc, lime_confidences, lime_percentages = compute_fidelity_deletion(model, img_tensor, lime_mask, target_class)
    lime_sensitivity_10 = compute_sensitivity_n(model, img_tensor, lime_mask, target_class, n_percent=10)
    
    metrics['lime_deletion_auc'] = lime_deletion_auc
    metrics['lime_sensitivity_10'] = lime_sensitivity_10
    
    # Localization agreement (if ground truth available)
    if ground_truth_mask is not None:
        gradcam_localization = compute_localization_agreement(gradcam, ground_truth_mask)
        lime_localization = compute_localization_agreement(lime_mask, ground_truth_mask)
        
        metrics['gradcam_localization_iou'] = gradcam_localization
        metrics['lime_localization_iou'] = lime_localization
    
    return metrics, gradcam, lime_mask

# ============================================================================
# RISE EXPLANATION
# ============================================================================

def explain_rise(model, img_tensor, target_class, num_masks=1000, mask_size=7, prob_threshold=0.5):
    """RISE (Randomized Input Sampling for Explanation) implementation"""
    model.eval()
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    # Generate random binary masks
    masks = np.random.binomial(1, prob_threshold, (num_masks, mask_size, mask_size))
    masks = masks.astype(np.float32)
    
    # Upsample masks to image size
    img_size = img_tensor.shape[-1]
    masks_upsampled = np.array([cv2.resize(mask, (img_size, img_size)) for mask in masks])
    
    # Apply masks and get predictions
    explanations = np.zeros((img_size, img_size))
    
    with torch.no_grad():
        for i, mask in enumerate(masks_upsampled):
            # Apply mask to image
            masked_img = img_tensor.clone()
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
            masked_img = masked_img * mask_tensor
            
            # Get prediction (regular detection model)
            output = model(masked_img.unsqueeze(0))
            
            # Get probability for target class
            prob = torch.softmax(output, dim=1)[0, target_class].item()
            
            # Weighted sum
            explanations += prob * mask
    
    # Normalize explanations
    explanations = (explanations - explanations.min()) / (explanations.max() - explanations.min() + 1e-8)
    return explanations

# ============================================================================
# SHAP EXPLANATION
# ============================================================================

def explain_shap(model, img_tensor, target_class, num_samples=100):
    """SHAP (SHapley Additive exPlanations) implementation using KernelExplainer"""
    try:
        import shap
    except ImportError:
        print("SHAP not available. Install with: pip install shap")
        return None
    
    model.eval()
    device = next(model.parameters()).device
    
    # Convert tensor to numpy for SHAP
    img_np = img_tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    
    # Create background dataset
    background = img_np.reshape(1, -1)
    
    # Create explainer
    explainer = shap.KernelExplainer(
        lambda x: _shap_predict_fn(x, model, device),
        background
    )
    
    # Get explanation
    shap_values = explainer.shap_values(
        img_np.reshape(1, -1),
        nsamples=num_samples
    )
    
    # Reshape back to image dimensions
    if isinstance(shap_values, list):
        shap_values = shap_values[target_class]
    
    # Process SHAP values
    shap_heatmap = shap_values.reshape(img_np.shape[:2])
    shap_heatmap = np.abs(shap_heatmap)  # Use absolute values
    shap_heatmap = (shap_heatmap - shap_heatmap.min()) / (shap_heatmap.max() - shap_heatmap.min() + 1e-8)
    
    return shap_heatmap

def _shap_predict_fn(x, model, device):
    """Helper function for SHAP predictions"""
    # Reshape input
    x_reshaped = x.reshape(-1, 3, 224, 224)
    
    # Convert to tensor
    x_tensor = torch.from_numpy(x_reshaped).float().to(device)
    
    # Get predictions (regular detection model)
    with torch.no_grad():
        output = model(x_tensor)
        probs = torch.softmax(output, dim=1)
    
    return probs.detach().cpu().numpy()

# ============================================================================
# GRAD-CAM++ EXPLANATION
# ============================================================================

def explain_gradcam_plus_plus(model, img_tensor, target_class):
    """Grad-CAM++ implementation (enhanced version of Grad-CAM)"""
    model.eval()
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    # For now, use the existing Grad-CAM implementation as base
    # Grad-CAM++ typically involves computing higher-order derivatives
    base_cam = explain_gradcam(model, img_tensor, target_class)
    
    # TODO: Enhance with Grad-CAM++ specific logic
    # This would involve computing higher-order gradients and weights
    return base_cam