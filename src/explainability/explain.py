import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from lime import lime_image
from sklearn.metrics import roc_auc_score

def explain_gradcam(det_model, img_tensor, cls):
    det_model.eval()
    # Get the last convolutional layer from the timm model
    target_layer = det_model.net.features[-1]  # For convnext_tiny
    cam = GradCAM(model=det_model, target_layers=[target_layer], use_cuda=next(det_model.parameters()).is_cuda)
    grayscale_cam = cam(input_tensor=img_tensor.unsqueeze(0), targets=[ClassifierOutputTarget(cls)])[0]
    return (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)

def explain_lime(predict_fn, img_np):
    explainer = lime_image.LimeImageExplainer()
    # Normalize image for LIME (0-1 range)
    img_normalized = img_np.astype(np.float32) / 255.0
    exp = explainer.explain_instance(img_normalized, predict_fn, num_samples=500)
    mask = exp.get_image_and_mask(exp.top_labels[0], positive_only=True, hide_rest=False)[1]
    return (mask > 0).astype(np.float32)

def overlay_numpy(img, mask, alpha=0.5):
    img = img.astype(np.float32) / 255 if img.max() > 1 else img.astype(np.float32)
    mask = np.stack([mask]*3, -1)
    return np.clip(img * (1 - alpha) + mask * alpha, 0, 1)

def compute_fidelity_deletion(model, img_tensor, heatmap, target_class, steps=10):
    """
    Compute deletion fidelity: how much confidence drops when removing top-k% of heatmap
    Returns AUC of confidence vs percentage of heatmap removed
    """
    original_conf = torch.softmax(model(img_tensor.unsqueeze(0)), dim=1)[0, target_class].item()
    
    # Sort heatmap values
    flat_heatmap = heatmap.flatten()
    sorted_indices = np.argsort(flat_heatmap)[::-1]  # Descending order
    
    confidences = []
    percentages = np.linspace(0, 1, steps)
    
    for pct in percentages:
        # Create masked image
        mask = np.ones_like(heatmap)
        n_pixels_to_mask = int(pct * len(flat_heatmap))
        if n_pixels_to_mask > 0:
            indices_to_mask = sorted_indices[:n_pixels_to_mask]
            mask_flat = np.ones(len(flat_heatmap))
            mask_flat[indices_to_mask] = 0
            mask = mask_flat.reshape(heatmap.shape)
        
        # Apply mask to image
        masked_img = img_tensor.clone()
        masked_img = masked_img * torch.from_numpy(mask).unsqueeze(0)
        
        # Get confidence
        with torch.no_grad():
            conf = torch.softmax(model(masked_img.unsqueeze(0)), dim=1)[0, target_class].item()
        confidences.append(conf)
    
    # Compute AUC (higher is better - means confidence drops quickly when removing important regions)
    try:
        auc = roc_auc_score([1] * len(percentages), confidences)
    except ValueError:
        auc = 0.5
    
    return auc, confidences, percentages

def compute_sensitivity_n(model, img_tensor, heatmap, target_class, n_percent=10):
    """
    Compute sensitivity-n: how much prediction changes when perturbing top-n% regions
    """
    original_conf = torch.softmax(model(img_tensor.unsqueeze(0)), dim=1)[0, target_class].item()
    
    # Get top n% of heatmap
    threshold = np.percentile(heatmap, 100 - n_percent)
    mask = (heatmap > threshold).astype(np.float32)
    
    # Perturb top regions (set to mean value)
    perturbed_img = img_tensor.clone()
    mean_val = img_tensor.mean()
    perturbed_img = perturbed_img * (1 - torch.from_numpy(mask).unsqueeze(0)) + mean_val * torch.from_numpy(mask).unsqueeze(0)
    
    # Get new confidence
    with torch.no_grad():
        new_conf = torch.softmax(model(perturbed_img.unsqueeze(0)), dim=1)[0, target_class].item()
    
    # Sensitivity is the absolute change in confidence
    sensitivity = abs(original_conf - new_conf)
    return sensitivity

def compute_localization_agreement(heatmap, ground_truth_mask, threshold=0.5):
    """
    Compute IoU between thresholded heatmap and ground truth segmentation mask
    """
    # Threshold heatmap
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

def evaluate_explainability(model, img_tensor, img_np, target_class, ground_truth_mask=None):
    """
    Comprehensive explainability evaluation
    Returns dictionary with all metrics
    """
    # Generate explanations
    gradcam = explain_gradcam(model, img_tensor, target_class)
    
    def lime_predict_fn(images):
        batch = []
        for img in images:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            img_tensor = (img_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            batch.append(img_tensor)
        batch_tensor = torch.stack(batch)
        with torch.no_grad():
            return model(batch_tensor).softmax(1).cpu().numpy()
    
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