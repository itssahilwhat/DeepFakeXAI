import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
from src.config import Config


class RobustnessTester:
    """Robustness testing for deepfake detection models"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def test_noise_robustness(self, images, masks, noise_levels=None):
        """Test robustness against noise"""
        if noise_levels is None:
            noise_levels = Config.ROBUSTNESS_NOISE_LEVELS
            
        results = {}
        
        for noise_level in noise_levels:
            if noise_level == 0:
                # Clean images
                perturbed_images = images
            else:
                # Add noise
                noise = torch.randn_like(images) * (noise_level / 255.0)
                perturbed_images = torch.clamp(images + noise, 0, 1)
            
            # Evaluate
            with torch.no_grad():
                self.model.eval()
                predictions = self.model(perturbed_images)
                if isinstance(predictions, tuple):
                    _, seg_logits = predictions
                else:
                    seg_logits = predictions
                    
                seg_output = torch.sigmoid(seg_logits)
                
                # Compute metrics
                metrics = self._compute_metrics(seg_output, masks)
                results[f'noise_{noise_level}'] = metrics
                
        return results
    
    def test_compression_robustness(self, images, masks, compression_levels=None):
        """Test robustness against JPEG compression"""
        if compression_levels is None:
            compression_levels = Config.ROBUSTNESS_COMPRESSION_LEVELS
            
        results = {}
        
        for quality in compression_levels:
            if quality == 100:
                # No compression
                perturbed_images = images
            else:
                # Apply JPEG compression
                perturbed_images = self._apply_jpeg_compression(images, quality)
            
            # Evaluate
            with torch.no_grad():
                self.model.eval()
                predictions = self.model(perturbed_images)
                if isinstance(predictions, tuple):
                    _, seg_logits = predictions
                else:
                    seg_logits = predictions
                    
                seg_output = torch.sigmoid(seg_logits)
                
                # Compute metrics
                metrics = self._compute_metrics(seg_output, masks)
                results[f'compression_{quality}'] = metrics
                
        return results
    
    def test_blur_robustness(self, images, masks, blur_levels=None):
        """Test robustness against blur"""
        if blur_levels is None:
            blur_levels = Config.ROBUSTNESS_BLUR_LEVELS
            
        results = {}
        
        for blur_level in blur_levels:
            if blur_level == 0:
                # No blur
                perturbed_images = images
            else:
                # Apply blur
                perturbed_images = self._apply_blur(images, blur_level)
            
            # Evaluate
            with torch.no_grad():
                self.model.eval()
                predictions = self.model(perturbed_images)
                if isinstance(predictions, tuple):
                    _, seg_logits = predictions
                else:
                    seg_logits = predictions
                    
                seg_output = torch.sigmoid(seg_logits)
                
                # Compute metrics
                metrics = self._compute_metrics(seg_output, masks)
                results[f'blur_{blur_level}'] = metrics
                
        return results
    
    def test_geometric_robustness(self, images, masks):
        """Test robustness against geometric transformations"""
        results = {}
        
        # Rotation
        rotated_images = self._apply_rotation(images, angle=15)
        with torch.no_grad():
            self.model.eval()
            predictions = self.model(rotated_images)
            if isinstance(predictions, tuple):
                _, seg_logits = predictions
            else:
                seg_logits = predictions
            seg_output = torch.sigmoid(seg_logits)
            results['rotation_15'] = self._compute_metrics(seg_output, masks)
        
        # Scaling
        scaled_images = self._apply_scaling(images, scale=0.8)
        with torch.no_grad():
            predictions = self.model(scaled_images)
            if isinstance(predictions, tuple):
                _, seg_logits = predictions
            else:
                seg_logits = predictions
            seg_output = torch.sigmoid(seg_logits)
            results['scaling_0.8'] = self._compute_metrics(seg_output, masks)
        
        return results
    
    def _apply_jpeg_compression(self, images, quality):
        """Apply JPEG compression to images"""
        compressed_images = []
        
        for i in range(images.shape[0]):
            # Convert to PIL
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            
            # Apply compression
            import io
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            
            # Load back
            compressed_pil = Image.open(buffer)
            compressed_np = np.array(compressed_pil).astype(np.float32) / 255.0
            compressed_tensor = torch.from_numpy(compressed_np.transpose(2, 0, 1)).to(self.device)
            compressed_images.append(compressed_tensor)
        
        return torch.stack(compressed_images)
    
    def _apply_blur(self, images, kernel_size):
        """Apply Gaussian blur to images"""
        blurred_images = []
        
        for i in range(images.shape[0]):
            # Convert to PIL
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            
            # Apply blur
            blurred_pil = pil_img.filter(ImageFilter.GaussianBlur(radius=kernel_size/2))
            blurred_np = np.array(blurred_pil).astype(np.float32) / 255.0
            blurred_tensor = torch.from_numpy(blurred_np.transpose(2, 0, 1)).to(self.device)
            blurred_images.append(blurred_tensor)
        
        return torch.stack(blurred_images)
    
    def _apply_rotation(self, images, angle):
        """Apply rotation to images"""
        rotated_images = []
        
        for i in range(images.shape[0]):
            # Convert to PIL
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            
            # Apply rotation
            rotated_pil = pil_img.rotate(angle, expand=True)
            # Resize back to original size
            rotated_pil = rotated_pil.resize((images.shape[3], images.shape[2]))
            rotated_np = np.array(rotated_pil).astype(np.float32) / 255.0
            rotated_tensor = torch.from_numpy(rotated_np.transpose(2, 0, 1)).to(self.device)
            rotated_images.append(rotated_tensor)
        
        return torch.stack(rotated_images)
    
    def _apply_scaling(self, images, scale):
        """Apply scaling to images"""
        scaled_images = []
        
        for i in range(images.shape[0]):
            # Convert to PIL
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            
            # Apply scaling
            new_size = (int(images.shape[3] * scale), int(images.shape[2] * scale))
            scaled_pil = pil_img.resize(new_size)
            # Resize back to original size
            scaled_pil = scaled_pil.resize((images.shape[3], images.shape[2]))
            scaled_np = np.array(scaled_pil).astype(np.float32) / 255.0
            scaled_tensor = torch.from_numpy(scaled_np.transpose(2, 0, 1)).to(self.device)
            scaled_images.append(scaled_tensor)
        
        return torch.stack(scaled_images)
    
    def _compute_metrics(self, predictions, targets):
        """Compute evaluation metrics"""
        from src.utils import dice_coefficient, iou_pytorch, precision_recall_f1
        
        # Convert to binary
        pred_binary = (predictions > 0.5).float()
        
        # Compute metrics
        dice = dice_coefficient(predictions, targets).item()
        iou = iou_pytorch(predictions, targets).mean().item()
        precision, recall, f1 = precision_recall_f1(pred_binary, targets)
        
        return {
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def comprehensive_test(self, images, masks):
        """Run comprehensive robustness testing"""
        results = {}
        
        # Test different types of perturbations
        results['noise'] = self.test_noise_robustness(images, masks)
        results['compression'] = self.test_compression_robustness(images, masks)
        results['blur'] = self.test_blur_robustness(images, masks)
        results['geometric'] = self.test_geometric_robustness(images, masks)
        
        return results 