"""
Deployment module for deepfake detection models.

This module provides functionality for exporting models to mobile-friendly formats
including TorchScript, ONNX, and quantized variants.
"""

from .model_export import MobileExporter

__all__ = ['MobileExporter']
