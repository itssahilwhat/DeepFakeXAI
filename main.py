#!/usr/bin/env python3
"""
Deepfake Detection System - Main Entry Point
============================================

This is the main entry point for the deepfake detection system.
It provides easy access to training, testing, and API functionality.

Usage:
    python main.py train --datasets celebahq,ffhq
    python main.py test --folder /path/to/images --dataset celebahq
    python main.py api --port 8000
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection System")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--datasets', type=str, required=True,
                             help='Comma-separated list of dataset names (e.g., celebahq,ffhq)')
    
    # Testing command
    test_parser = subparsers.add_parser('test', help='Test the model on images')
    test_parser.add_argument('--folder', type=str, required=True,
                            help='Path to folder containing test images')
    test_parser.add_argument('--dataset', type=str, required=True,
                            help='Dataset name for loading model')
    
    # API command
    api_parser = subparsers.add_parser('api', help='Start the API server')
    api_parser.add_argument('--port', type=int, default=8000,
                           help='Port to run the API server on')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export model to ONNX')
    export_parser.add_argument('--checkpoint', type=str, required=True,
                              help='Path to trained checkpoint')
    export_parser.add_argument('--output', type=str, default='model.onnx',
                              help='Output path for ONNX model')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        from src.train import train_model
        train_model(args.datasets.split(','))
    
    elif args.command == 'test':
        from src.test_system import test_system
        test_system(args.folder, args.dataset)
    
    elif args.command == 'api':
        import uvicorn
        from api.fastapi_app import app
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    
    elif args.command == 'export':
        from scripts.export_onnx import export_to_onnx
        export_to_onnx(args.checkpoint, args.output)
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 