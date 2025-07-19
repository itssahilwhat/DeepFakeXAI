from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="deepfake-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A production-grade deepfake detection system (image-only)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/deepfake-detection",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "torchvision",
        "fastapi",
        "uvicorn",
        "albumentations",
        "scikit-learn",
        "pillow",
        "tqdm",
        "tensorboard",
        "numpy"
    ],
    entry_points={
        "console_scripts": [
            "deepfake-train=src.train:train",
        ],
    },
) 