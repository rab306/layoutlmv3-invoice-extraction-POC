from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="invoice-ocr-layoutlmv3",
    version="0.1.0",
    author="Rabea El-hadad",
    author_email="your.email@example.com",
    description="Invoice information extraction using LayoutLMv3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/invoice-ocr-layoutlmv3",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core ML
        "torch==2.9.1",
        "torchvision==0.24.1",
        "transformers==4.57.3",
        "datasets==4.4.2",
        "evaluate==0.4.6",
        "accelerate==1.12.0",
        "huggingface-hub==0.36.0",
        "tokenizers==0.22.2",

        # OCR
        "pytesseract==0.3.13",
        "Pillow==12.1.0",
        "opencv-python==4.12.0.88",

        # Data processing
        "numpy==2.2.6",
        "pandas==2.3.3",
        "scikit-learn==1.8.0",
        "RapidFuzz==3.14.3",

        # Evaluation
        "seqeval==1.2.2",

        # Utilities
        "tqdm==4.67.1",
        "pyyaml==6.0.3",
        "regex==2025.11.3",
        "requests==2.32.5",

        # Visualization (optional, can be installed separately)
        "matplotlib==3.10.8",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": [
            "jupyter-client==8.7.0",
            "ipykernel==7.1.0",
            "black>=23.0.0",
            "pytest>=7.4.0"
        ]
    }
)
