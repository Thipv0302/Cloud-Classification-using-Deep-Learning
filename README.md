# 🌥️ Cloud Classification using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning project for classifying cloud types from images using Convolutional Neural Networks (CNN) and Transfer Learning.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a cloud classification system that can identify 7 different types of clouds:

- **Cirriform clouds** (Cirrus)
- **Clear sky**
- **Cumulonimbus clouds**
- **Cumulus clouds**
- **High cumuliform clouds**
- **Stratiform clouds**
- **Stratocumulus clouds**

The system uses state-of-the-art deep learning models including ResNet50, EfficientNet, and MobileNetV2 with transfer learning for accurate classification.

## ✨ Features

- 🔄 **Multiple Model Architectures**: Support for Simple CNN, ResNet50, EfficientNet, and MobileNetV2
- 📊 **Comprehensive Evaluation**: Confusion matrix, classification reports, and visualization tools
- 🎨 **Data Augmentation**: Built-in augmentation to improve model generalization
- 📈 **Training Monitoring**: Real-time metrics tracking with early stopping and learning rate scheduling
- 🔍 **Easy Inference**: Simple API for single image and batch predictions
- 📉 **Visualization Tools**: Automatic generation of training curves and prediction visualizations

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- TensorFlow 2.10 or higher
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cloud-classification.git
cd cloud-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python test_setup.py
```

## 🏃 Quick Start

### 1. Explore the Dataset

```bash
python data_loader.py
```

This will display dataset information and visualize sample images.

### 2. Train a Model

```bash
# Train with ResNet50 (recommended)
python train.py --model resnet50 --epochs 50 --batch_size 32
```

### 3. Evaluate the Model

```bash
python evaluate.py --model models/resnet50_YYYYMMDD_HHMMSS/best_model.h5 --visualize
```

### 4. Make Predictions

```bash
python predict.py --model models/resnet50_YYYYMMDD_HHMMSS/best_model.h5 --image path/to/image.jpg
```

## 💻 Usage

### Training

#### Basic Training
```bash
python train.py --model resnet50 --epochs 50
```

#### Advanced Training Options
```bash
python train.py \
    --model resnet50 \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --save_dir models
```

#### Available Models
- `simple`: Simple CNN from scratch
- `resnet50`: ResNet50 with transfer learning (recommended)
- `efficientnet`: EfficientNetB0 (high accuracy)
- `mobilenet`: MobileNetV2 (lightweight, fast)

#### Training Options
- `--no_aug`: Disable data augmentation
- `--unfreeze`: Unfreeze base model for fine-tuning
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)

### Evaluation

```bash
# Basic evaluation
python evaluate.py --model models/resnet50_YYYYMMDD_HHMMSS/best_model.h5

# With visualizations
python evaluate.py --model models/resnet50_YYYYMMDD_HHMMSS/best_model.h5 --visualize
```

### Prediction

```bash
# Single image
python predict.py --model models/resnet50_YYYYMMDD_HHMMSS/best_model.h5 --image image.jpg

# Batch prediction
python predict.py --model models/resnet50_YYYYMMDD_HHMMSS/best_model.h5 --dir images/

# Top-K predictions
python predict.py --model models/resnet50_YYYYMMDD_HHMMSS/best_model.h5 --image image.jpg --top_k 5
```

### Data Visualization

```bash
# Analyze dataset
python visualize_data.py --all

# Specific analyses
python visualize_data.py --stats    # Dataset statistics
python visualize_data.py --samples  # Visualize samples
python visualize_data.py --sizes    # Image size analysis
```

## 📁 Project Structure

```
cloud-classification/
├── clouds_train/              # Training dataset
│   ├── cirriform clouds/
│   ├── clear sky/
│   ├── cumulonimbus clouds/
│   ├── cumulus clouds/
│   ├── high cumuliform clouds/
│   ├── stratiform clouds/
│   └── stratocumulus clouds/
├── clouds_test/               # Test dataset
│   └── [same structure as train]
├── models/                    # Trained models (generated)
│   └── resnet50_YYYYMMDD_HHMMSS/
│       ├── best_model.h5
│       ├── final_model.h5
│       ├── class_mapping.json
│       ├── training_log.csv
│       └── training_curves.png
├── evaluation_results/       # Evaluation outputs (generated)
├── config.py                 # Configuration file
├── data_loader.py             # Data loading and preprocessing
├── model.py                   # Model architectures
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
├── predict.py                 # Prediction script
├── visualize_data.py          # Data visualization tools
├── example_usage.py           # Usage examples
├── quick_start.py             # Quick start guide
├── test_setup.py              # Setup verification
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🏗️ Model Architectures

### 1. Simple CNN
A custom CNN architecture built from scratch with:
- 4 convolutional blocks
- Batch normalization and dropout
- Global average pooling
- Dense layers for classification

### 2. ResNet50 (Recommended)
Transfer learning with ImageNet pretrained ResNet50:
- Freeze base model option
- Fine-tuning support
- High accuracy

### 3. EfficientNetB0
EfficientNet architecture optimized for accuracy and efficiency:
- Compound scaling method
- State-of-the-art performance

### 4. MobileNetV2
Lightweight model suitable for mobile and edge devices:
- Depthwise separable convolutions
- Fast inference time

## 📊 Results

After training, you'll get:

### Model Files
- `best_model.h5`: Best model based on validation accuracy
- `final_model.h5`: Final model after all epochs
- `class_mapping.json`: Class to index mapping
- `training_log.csv`: Training history
- `training_curves.png`: Visualization of training metrics

### Evaluation Outputs
- `confusion_matrix.png`: Confusion matrix visualization
- `class_metrics.png`: Per-class precision, recall, F1-score
- `prediction_samples.png`: Sample predictions with images
- `evaluation_report.txt`: Detailed classification report

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Image settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Model settings
MODEL_NAME = 'resnet50'
PRETRAINED = True
FREEZE_BASE = True

# Data augmentation
USE_AUGMENTATION = True
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'shear_range': 0.1,
    'zoom_range': 0.1,
    'horizontal_flip': True,
    'brightness_range': [0.8, 1.2]
}
```

## 📈 Tips for Better Results

1. **Use Data Augmentation**: Always enable augmentation to increase data diversity
2. **Transfer Learning**: Use pretrained models (ResNet50, EfficientNet) instead of training from scratch
3. **Fine-tuning**: After training with frozen base, unfreeze and fine-tune with lower learning rate
4. **Hyperparameter Tuning**: Experiment with different learning rates and batch sizes
5. **Ensemble Methods**: Combine multiple models for better accuracy

## 🐛 Troubleshooting

### GPU Not Detected
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### Out of Memory
- Reduce `batch_size` (e.g., `--batch_size 16`)
- Use a lighter model (MobileNet)
- Reduce image size in `config.py`

### Training Too Slow
- Use GPU acceleration
- Increase `batch_size` if you have enough RAM
- Use data generators (already implemented)

### Import Errors
```bash
pip install -r requirements.txt
```

## 📚 Examples

See `example_usage.py` for comprehensive usage examples:

```bash
python example_usage.py
```

This interactive script guides you through:
1. Data exploration
2. Data loading
3. Model creation
4. Training
5. Evaluation
6. Prediction

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **TensorFlow/Keras**: Deep learning framework
- **ImageNet**: Pretrained model weights
- **Dataset**: Cloud images classification dataset

## 👤 Author

**Thipv0302**

- GitHub: [@Thipv0302](https://github.com/Thipv0302)

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for cloud classification**

⭐ If you find this project useful, please consider giving it a star!
