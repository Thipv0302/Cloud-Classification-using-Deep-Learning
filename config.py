"""
Configuration file cho Cloud Classification Project
Có thể chỉnh sửa các tham số ở đây
"""

# Data paths
TRAIN_DIR = 'clouds_train'
TEST_DIR = 'clouds_test'

# Image settings
IMG_SIZE = (224, 224)  # (width, height)
IMG_CHANNELS = 3  # RGB

# Training settings
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
USE_AUGMENTATION = True
FREEZE_BASE = True  # Cho transfer learning

# Model settings
MODEL_NAME = 'resnet50'  # 'simple', 'resnet50', 'efficientnet', 'mobilenet'
PRETRAINED = True

# Data augmentation settings
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'shear_range': 0.1,
    'zoom_range': 0.1,
    'horizontal_flip': True,
    'fill_mode': 'nearest',
    'brightness_range': [0.8, 1.2]
}

# Callbacks settings
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-7

# Directories
MODEL_SAVE_DIR = 'models'
EVALUATION_SAVE_DIR = 'evaluation_results'

# Evaluation settings
TOP_K_PREDICTIONS = 3
NUM_VISUALIZATION_SAMPLES = 16

