"""
Model Architecture cho Cloud Classification
Sử dụng CNN với các kiến trúc khác nhau
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    ResNet50, EfficientNetB0, MobileNetV2
)


def create_simple_cnn(input_shape=(224, 224, 3), num_classes=7):
    """
    Tạo một CNN đơn giản từ đầu
    
    Args:
        input_shape: Kích thước input (height, width, channels)
        num_classes: Số lượng classes
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Global Average Pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_resnet50_model(input_shape=(224, 224, 3), num_classes=7, 
                         pretrained=True, freeze_base=False):
    """
    Tạo model dựa trên ResNet50 (Transfer Learning)
    
    Args:
        input_shape: Kích thước input
        num_classes: Số lượng classes
        pretrained: Sử dụng pretrained weights
        freeze_base: Đóng băng base model
    
    Returns:
        Compiled Keras model
    """
    if pretrained:
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    else:
        base_model = ResNet50(
            weights=None,
            include_top=False,
            input_shape=input_shape
        )
    
    if freeze_base:
        base_model.trainable = False
    else:
        # Fine-tuning: chỉ train một số layers cuối
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_efficientnet_model(input_shape=(224, 224, 3), num_classes=7,
                             pretrained=True, freeze_base=False):
    """
    Tạo model dựa trên EfficientNetB0
    
    Args:
        input_shape: Kích thước input
        num_classes: Số lượng classes
        pretrained: Sử dụng pretrained weights
        freeze_base: Đóng băng base model
    
    Returns:
        Compiled Keras model
    """
    if pretrained:
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    else:
        base_model = EfficientNetB0(
            weights=None,
            include_top=False,
            input_shape=input_shape
        )
    
    if freeze_base:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_mobilenet_model(input_shape=(224, 224, 3), num_classes=7,
                          pretrained=True, freeze_base=False):
    """
    Tạo model dựa trên MobileNetV2 (nhẹ, nhanh)
    
    Args:
        input_shape: Kích thước input
        num_classes: Số lượng classes
        pretrained: Sử dụng pretrained weights
        freeze_base: Đóng băng base model
    
    Returns:
        Compiled Keras model
    """
    if pretrained:
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    else:
        base_model = MobileNetV2(
            weights=None,
            include_top=False,
            input_shape=input_shape
        )
    
    if freeze_base:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model, learning_rate=0.001, optimizer='adam'):
    """
    Compile model với optimizer và loss function phù hợp
    
    Args:
        model: Keras model
        learning_rate: Learning rate
        optimizer: 'adam', 'sgd', hoặc 'rmsprop'
    
    Returns:
        Compiled model
    """
    if optimizer.lower() == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer.lower() == 'rmsprop':
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')]
    )
    
    return model


def get_model_summary(model):
    """In summary của model"""
    model.summary()
    
    # Tính số parameters
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    print("\n" + "="*60)
    print("MODEL STATISTICS")
    print("="*60)
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print("="*60)


if __name__ == "__main__":
    # Test các model
    print("Testing Simple CNN...")
    model1 = create_simple_cnn(num_classes=7)
    model1 = compile_model(model1)
    get_model_summary(model1)
    
    print("\n\nTesting ResNet50...")
    model2 = create_resnet50_model(num_classes=7, pretrained=True, freeze_base=True)
    model2 = compile_model(model2)
    get_model_summary(model2)

