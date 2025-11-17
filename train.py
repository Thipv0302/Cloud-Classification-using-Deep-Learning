"""
Training Script cho Cloud Classification Model
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import matplotlib.pyplot as plt

from data_loader import CloudDataLoader
from model import (
    create_simple_cnn, create_resnet50_model, 
    create_efficientnet_model, create_mobilenet_model,
    compile_model
)


class TrainingCallback(keras.callbacks.Callback):
    """Custom callback để in thông tin training"""
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:
            print(f"\nEpoch {epoch + 1}/{self.params['epochs']}")
            print(f"  Loss: {logs['loss']:.4f} - Accuracy: {logs['accuracy']:.4f}")
            if 'val_loss' in logs:
                print(f"  Val Loss: {logs['val_loss']:.4f} - Val Accuracy: {logs['val_accuracy']:.4f}")


def train_model(model_name='resnet50', epochs=50, batch_size=32, 
               learning_rate=0.001, use_augmentation=True,
               save_dir='models', freeze_base=True):
    """
    Train model
    
    Args:
        model_name: 'simple', 'resnet50', 'efficientnet', 'mobilenet'
        epochs: Số epochs
        batch_size: Batch size
        learning_rate: Learning rate
        use_augmentation: Sử dụng data augmentation
        save_dir: Thư mục lưu model
        freeze_base: Đóng băng base model (cho transfer learning)
    """
    print("="*70)
    print("CLOUD CLASSIFICATION - TRAINING")
    print("="*70)
    
    # Tạo thư mục lưu model
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(save_dir, f"{model_name}_{timestamp}")
    os.makedirs(model_save_path, exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading data...")
    try:
        loader = CloudDataLoader()
        info = loader.get_data_info()
        train_gen, test_gen = loader.get_data_generators(
            batch_size=batch_size, 
            use_augmentation=use_augmentation
        )
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")
    
    num_classes = info['num_classes']
    if num_classes == 0:
        raise ValueError("No classes found in training data. Please check your data directory structure.")
    
    steps_per_epoch = train_gen.samples // batch_size
    validation_steps = test_gen.samples // batch_size
    
    if steps_per_epoch == 0:
        raise ValueError(f"Not enough training samples. Need at least {batch_size} samples, but found {train_gen.samples}.")
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    
    # Tạo model
    print(f"\n[2/5] Creating {model_name} model...")
    input_shape = (224, 224, 3)
    
    try:
        if model_name == 'simple':
            model = create_simple_cnn(input_shape=input_shape, num_classes=num_classes)
        elif model_name == 'resnet50':
            model = create_resnet50_model(
                input_shape=input_shape, 
                num_classes=num_classes,
                pretrained=True,
                freeze_base=freeze_base
            )
        elif model_name == 'efficientnet':
            model = create_efficientnet_model(
                input_shape=input_shape,
                num_classes=num_classes,
                pretrained=True,
                freeze_base=freeze_base
            )
        elif model_name == 'mobilenet':
            model = create_mobilenet_model(
                input_shape=input_shape,
                num_classes=num_classes,
                pretrained=True,
                freeze_base=freeze_base
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
    except Exception as e:
        raise RuntimeError(f"Error creating model: {e}")
    
    # Compile model
    try:
        model = compile_model(model, learning_rate=learning_rate)
    except Exception as e:
        raise RuntimeError(f"Error compiling model: {e}")
    
    # Callbacks
    print("\n[3/5] Setting up callbacks...")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_save_path, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            filename=os.path.join(model_save_path, 'training_log.csv')
        ),
        TrainingCallback()
    ]
    
    # Training
    print("\n[4/5] Starting training...")
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Data augmentation: {use_augmentation}")
    print("-"*70)
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=test_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Lưu model cuối cùng
    print("\n[5/5] Saving final model...")
    model.save(os.path.join(model_save_path, 'final_model.h5'))
    
    # Lưu class mapping
    import json
    class_mapping = {
        'class_to_idx': loader.class_to_idx,
        'idx_to_class': {str(k): v for k, v in loader.idx_to_class.items()},
        'classes': loader.classes
    }
    with open(os.path.join(model_save_path, 'class_mapping.json'), 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    # Vẽ training curves
    plot_training_history(history, save_path=os.path.join(model_save_path, 'training_curves.png'))
    
    # Evaluate
    print("\nEvaluating on test set...")
    test_loss, test_accuracy, test_precision, test_recall, test_top2 = model.evaluate(
        test_gen, steps=validation_steps, verbose=1
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test Top-2 Accuracy: {test_top2:.4f}")
    print(f"\nModel saved to: {model_save_path}")
    print("="*70)
    
    return model, history, model_save_path


def plot_training_history(history, save_path=None):
    """Vẽ training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train Precision')
        if 'val_precision' in history.history:
            axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Train Recall')
        if 'val_recall' in history.history:
            axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Cloud Classification Model')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['simple', 'resnet50', 'efficientnet', 'mobilenet'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--no_aug', action='store_true', help='Disable data augmentation')
    parser.add_argument('--unfreeze', action='store_true', help='Unfreeze base model')
    parser.add_argument('--save_dir', type=str, default='models', help='Save directory')
    
    args = parser.parse_args()
    
    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_augmentation=not args.no_aug,
        save_dir=args.save_dir,
        freeze_base=not args.unfreeze
    )

