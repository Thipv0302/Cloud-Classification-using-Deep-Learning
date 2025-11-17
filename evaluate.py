"""
Evaluation Script - Đánh giá model và tạo visualizations
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
from PIL import Image

from data_loader import CloudDataLoader


def load_model_and_mapping(model_path):
    """Load model và class mapping"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        model = keras.models.load_model(model_path)
    except Exception as e:
        raise ValueError(f"Error loading model from {model_path}: {e}")
    
    # Load class mapping
    mapping_path = os.path.join(os.path.dirname(model_path), 'class_mapping.json')
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
            class_to_idx = mapping['class_to_idx']
            idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}
            classes = mapping['classes']
        except Exception as e:
            print(f"Warning: Error loading class mapping: {e}. Using fallback from data loader.")
            loader = CloudDataLoader()
            loader.get_classes()
            class_to_idx = loader.class_to_idx
            idx_to_class = loader.idx_to_class
            classes = loader.classes
    else:
        # Fallback: lấy từ data loader
        print(f"Warning: Class mapping file not found at {mapping_path}. Using fallback from data loader.")
        loader = CloudDataLoader()
        loader.get_classes()
        class_to_idx = loader.class_to_idx
        idx_to_class = loader.idx_to_class
        classes = loader.classes
    
    return model, class_to_idx, idx_to_class, classes


def evaluate_model(model_path, test_dir='clouds_test', save_dir='evaluation_results'):
    """
    Đánh giá model trên test set
    
    Args:
        model_path: Đường dẫn đến model file
        test_dir: Thư mục test data
        save_dir: Thư mục lưu kết quả
    """
    print("="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    print("\n[1/4] Loading model...")
    model, class_to_idx, idx_to_class, classes = load_model_and_mapping(model_path)
    print(f"Model loaded: {model_path}")
    print(f"Classes: {classes}")
    
    # Load test data
    print("\n[2/4] Loading test data...")
    loader = CloudDataLoader(test_dir=test_dir)
    test_gen, _ = loader.get_data_generators(batch_size=32, use_augmentation=False)
    
    # Predictions
    print("\n[3/4] Making predictions...")
    y_true = []
    y_pred = []
    image_paths = []
    
    test_gen.reset()
    for i in range(len(test_gen)):
        batch_x, batch_y = test_gen[i]
        predictions = model.predict(batch_x, verbose=0)
        
        y_true.extend(np.argmax(batch_y, axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))
        
        # Lấy image paths (nếu có)
        if hasattr(test_gen, 'filepaths'):
            start_idx = i * test_gen.batch_size
            end_idx = min(start_idx + len(batch_x), len(test_gen.filepaths))
            image_paths.extend(test_gen.filepaths[start_idx:end_idx])
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Metrics
    print("\n[4/4] Calculating metrics...")
    accuracy = np.mean(y_true == y_pred)
    
    # Classification report
    class_names = [idx_to_class[i] for i in range(len(classes))]
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Visualizations
    plot_confusion_matrix(cm, class_names, 
                         save_path=os.path.join(save_dir, 'confusion_matrix.png'))
    
    plot_class_metrics(report, class_names,
                      save_path=os.path.join(save_dir, 'class_metrics.png'))
    
    # Save detailed report
    with open(os.path.join(save_dir, 'evaluation_report.txt'), 'w') as f:
        f.write("="*70 + "\n")
        f.write("DETAILED EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names))
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
    
    print(f"\nResults saved to: {save_dir}")
    print("="*70)
    
    return {
        'accuracy': accuracy,
        'y_true': y_true,
        'y_pred': y_pred,
        'confusion_matrix': cm,
        'classification_report': report
    }


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Vẽ confusion matrix"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def plot_class_metrics(report, class_names, save_path=None):
    """Vẽ metrics cho từng class"""
    metrics = ['precision', 'recall', 'f1-score']
    data = {metric: [report[cls][metric] for cls in class_names] 
            for metric in metrics}
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, metric in enumerate(metrics):
        offset = (i - 1) * width
        ax.bar(x + offset, data[metric], width, label=metric.capitalize())
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Metrics per Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Class metrics saved to: {save_path}")
    plt.close()


def visualize_predictions(model_path, test_dir='clouds_test', 
                         num_samples=16, save_path='prediction_samples.png'):
    """
    Visualize một số predictions với ảnh
    
    Args:
        model_path: Đường dẫn model
        test_dir: Thư mục test
        num_samples: Số lượng mẫu để visualize
        save_path: Đường dẫn lưu ảnh
    """
    print("\nVisualizing predictions...")
    
    model, _, idx_to_class, classes = load_model_and_mapping(model_path)
    loader = CloudDataLoader(test_dir=test_dir)
    test_gen, _ = loader.get_data_generators(batch_size=1, use_augmentation=False)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    test_gen.reset()
    for i in range(min(num_samples, len(test_gen))):
        batch_x, batch_y = test_gen[i]
        prediction = model.predict(batch_x, verbose=0)[0]
        true_label = np.argmax(batch_y[0])
        pred_label = np.argmax(prediction)
        confidence = prediction[pred_label]
        
        # Hiển thị ảnh
        img = batch_x[0]
        axes[i].imshow(img)
        
        # Title với kết quả
        true_name = idx_to_class[true_label]
        pred_name = idx_to_class[pred_label]
        color = 'green' if true_label == pred_label else 'red'
        
        title = f"True: {true_name}\nPred: {pred_name} ({confidence:.2f})"
        axes[i].set_title(title, fontsize=10, color=color, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Prediction samples saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Cloud Classification Model')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to model file')
    parser.add_argument('--test_dir', type=str, default='clouds_test',
                       help='Test directory')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                       help='Save directory for results')
    parser.add_argument('--visualize', action='store_true',
                       help='Create prediction visualizations')
    
    args = parser.parse_args()
    
    results = evaluate_model(args.model, args.test_dir, args.save_dir)
    
    if args.visualize:
        visualize_predictions(
            args.model, 
            args.test_dir,
            save_path=os.path.join(args.save_dir, 'prediction_samples.png')
        )

