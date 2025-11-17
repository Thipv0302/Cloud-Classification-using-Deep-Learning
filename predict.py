"""
Inference Script - Dự đoán loại mây từ ảnh mới
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import json
import matplotlib.pyplot as plt


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
            idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}
            classes = mapping['classes']
        except Exception as e:
            print(f"Warning: Error loading class mapping: {e}. Using fallback.")
            idx_to_class = {i: f"Class_{i}" for i in range(7)}
            classes = list(idx_to_class.values())
    else:
        # Fallback
        print(f"Warning: Class mapping file not found at {mapping_path}. Using fallback.")
        idx_to_class = {i: f"Class_{i}" for i in range(7)}
        classes = list(idx_to_class.values())
    
    return model, idx_to_class, classes


def preprocess_image(image_path, img_size=(224, 224)):
    """
    Preprocess ảnh để đưa vào model
    
    Args:
        image_path: Đường dẫn ảnh
        img_size: Kích thước ảnh
    
    Returns:
        Preprocessed image array
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize(img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img
    except Exception as e:
        raise ValueError(f"Error preprocessing image {image_path}: {e}")


def predict_image(model_path, image_path, top_k=3, show_image=True):
    """
    Dự đoán loại mây từ ảnh
    
    Args:
        model_path: Đường dẫn model
        image_path: Đường dẫn ảnh
        top_k: Số lượng top predictions
        show_image: Hiển thị ảnh và kết quả
    
    Returns:
        Dictionary chứa predictions
    """
    # Load model
    model, idx_to_class, classes = load_model_and_mapping(model_path)
    
    # Preprocess image
    img_array, original_img = preprocess_image(image_path)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Get top k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_predictions = [
        {
            'class': idx_to_class[idx],
            'confidence': float(predictions[idx])
        }
        for idx in top_indices
    ]
    
    # Print results
    print("="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"\nTop {top_k} Predictions:")
    for i, pred in enumerate(top_predictions, 1):
        print(f"  {i}. {pred['class']}: {pred['confidence']:.4f} ({pred['confidence']*100:.2f}%)")
    print("="*60)
    
    # Visualize
    if show_image:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Predictions bar chart
        classes_list = [p['class'] for p in top_predictions]
        confidences = [p['confidence'] for p in top_predictions]
        
        colors = plt.cm.RdYlGn(confidences)
        bars = axes[1].barh(classes_list, confidences, color=colors)
        axes[1].set_xlabel('Confidence', fontsize=12)
        axes[1].set_title('Top Predictions', fontsize=14, fontweight='bold')
        axes[1].set_xlim([0, 1])
        axes[1].grid(True, alpha=0.3, axis='x')
        
        # Add confidence values on bars
        for i, (bar, conf) in enumerate(zip(bars, confidences)):
            axes[1].text(conf + 0.01, i, f'{conf:.3f}', 
                        va='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.splitext(image_path)[0] + '_prediction.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
        plt.show()
    
    return {
        'image_path': image_path,
        'predictions': top_predictions,
        'top_prediction': top_predictions[0]
    }


def predict_batch(model_path, image_dir, output_file='batch_predictions.txt'):
    """
    Dự đoán cho nhiều ảnh trong một thư mục
    
    Args:
        model_path: Đường dẫn model
        image_dir: Thư mục chứa ảnh
        output_file: File lưu kết quả
    """
    model, idx_to_class, classes = load_model_and_mapping(model_path)
    
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    results = []
    
    print(f"\nProcessing {len(image_files)} images...")
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(image_dir, img_file)
        print(f"[{i}/{len(image_files)}] Processing: {img_file}")
        
        try:
            img_array, _ = preprocess_image(img_path)
            predictions = model.predict(img_array, verbose=0)[0]
            pred_idx = np.argmax(predictions)
            confidence = predictions[pred_idx]
            
            result = {
                'image': img_file,
                'predicted_class': idx_to_class[pred_idx],
                'confidence': float(confidence)
            }
            results.append(result)
            
        except Exception as e:
            print(f"  Error processing {img_file}: {e}")
    
    # Save results
    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("BATCH PREDICTION RESULTS\n")
        f.write("="*60 + "\n\n")
        for result in results:
            f.write(f"Image: {result['image']}\n")
            f.write(f"  Predicted: {result['predicted_class']}\n")
            f.write(f"  Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)\n")
            f.write("\n")
    
    print(f"\nResults saved to: {output_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Cloud Type from Image')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model file')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image')
    parser.add_argument('--dir', type=str, default=None,
                       help='Directory containing images (for batch prediction)')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of top predictions to show')
    parser.add_argument('--no_show', action='store_true',
                       help='Don\'t show visualization')
    
    args = parser.parse_args()
    
    if args.image:
        predict_image(args.model, args.image, top_k=args.top_k, 
                     show_image=not args.no_show)
    elif args.dir:
        predict_batch(args.model, args.dir)
    else:
        print("Please provide either --image or --dir argument")
        parser.print_help()

