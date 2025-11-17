"""
Quick Start Script - Chạy nhanh toàn bộ pipeline
"""

import os
import sys

def main():
    print("="*70)
    print("CLOUD CLASSIFICATION - QUICK START")
    print("="*70)
    
    print("\nBước 1: Khám phá dữ liệu...")
    from data_loader import CloudDataLoader
    loader = CloudDataLoader()
    info = loader.get_data_info()
    
    print("\nBước 2: Visualize samples...")
    try:
        loader.visualize_samples(num_samples=3, save_path='data_samples.png')
        print("✓ Đã lưu data_samples.png")
    except Exception as e:
        print(f"⚠ Lỗi khi visualize: {e}")
    
    print("\nBước 3: Training model...")
    print("Bạn có muốn bắt đầu training không? (y/n): ", end='')
    response = input().strip().lower()
    
    if response == 'y':
        from train import train_model
        print("\nChọn model:")
        print("1. ResNet50 (Khuyến nghị)")
        print("2. EfficientNet")
        print("3. MobileNet")
        print("4. Simple CNN")
        choice = input("Nhập số (1-4): ").strip()
        
        model_map = {'1': 'resnet50', '2': 'efficientnet', 
                    '3': 'mobilenet', '4': 'simple'}
        model_name = model_map.get(choice, 'resnet50')
        
        epochs = input("Số epochs (mặc định 30): ").strip()
        epochs = int(epochs) if epochs else 30
        
        print(f"\nBắt đầu training với {model_name}...")
        model, history, save_path = train_model(
            model_name=model_name,
            epochs=epochs,
            batch_size=32,
            learning_rate=0.001,
            use_augmentation=True
        )
        
        print(f"\n✓ Training hoàn thành! Model đã lưu tại: {save_path}")
        
        print("\nBước 4: Đánh giá model...")
        model_path = os.path.join(save_path, 'best_model.h5')
        if os.path.exists(model_path):
            from evaluate import evaluate_model, visualize_predictions
            results = evaluate_model(model_path, save_dir='evaluation_results')
            visualize_predictions(model_path, save_path='evaluation_results/prediction_samples.png')
            print("\n✓ Đánh giá hoàn thành!")
        else:
            print("⚠ Không tìm thấy best_model.h5")
    else:
        print("\nBỏ qua training. Bạn có thể chạy sau bằng:")
        print("  python train.py --model resnet50 --epochs 50")
    
    print("\n" + "="*70)
    print("QUICK START HOÀN THÀNH!")
    print("="*70)
    print("\nCác lệnh hữu ích:")
    print("  - Training: python train.py --model resnet50 --epochs 50")
    print("  - Evaluate: python evaluate.py --model <model_path>")
    print("  - Predict: python predict.py --model <model_path> --image <image_path>")
    print("="*70)


if __name__ == "__main__":
    main()

