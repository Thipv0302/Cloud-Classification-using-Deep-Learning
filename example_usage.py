"""
Example Usage - Ví dụ cách sử dụng các module
"""

import os
from data_loader import CloudDataLoader
from model import create_resnet50_model, compile_model
from train import train_model
from evaluate import evaluate_model, visualize_predictions
from predict import predict_image


def example_1_explore_data():
    """Ví dụ 1: Khám phá dữ liệu"""
    print("\n" + "="*60)
    print("VÍ DỤ 1: Khám phá dữ liệu")
    print("="*60)
    
    loader = CloudDataLoader()
    info = loader.get_data_info()
    loader.visualize_samples(num_samples=3, save_path='example_data_samples.png')
    
    print("\n✓ Hoàn thành!")


def example_2_load_data():
    """Ví dụ 2: Load dữ liệu"""
    print("\n" + "="*60)
    print("VÍ DỤ 2: Load dữ liệu")
    print("="*60)
    
    loader = CloudDataLoader()
    train_gen, test_gen = loader.get_data_generators(
        batch_size=32, 
        use_augmentation=True
    )
    
    print(f"Số lượng classes: {len(loader.classes)}")
    print(f"Classes: {loader.classes}")
    print(f"Train samples: {train_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    
    # Lấy một batch
    batch_x, batch_y = train_gen[0]
    print(f"\nBatch shape: {batch_x.shape}")
    print(f"Labels shape: {batch_y.shape}")
    
    print("\n✓ Hoàn thành!")


def example_3_create_model():
    """Ví dụ 3: Tạo model"""
    print("\n" + "="*60)
    print("VÍ DỤ 3: Tạo model")
    print("="*60)
    
    # Tạo ResNet50 model
    model = create_resnet50_model(
        input_shape=(224, 224, 3),
        num_classes=7,
        pretrained=True,
        freeze_base=True
    )
    
    # Compile
    model = compile_model(model, learning_rate=0.001)
    
    # In summary
    print("\nModel Summary:")
    model.summary()
    
    print("\n✓ Hoàn thành!")


def example_4_training():
    """Ví dụ 4: Training (chạy nhanh với ít epochs)"""
    print("\n" + "="*60)
    print("VÍ DỤ 4: Training model")
    print("="*60)
    print("⚠ Lưu ý: Training sẽ mất thời gian!")
    print("Bạn có muốn tiếp tục? (y/n): ", end='')
    
    response = input().strip().lower()
    if response != 'y':
        print("Bỏ qua training.")
        return
    
    model, history, save_path = train_model(
        model_name='resnet50',
        epochs=5,  # Chỉ train 5 epochs để demo
        batch_size=32,
        learning_rate=0.001,
        use_augmentation=True
    )
    
    print(f"\n✓ Training hoàn thành! Model đã lưu tại: {save_path}")


def example_5_evaluation():
    """Ví dụ 5: Đánh giá model"""
    print("\n" + "="*60)
    print("VÍ DỤ 5: Đánh giá model")
    print("="*60)
    
    # Tìm model mới nhất
    models_dir = 'models'
    if not os.path.exists(models_dir):
        print("⚠ Không tìm thấy thư mục models. Vui lòng train model trước.")
        return
    
    # Tìm model mới nhất
    model_dirs = [d for d in os.listdir(models_dir) 
                 if os.path.isdir(os.path.join(models_dir, d))]
    if not model_dirs:
        print("⚠ Không tìm thấy model nào. Vui lòng train model trước.")
        return
    
    latest_model_dir = sorted(model_dirs)[-1]
    model_path = os.path.join(models_dir, latest_model_dir, 'best_model.h5')
    
    if not os.path.exists(model_path):
        print(f"⚠ Không tìm thấy {model_path}")
        return
    
    print(f"Đang đánh giá model: {model_path}")
    results = evaluate_model(model_path, save_dir='example_evaluation')
    visualize_predictions(model_path, save_path='example_evaluation/prediction_samples.png')
    
    print("\n✓ Đánh giá hoàn thành!")


def example_6_prediction():
    """Ví dụ 6: Dự đoán trên ảnh mới"""
    print("\n" + "="*60)
    print("VÍ DỤ 6: Dự đoán trên ảnh mới")
    print("="*60)
    
    # Tìm model mới nhất
    models_dir = 'models'
    if not os.path.exists(models_dir):
        print("⚠ Không tìm thấy thư mục models.")
        return
    
    model_dirs = [d for d in os.listdir(models_dir) 
                 if os.path.isdir(os.path.join(models_dir, d))]
    if not model_dirs:
        print("⚠ Không tìm thấy model nào.")
        return
    
    latest_model_dir = sorted(model_dirs)[-1]
    model_path = os.path.join(models_dir, latest_model_dir, 'best_model.h5')
    
    if not os.path.exists(model_path):
        print(f"⚠ Không tìm thấy {model_path}")
        return
    
    # Tìm một ảnh test để demo
    test_dir = 'clouds_test'
    if os.path.exists(test_dir):
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, file)
                    print(f"\nDự đoán cho ảnh: {image_path}")
                    result = predict_image(model_path, image_path, top_k=3, show_image=True)
                    return
    
    print("⚠ Không tìm thấy ảnh test để demo.")


def main():
    """Chạy tất cả các ví dụ"""
    print("="*60)
    print("CLOUD CLASSIFICATION - EXAMPLE USAGE")
    print("="*60)
    
    examples = {
        '1': ('Khám phá dữ liệu', example_1_explore_data),
        '2': ('Load dữ liệu', example_2_load_data),
        '3': ('Tạo model', example_3_create_model),
        '4': ('Training (cần thời gian)', example_4_training),
        '5': ('Đánh giá model', example_5_evaluation),
        '6': ('Dự đoán ảnh', example_6_prediction),
    }
    
    print("\nChọn ví dụ để chạy:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("  all. Chạy tất cả (trừ training)")
    print("  q. Thoát")
    
    choice = input("\nNhập lựa chọn: ").strip().lower()
    
    if choice == 'q':
        return
    elif choice == 'all':
        for key, (_, func) in examples.items():
            if key != '4':  # Bỏ qua training
                try:
                    func()
                except Exception as e:
                    print(f"⚠ Lỗi: {e}")
    elif choice in examples:
        try:
            examples[choice][1]()
        except Exception as e:
            print(f"⚠ Lỗi: {e}")
    else:
        print("Lựa chọn không hợp lệ!")


if __name__ == "__main__":
    main()

