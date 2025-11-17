"""
Test Script - Kiểm tra setup và dữ liệu
"""

import os
import sys

def test_imports():
    """Kiểm tra các thư viện đã cài đặt"""
    print("="*60)
    print("KIỂM TRA THƯ VIỆN")
    print("="*60)
    
    required_packages = {
        'tensorflow': 'TensorFlow',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn',
        'PIL': 'Pillow',
        'seaborn': 'Seaborn'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - CHƯA CÀI ĐẶT")
            missing.append(name)
    
    if missing:
        print(f"\n⚠ Thiếu {len(missing)} thư viện. Chạy: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ Tất cả thư viện đã được cài đặt!")
        return True


def test_data_structure():
    """Kiểm tra cấu trúc dữ liệu"""
    print("\n" + "="*60)
    print("KIỂM TRA DỮ LIỆU")
    print("="*60)
    
    train_dir = 'clouds_train'
    test_dir = 'clouds_test'
    
    if not os.path.exists(train_dir):
        print(f"✗ Không tìm thấy {train_dir}")
        return False
    
    if not os.path.exists(test_dir):
        print(f"✗ Không tìm thấy {test_dir}")
        return False
    
    print(f"✓ Tìm thấy {train_dir}")
    print(f"✓ Tìm thấy {test_dir}")
    
    # Kiểm tra classes
    try:
        from data_loader import CloudDataLoader
        loader = CloudDataLoader()
        classes = loader.get_classes()
        
        if len(classes) == 0:
            print("✗ Không tìm thấy classes nào")
            return False
        
        print(f"\n✓ Tìm thấy {len(classes)} classes:")
        for cls in classes:
            train_path = os.path.join(train_dir, cls)
            test_path = os.path.join(test_dir, cls)
            train_count = len([f for f in os.listdir(train_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(train_path) else 0
            test_count = len([f for f in os.listdir(test_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(test_path) else 0
            print(f"  - {cls}: {train_count} train, {test_count} test")
        
        return True
    except Exception as e:
        print(f"✗ Lỗi khi kiểm tra dữ liệu: {e}")
        return False


def test_data_loader():
    """Kiểm tra data loader"""
    print("\n" + "="*60)
    print("KIỂM TRA DATA LOADER")
    print("="*60)
    
    try:
        from data_loader import CloudDataLoader
        
        loader = CloudDataLoader()
        info = loader.get_data_info()
        
        print("✓ Data loader hoạt động tốt!")
        return True
    except Exception as e:
        print(f"✗ Lỗi: {e}")
        return False


def test_model_creation():
    """Kiểm tra tạo model"""
    print("\n" + "="*60)
    print("KIỂM TRA MODEL CREATION")
    print("="*60)
    
    try:
        from model import create_simple_cnn, compile_model
        
        print("Đang tạo Simple CNN model...")
        model = create_simple_cnn(num_classes=7)
        model = compile_model(model)
        
        print("✓ Model creation hoạt động tốt!")
        return True
    except Exception as e:
        print(f"✗ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu():
    """Kiểm tra GPU"""
    print("\n" + "="*60)
    print("KIỂM TRA GPU")
    print("="*60)
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 0:
            print(f"✓ Tìm thấy {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
        else:
            print("⚠ Không tìm thấy GPU. Sẽ sử dụng CPU (chậm hơn).")
        
        return True
    except Exception as e:
        print(f"⚠ Không thể kiểm tra GPU: {e}")
        return True  # Không phải lỗi nghiêm trọng


def main():
    """Chạy tất cả các test"""
    print("\n" + "="*60)
    print("CLOUD CLASSIFICATION - SETUP TEST")
    print("="*60)
    
    results = {
        'imports': test_imports(),
        'data': test_data_structure(),
        'data_loader': test_data_loader(),
        'model': test_model_creation(),
        'gpu': test_gpu()
    }
    
    print("\n" + "="*60)
    print("KẾT QUẢ KIỂM TRA")
    print("="*60)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name.upper():20} {status}")
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 Tất cả kiểm tra đều PASS! Bạn có thể bắt đầu training.")
        print("\nChạy lệnh sau để bắt đầu:")
        print("  python quick_start.py")
        print("  hoặc")
        print("  python train.py --model resnet50 --epochs 50")
    else:
        print("\n⚠ Một số kiểm tra FAIL. Vui lòng sửa lỗi trước khi tiếp tục.")
        if not results['imports']:
            print("\nCài đặt thư viện:")
            print("  pip install -r requirements.txt")
        if not results['data']:
            print("\nKiểm tra lại cấu trúc dữ liệu trong clouds_train/ và clouds_test/")
    
    print("="*60)


if __name__ == "__main__":
    main()


