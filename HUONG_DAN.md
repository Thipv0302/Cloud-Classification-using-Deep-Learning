# 🌥️ Hướng Dẫn Sử Dụng - Cloud Classification Project

## 🚀 Bắt Đầu Nhanh

### Bước 1: Cài đặt

```bash
# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### Bước 2: Khám phá dữ liệu

```bash
# Xem thông tin về dataset
python data_loader.py

# Hoặc visualize dữ liệu chi tiết
python visualize_data.py --all
```

### Bước 3: Training Model

#### Cách đơn giản nhất:
```bash
python quick_start.py
```

#### Hoặc training trực tiếp:
```bash
# Training với ResNet50 (khuyến nghị)
python train.py --model resnet50 --epochs 50 --batch_size 32

# Training với EfficientNet (chính xác cao)
python train.py --model efficientnet --epochs 50

# Training với MobileNet (nhanh, nhẹ)
python train.py --model mobilenet --epochs 50
```

### Bước 4: Đánh giá Model

```bash
# Tìm đường dẫn model vừa train (trong thư mục models/)
python evaluate.py --model models/resnet50_YYYYMMDD_HHMMSS/best_model.h5 --visualize
```

### Bước 5: Dự đoán ảnh mới

```bash
# Dự đoán một ảnh
python predict.py --model models/resnet50_YYYYMMDD_HHMMSS/best_model.h5 --image path/to/image.jpg

# Dự đoán nhiều ảnh trong thư mục
python predict.py --model models/resnet50_YYYYMMDD_HHMMSS/best_model.h5 --dir path/to/images/
```

## 📚 Các File Chính

### 1. `data_loader.py`
- Load và preprocess dữ liệu
- Data augmentation
- Tạo data generators

**Sử dụng:**
```python
from data_loader import CloudDataLoader

loader = CloudDataLoader()
info = loader.get_data_info()  # Xem thông tin dataset
train_gen, test_gen = loader.get_data_generators(batch_size=32)
```

### 2. `model.py`
- Các kiến trúc model: Simple CNN, ResNet50, EfficientNet, MobileNet
- Transfer learning với pretrained weights

**Sử dụng:**
```python
from model import create_resnet50_model, compile_model

model = create_resnet50_model(num_classes=7, pretrained=True)
model = compile_model(model, learning_rate=0.001)
```

### 3. `train.py`
- Script training với đầy đủ callbacks
- Early stopping, learning rate scheduling
- Tự động lưu model tốt nhất

**Sử dụng:**
```bash
python train.py --model resnet50 --epochs 50 --batch_size 32 --lr 0.001
```

### 4. `evaluate.py`
- Đánh giá model trên test set
- Confusion matrix, classification report
- Visualization predictions

**Sử dụng:**
```bash
python evaluate.py --model path/to/model.h5 --visualize
```

### 5. `predict.py`
- Dự đoán loại mây từ ảnh mới
- Hỗ trợ single image và batch prediction
- Visualization kết quả

**Sử dụng:**
```bash
python predict.py --model path/to/model.h5 --image image.jpg
```

## 🎯 Workflow Đề Xuất

### 1. Lần đầu tiên sử dụng:
```bash
# 1. Khám phá dữ liệu
python visualize_data.py --all

# 2. Quick start (tự động)
python quick_start.py

# 3. Hoặc training thủ công
python train.py --model resnet50 --epochs 30
```

### 2. Sau khi training:
```bash
# 1. Đánh giá model
python evaluate.py --model models/.../best_model.h5 --visualize

# 2. Test trên ảnh mới
python predict.py --model models/.../best_model.h5 --image test.jpg
```

### 3. Fine-tuning (nâng cao):
```bash
# Bước 1: Train với frozen base
python train.py --model resnet50 --epochs 30 --freeze_base

# Bước 2: Unfreeze và fine-tune
python train.py --model resnet50 --epochs 20 --unfreeze --lr 0.0001
```

## 🔧 Tùy Chỉnh

### Thay đổi cấu hình trong `config.py`:
```python
# Kích thước ảnh
IMG_SIZE = (224, 224)

# Training settings
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Model
MODEL_NAME = 'resnet50'
```

### Hoặc dùng command line arguments:
```bash
python train.py \
    --model resnet50 \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --no_aug      # Tắt augmentation
    --unfreeze    # Unfreeze base model
```

## 📊 Kết Quả Sau Training

Sau khi training, bạn sẽ có:

```
models/
└── resnet50_YYYYMMDD_HHMMSS/
    ├── best_model.h5          # Model tốt nhất
    ├── final_model.h5         # Model cuối cùng
    ├── class_mapping.json     # Mapping classes
    ├── training_log.csv       # Log training
    └── training_curves.png    # Biểu đồ training
```

Sau khi evaluate:

```
evaluation_results/
├── confusion_matrix.png       # Ma trận confusion
├── class_metrics.png          # Metrics từng class
├── prediction_samples.png     # Mẫu predictions
└── evaluation_report.txt      # Báo cáo chi tiết
```

## 💡 Tips & Tricks

### 1. Cải thiện độ chính xác:
- ✅ Sử dụng data augmentation
- ✅ Sử dụng transfer learning (ResNet50, EfficientNet)
- ✅ Fine-tuning với learning rate nhỏ
- ✅ Tăng số epochs (với early stopping)
- ✅ Thử ensemble nhiều models

### 2. Tăng tốc training:
- ✅ Sử dụng GPU (CUDA)
- ✅ Tăng batch_size (nếu đủ RAM)
- ✅ Sử dụng MobileNet (nhẹ hơn)
- ✅ Giảm image size (nếu chấp nhận được)

### 3. Xử lý lỗi thường gặp:

**Out of Memory:**
```bash
# Giảm batch_size
python train.py --model resnet50 --batch_size 16
```

**Training quá chậm:**
```bash
# Sử dụng model nhẹ hơn
python train.py --model mobilenet --epochs 30
```

**Overfitting:**
```bash
# Tăng dropout, sử dụng augmentation
python train.py --model resnet50 --epochs 50
# (augmentation đã được bật mặc định)
```

## 🎓 Học Thêm

### Chạy các ví dụ:
```bash
python example_usage.py
```

Script này sẽ hướng dẫn bạn qua từng bước:
1. Khám phá dữ liệu
2. Load dữ liệu
3. Tạo model
4. Training
5. Đánh giá
6. Dự đoán

## 📝 Ghi Chú

- Model sẽ tự động lưu checkpoint tốt nhất
- Early stopping sẽ dừng training nếu không cải thiện
- Learning rate sẽ tự động giảm khi cần
- Tất cả kết quả đều được lưu tự động

## 🆘 Hỗ Trợ

Nếu gặp vấn đề:
1. Kiểm tra requirements đã cài đủ chưa
2. Kiểm tra đường dẫn data đúng chưa
3. Xem log lỗi trong terminal
4. Thử với model đơn giản hơn (simple CNN)

---

**Chúc bạn thành công! 🌟**

