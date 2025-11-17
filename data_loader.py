"""
Data Loader và Preprocessing cho Cloud Classification
Tải và xử lý dữ liệu ảnh mây từ thư mục
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


class CloudDataLoader:
    """Class để load và preprocess dữ liệu ảnh mây"""
    
    def __init__(self, train_dir='clouds_train', test_dir='clouds_test', img_size=(224, 224)):
        """
        Khởi tạo DataLoader
        
        Args:
            train_dir: Đường dẫn thư mục training data
            test_dir: Đường dẫn thư mục test data
            img_size: Kích thước ảnh sau khi resize (width, height)
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.img_size = img_size
        self.classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
    def get_classes(self):
        """Lấy danh sách các class từ thư mục train"""
        if os.path.exists(self.train_dir):
            self.classes = sorted([d for d in os.listdir(self.train_dir) 
                                  if os.path.isdir(os.path.join(self.train_dir, d))])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        return self.classes
    
    def get_data_info(self):
        """In thông tin về dataset"""
        self.get_classes()
        print("=" * 60)
        print("THÔNG TIN DATASET")
        print("=" * 60)
        print(f"\nSố lượng classes: {len(self.classes)}")
        print(f"\nDanh sách classes:")
        for idx, cls in enumerate(self.classes):
            train_count = len(os.listdir(os.path.join(self.train_dir, cls)))
            test_count = len(os.listdir(os.path.join(self.test_dir, cls)))
            print(f"  {idx}: {cls}")
            print(f"     - Train: {train_count} ảnh")
            print(f"     - Test: {test_count} ảnh")
        
        # Tính tổng
        total_train = sum(len(os.listdir(os.path.join(self.train_dir, cls))) 
                         for cls in self.classes)
        total_test = sum(len(os.listdir(os.path.join(self.test_dir, cls))) 
                        for cls in self.classes)
        print(f"\nTổng số ảnh:")
        print(f"  - Train: {total_train} ảnh")
        print(f"  - Test: {total_test} ảnh")
        print("=" * 60)
        
        return {
            'classes': self.classes,
            'num_classes': len(self.classes),
            'total_train': total_train,
            'total_test': total_test
        }
    
    def load_images_from_folder(self, folder_path, label):
        """Load tất cả ảnh từ một folder"""
        images = []
        labels = []
        image_paths = []
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, filename)
                try:
                    img = Image.open(img_path)
                    img = img.convert('RGB')  # Đảm bảo ảnh là RGB
                    img = img.resize(self.img_size)
                    img_array = np.array(img) / 255.0  # Normalize về [0, 1]
                    images.append(img_array)
                    labels.append(label)
                    image_paths.append(img_path)
                except Exception as e:
                    print(f"Lỗi khi load ảnh {img_path}: {e}")
        
        return images, labels, image_paths
    
    def load_data(self, use_test_split=False, val_split=0.2):
        """
        Load toàn bộ dữ liệu
        
        Args:
            use_test_split: Nếu True, chia train thành train/val
            val_split: Tỷ lệ validation nếu use_test_split=True
        
        Returns:
            Tuple chứa (X_train, y_train, X_test, y_test) hoặc 
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        self.get_classes()
        
        # Load training data
        X_train = []
        y_train = []
        train_paths = []
        
        for cls in self.classes:
            cls_path = os.path.join(self.train_dir, cls)
            images, labels, paths = self.load_images_from_folder(
                cls_path, self.class_to_idx[cls]
            )
            X_train.extend(images)
            y_train.extend(labels)
            train_paths.extend(paths)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Load test data
        X_test = []
        y_test = []
        test_paths = []
        
        for cls in self.classes:
            cls_path = os.path.join(self.test_dir, cls)
            images, labels, paths = self.load_images_from_folder(
                cls_path, self.class_to_idx[cls]
            )
            X_test.extend(images)
            y_test.extend(labels)
            test_paths.extend(paths)
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # Convert labels to categorical
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(self.classes))
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(self.classes))
        
        if use_test_split:
            # Chia train thành train/val
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_split, random_state=42, stratify=y_train
            )
            return X_train, y_train, X_val, y_val, X_test, y_test
        
        return X_train, y_train, X_test, y_test
    
    def get_data_generators(self, batch_size=32, use_augmentation=True):
        """
        Tạo data generators với augmentation cho training
        
        Args:
            batch_size: Kích thước batch
            use_augmentation: Có sử dụng data augmentation không
        
        Returns:
            Tuple (train_gen, test_gen)
        """
        if use_augmentation:
            train_datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest',
                brightness_range=[0.8, 1.2]
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Lưu class mapping
        self.class_to_idx = train_generator.class_indices
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = list(self.class_to_idx.keys())
        
        return train_generator, test_generator
    
    def visualize_samples(self, num_samples=5, save_path=None):
        """Visualize một số mẫu ảnh từ mỗi class"""
        self.get_classes()
        
        fig, axes = plt.subplots(len(self.classes), num_samples, 
                                figsize=(15, 3*len(self.classes)))
        if len(self.classes) == 1:
            axes = axes.reshape(1, -1)
        
        for class_idx, cls in enumerate(self.classes):
            cls_path = os.path.join(self.train_dir, cls)
            images = [f for f in os.listdir(cls_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for sample_idx in range(num_samples):
                if sample_idx < len(images):
                    img_path = os.path.join(cls_path, images[sample_idx])
                    img = Image.open(img_path)
                    axes[class_idx, sample_idx].imshow(img)
                    axes[class_idx, sample_idx].set_title(f'{cls}')
                    axes[class_idx, sample_idx].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Test data loader
    loader = CloudDataLoader()
    info = loader.get_data_info()
    loader.visualize_samples(num_samples=3, save_path='data_samples.png')

