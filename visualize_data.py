"""
Script để visualize và phân tích dữ liệu
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from data_loader import CloudDataLoader


def analyze_dataset():
    """Phân tích dataset và tạo visualizations"""
    loader = CloudDataLoader()
    info = loader.get_data_info()
    
    # Thống kê số lượng ảnh
    train_counts = []
    test_counts = []
    classes = []
    
    for cls in loader.get_classes():
        train_path = os.path.join(loader.train_dir, cls)
        test_path = os.path.join(loader.test_dir, cls)
        train_count = len([f for f in os.listdir(train_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        test_count = len([f for f in os.listdir(test_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        train_counts.append(train_count)
        test_counts.append(test_count)
        classes.append(cls)
    
    # Vẽ biểu đồ
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Bar chart - số lượng ảnh
    x = np.arange(len(classes))
    width = 0.35
    axes[0, 0].bar(x - width/2, train_counts, width, label='Train', alpha=0.8)
    axes[0, 0].bar(x + width/2, test_counts, width, label='Test', alpha=0.8)
    axes[0, 0].set_xlabel('Class', fontsize=12)
    axes[0, 0].set_ylabel('Number of Images', fontsize=12)
    axes[0, 0].set_title('Dataset Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(classes, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Pie chart - tỷ lệ train/test
    total_train = sum(train_counts)
    total_test = sum(test_counts)
    axes[0, 1].pie([total_train, total_test], 
                   labels=['Train', 'Test'],
                   autopct='%1.1f%%',
                   startangle=90,
                   colors=['#66b3ff', '#ff9999'])
    axes[0, 1].set_title('Train/Test Split', fontsize=14, fontweight='bold')
    
    # Stacked bar chart
    axes[1, 0].bar(classes, train_counts, label='Train', alpha=0.8)
    axes[1, 0].bar(classes, test_counts, bottom=train_counts, label='Test', alpha=0.8)
    axes[1, 0].set_xlabel('Class', fontsize=12)
    axes[1, 0].set_ylabel('Number of Images', fontsize=12)
    axes[1, 0].set_title('Stacked Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticklabels(classes, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Summary statistics
    axes[1, 1].axis('off')
    stats_text = f"""
    DATASET STATISTICS
    
    Total Classes: {len(classes)}
    Total Images: {total_train + total_test}
    
    Training Set:
      - Total: {total_train} images
      - Average per class: {total_train/len(classes):.1f}
      - Min: {min(train_counts)} ({classes[train_counts.index(min(train_counts))]})
      - Max: {max(train_counts)} ({classes[train_counts.index(max(train_counts))]})
    
    Test Set:
      - Total: {total_test} images
      - Average per class: {total_test/len(classes):.1f}
      - Min: {min(test_counts)} ({classes[test_counts.index(min(test_counts))]})
      - Max: {max(test_counts)} ({classes[test_counts.index(max(test_counts))]})
    
    Train/Test Ratio: {total_train/total_test:.2f}:1
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, 
                    verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Đã lưu dataset_analysis.png")
    plt.show()


def visualize_class_samples(num_samples=5):
    """Visualize nhiều mẫu từ mỗi class"""
    loader = CloudDataLoader()
    classes = loader.get_classes()
    
    fig, axes = plt.subplots(len(classes), num_samples, 
                            figsize=(3*num_samples, 3*len(classes)))
    if len(classes) == 1:
        axes = axes.reshape(1, -1)
    
    for class_idx, cls in enumerate(classes):
        train_path = os.path.join(loader.train_dir, cls)
        images = [f for f in os.listdir(train_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for sample_idx in range(num_samples):
            if sample_idx < len(images):
                img_path = os.path.join(train_path, images[sample_idx])
                img = Image.open(img_path)
                axes[class_idx, sample_idx].imshow(img)
                if sample_idx == 0:
                    axes[class_idx, sample_idx].set_ylabel(cls, fontsize=10, fontweight='bold')
                axes[class_idx, sample_idx].axis('off')
    
    plt.suptitle('Sample Images from Each Class', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('class_samples.png', dpi=150, bbox_inches='tight')
    print("✓ Đã lưu class_samples.png")
    plt.show()


def analyze_image_sizes():
    """Phân tích kích thước ảnh trong dataset"""
    loader = CloudDataLoader()
    classes = loader.get_classes()
    
    widths = []
    heights = []
    
    for cls in classes:
        train_path = os.path.join(loader.train_dir, cls)
        images = [f for f in os.listdir(train_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in images[:10]:  # Sample một số ảnh
            img_path = os.path.join(train_path, img_file)
            try:
                img = Image.open(img_path)
                widths.append(img.width)
                heights.append(img.height)
            except Exception as e:
                # Skip corrupted or unreadable images
                continue
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(widths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Width (pixels)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Image Width Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(heights, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1].set_xlabel('Height (pixels)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Image Height Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('image_size_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Đã lưu image_size_analysis.png")
    print(f"\nThống kê kích thước ảnh:")
    print(f"  Width: min={min(widths)}, max={max(widths)}, avg={np.mean(widths):.1f}")
    print(f"  Height: min={min(heights)}, max={max(heights)}, avg={np.mean(heights):.1f}")
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize and analyze dataset')
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    parser.add_argument('--stats', action='store_true', help='Dataset statistics')
    parser.add_argument('--samples', action='store_true', help='Visualize samples')
    parser.add_argument('--sizes', action='store_true', help='Analyze image sizes')
    
    args = parser.parse_args()
    
    if args.all or args.stats:
        analyze_dataset()
    
    if args.all or args.samples:
        visualize_class_samples(num_samples=5)
    
    if args.all or args.sizes:
        analyze_image_sizes()
    
    if not any([args.all, args.stats, args.samples, args.sizes]):
        print("Chạy tất cả analyses...")
        analyze_dataset()
        visualize_class_samples()
        analyze_image_sizes()

