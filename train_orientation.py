import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
import progressbar
import argparse
import imutils
import random
import cv2
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input directory of images")
ap.add_argument("-o", "--output", required=True, help="path to output directory of rotated iamges")
args = vars(ap.parse_args())

# --- CONFIGURATION ---
DATASET_ROOT = args["dataset"]  # Thư mục chứa các ảnh sau khi giải nén
OUTPUT_DIR = args["output"]       # Thư mục chứa các ảnh đã xoay
NUM_EPOCHS = 30                   # Số epochs để huấn luyện, 3-5 là đủ cho bài toán này
BATCH_SIZE = 32                    # Tùy chỉnh theo VRAM của bạn
LEARNING_RATE = 1e-4               # Tốc độ học cho lớp classifier
TRAIN_VAL_SPLIT = 0.9              # Tỷ lệ 90% cho tập huấn luyện, 10% cho tập xác thực

# --- 1. DATASET CLASS ---
# Tạo một Dataset tùy chỉnh để xoay ảnh và gán nhãn giả
class RotationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        # Lấy tất cả đường dẫn ảnh từ các thư mục con
        for category in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                for img_name in os.listdir(category_path):
                    if img_name.lower().endswith('.jpg'):
                        self.image_paths.append(os.path.join(category_path, img_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Mở ảnh gốc
        image = Image.open(img_path).convert('RGB')

        # Chọn ngẫu nhiên một góc xoay: 0, 90, 180, 270 độ
        # Tương ứng với các nhãn 0, 1, 2, 3
        rotation_label = np.random.randint(4)
        rotated_image = image.rotate(rotation_label * 90)

        # Áp dụng các phép biến đổi (resize, normalize)
        if self.transform:
            tensor_image = self.transform(rotated_image)

        return tensor_image, torch.tensor(rotation_label, dtype=torch.long)

# --- 2. MODEL DEFINITION ---
# Định nghĩa mô hình bao gồm DINOv2 và một lớp classifier
class OrientationClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(OrientationClassifier, self).__init__()
        # Tải mô hình DINOv2 đã được huấn luyện sẵn từ PyTorch Hub
        # dinov2_vitb14 là phiên bản "base" với Vision Transformer
        # self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        
        
        # --- QUAN TRỌNG: Đóng băng DINOv2 ---
        # Chúng ta không muốn huấn luyện lại DINOv2, chỉ dùng nó để trích xuất đặc trưng
        for param in self.dinov2.parameters():
            param.requires_grad = False
            
        # Lấy số chiều đặc trưng đầu ra của DINOv2 (với ViT-Base là 768)
        feature_dim = self.dinov2.embed_dim

        # Tạo một lớp classifier đơn giản để phân loại 4 góc xoay
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # Đưa ảnh qua DINOv2 để lấy vector đặc trưng
        # DINOv2 trả về một dict, chúng ta chỉ cần feature cuối cùng
        features = self.dinov2(x)
        # Đưa vector đặc trưng qua lớp classifier
        output = self.classifier(features)
        return output

# --- 3. TRAINING AND VALIDATION LOOP ---
def train_model(train_loader, val_loader, device):
    """
    Hàm huấn luyện model, trả về model đã huấn luyện.
    """
    # Khởi tạo mô hình và chuyển sang device
    model = OrientationClassifier(num_classes=4).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE)

    # Vòng lặp huấn luyện
    for epoch in range(NUM_EPOCHS):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {epoch_loss:.4f}")

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_preds += labels.size(0)
                correct_preds += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = (correct_preds / total_preds) * 100
        logger.info(f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.2f}%")
        logger.info("-" * 30)

    logger.info("Hoàn thành huấn luyện!")
    # Lưu lại chỉ các trọng số của lớp classifier đã huấn luyện
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
     # Lưu mô hình đã huấn luyện
    logger.info("\n--- Lưu mô hình đã huấn luyện ---")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'orientation_classifier_head.pth'))

    logger.info(f"Mô hình đã được lưu tại: {os.path.join(OUTPUT_DIR, 'orientation_classifier_head.pth')}")
    
    return model


# --- 4. INFERENCE FUNCTION ---
# Hàm để kiểm tra và chỉnh sửa hướng của một ảnh mới
def correct_image_orientation(image_path, model, transform):
    device = next(model.parameters()).device
    model.eval()

    # Mở ảnh và xoay nó một góc ngẫu nhiên để thử nghiệm
    original_image = Image.open(image_path).convert('RGB')
    random_rotation_deg = np.random.choice([90, 180, 270])
    test_image = original_image.rotate(random_rotation_deg, expand=True)
    logger.info(f"Ảnh đã được xoay đi {random_rotation_deg} độ để kiểm tra.")

    # Chuẩn bị ảnh đầu vào cho model
    input_tensor = transform(test_image).unsqueeze(0).to(device)

    # Dự đoán
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_label = torch.max(output.data, 1)
        predicted_rotation = predicted_label.item() * 90

    logger.info(f"Model dự đoán ảnh đã bị xoay: {predicted_rotation} độ.")

    # Tính toán góc cần xoay lại để chỉnh sửa
    correction_angle = -predicted_rotation
    corrected_image = test_image.rotate(correction_angle, expand=True)
    logger.info(f"Áp dụng góc xoay {correction_angle} độ để sửa lại.")

    # Hiển thị kết quả
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("Ảnh Gốc (0°)")
    axes[0].axis('off')

    axes[1].imshow(test_image)
    axes[1].set_title(f"Ảnh Bị Xoay ({random_rotation_deg}°)")
    axes[1].axis('off')

    axes[2].imshow(corrected_image)
    axes[2].set_title("Ảnh Đã Chỉnh Sửa")
    axes[2].axis('off')

    plt.show()



# --- NEW EVALUATION FUNCTION ---
def evaluate_model(model, test_loader, device):
    """
    Hàm để đánh giá model trên tập test.
    """
    logger.info("\n--- Bắt đầu đánh giá trên tập Test ---")
    model.eval()  # Chuyển model sang chế độ đánh giá
    
    all_labels = []
    all_predictions = []

    # Không tính toán gradient trong quá trình đánh giá
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Tính toán các chỉ số
    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, target_names=['0°', '90°', '180°', '270°'])
    conf_matrix = confusion_matrix(all_labels, all_predictions)
 
    logger.info(f"\n✅ Độ chính xác trên tập Test: {accuracy * 100:.2f}%")
    logger.info("\n📊 Báo cáo phân loại chi tiết:")
    logger.info(report)
    logger.info("\n📈 Ma trận nhầm lẫn:")
    logger.info(conf_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['0°', '90°', '180°', '270°'], 
                yticklabels=['0°', '90°', '180°', '270°'])
    plt.xlabel('Nhãn dự đoán (Predicted Label)')
    plt.ylabel('Nhãn thật (True Label)')
    plt.title('Confusion Matrix')
    plt.show()
    
# --- MAIN EXECUTION (UPDATED) ---
if __name__ == '__main__':
    # Xác định thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Sử dụng thiết bị: {device}")

    # Định nghĩa các phép biến đổi ảnh
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Khởi tạo dataset
    full_dataset = RotationDataset(root_dir=DATASET_ROOT, transform=transform)
    
    # --- CẬP NHẬT: Chia dữ liệu thành 3 phần: 80% Train, 10% Val, 10% Test ---
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    logger.info(f"Kích thước tập huấn luyện: {len(train_dataset)}")
    logger.info(f"Kích thước tập xác thực:  {len(val_dataset)}")
    logger.info(f"Kích thước tập kiểm tra:   {len(test_dataset)}")

    # Tạo DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Save the dataloader
    if not os.path.exists(f'{OUTPUT_DIR}/data_split'):
        os.makedirs(f'{OUTPUT_DIR}/data_split')
    torch.save(train_loader, f'{OUTPUT_DIR}/data_split/train_loader.pt')
    torch.save(val_loader, f'{OUTPUT_DIR}/data_split/val_loader.pt')
    torch.save(test_loader, f'{OUTPUT_DIR}/data_split/test_loader.pt')

    # 1. Huấn luyện mô hình
    trained_model = train_model(train_loader, val_loader, device)
    
    # 2. Đánh giá mô hình trên tập Test
    evaluate_model(trained_model, test_loader, device)

    # 3. Thử nghiệm chỉnh sửa một ảnh
    logger.info("\n--- Thử nghiệm chỉnh sửa một ảnh ngẫu nhiên ---")
    # test_image_path = os.path.join(DATASET_ROOT, 'office', 'image_0010.jpg') 
    # test_image_path = random.choice(test_dataset.image_paths)  # Chọn ngẫu nhiên một ảnh từ tập test
    test_image_path = '/content/image_orientation/indoor_cvpr/Images/airport_inside/airport_inside_0001.jpg'
    correct_image_orientation(test_image_path, trained_model, transform)