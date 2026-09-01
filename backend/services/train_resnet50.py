"""
Simple ResNet-50 Fine-tuning for Osteoporosis Detection
Minimal dependencies, focus on improving accuracy with weighted loss and data augmentation
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Loading dependencies...")

# Simple data augmentation without torchvision
class SimpleTransform:
    def __init__(self, size=224, augment=False):
        self.size = size
        self.augment = augment
    
    def __call__(self, img):
        # Resize
        img = img.resize((self.size, self.size), Image.Resampling.BILINEAR)
        
        # Simple augmentation for training
        if self.augment:
            # Random vertical flip
            if np.random.random() > 0.8:
                img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            # Random rotation
            if np.random.random() > 0.7:
                angle = np.random.randint(-15, 15)
                img = img.rotate(angle)
        
        # Convert to tensor
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # Add channel dimension
        
        # Normalize
        img_tensor = (img_tensor - 0.5) / 0.5
        
        return img_tensor


class OsteoXrayDataset(Dataset):
    def __init__(self, image_paths, labels, augment=False):
        self.paths = image_paths
        self.labels = labels
        self.transform = SimpleTransform(augment=augment)
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


# Inline ResNet-50 architecture
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super().__init__()
        self.expansion = expansion
        mid_channels = out_channels // expansion
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return torch.relu(out)


class ResNet50(nn.Module):
    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        
        # Classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels * 4, stride, expansion=4))
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels * 4, out_channels * 4, stride=1, expansion=4))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        features = x.flatten(1)
        logits = self.fc(features)
        return logits, features


def load_dataset(dataset_path, split_ratio=0.8):
    """Load dataset from normal/ and osteoporosis/ folders"""
    normal_dir = os.path.join(dataset_path, 'normal')
    osteo_dir = os.path.join(dataset_path, 'osteoporosis')
    
    normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    osteo_files = [os.path.join(osteo_dir, f) for f in os.listdir(osteo_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Combine and shuffle
    all_files = normal_files + osteo_files
    all_labels = [0] * len(normal_files) + [1] * len(osteo_files)
    
    indices = np.random.permutation(len(all_files))
    all_files = [all_files[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]
    
    # Split
    split_idx = int(len(all_files) * split_ratio)
    train_files, train_labels = all_files[:split_idx], all_labels[:split_idx]
    test_files, test_labels = all_files[split_idx:], all_labels[split_idx:]
    
    print(f"\n📊 Dataset: {len(all_files)} images")
    print(f"   Train: {len(train_files)} | Test: {len(test_files)}")
    print(f"   Normal: {all_labels.count(0)} | Osteoporosis: {all_labels.count(1)}")
    
    train_ds = OsteoXrayDataset(train_files, train_labels, augment=True)
    test_ds = OsteoXrayDataset(test_files, test_labels, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    return train_loader, test_loader, train_labels


def train():
    """Train ResNet-50 with weighted loss"""
    print("\n" + "="*70)
    print("ResNet-50 Fine-tuning for Osteoporosis Detection")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    dataset_path = 'dataset/expanded_benchmark'
    output_dir = 'backend/models'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    train_loader, test_loader, train_labels = load_dataset(dataset_path, split_ratio=0.8)
    
    # Model
    print("\n🧠 Building ResNet-50...")
    model = ResNet50(num_classes=2, in_channels=1)
    model.to(device)
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Class weights for imbalanced dataset
    n_normal = train_labels.count(0)
    n_osteo = train_labels.count(1)
    weight_normal = n_osteo / len(train_labels)
    weight_osteo = n_normal / len(train_labels)
    class_weights = torch.tensor([weight_normal, weight_osteo]).to(device)
    print(f"\n⚖️  Class weights: normal={weight_normal:.4f}, osteoporosis={weight_osteo:.4f}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    
    # Training
    best_acc = 0.0
    best_recall = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_recall': []}
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    for epoch in range(50):
        # Train
        model.train()
        train_loss = 0.0
        train_preds, train_true = [], []
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/50 [Train]", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_true.extend(labels.cpu().numpy())
        
        train_acc = accuracy_score(train_true, train_preds)
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validate
        model.eval()
        val_preds, val_true = [], []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/50 [Val]", leave=False):
                images = images.to(device)
                logits, _ = model(images)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(labels.numpy())
        
        val_acc = accuracy_score(val_true, val_preds)
        val_recall = recall_score(val_true, val_preds, zero_division=0)
        val_precision = precision_score(val_true, val_preds, zero_division=0)
        val_f1 = f1_score(val_true, val_preds, zero_division=0)
        
        history['val_acc'].append(val_acc)
        history['val_recall'].append(val_recall)
        
        # Log
        print(f"Epoch {epoch+1:2d}/50 | Loss: {train_loss:.4f} | " + 
              f"Train Acc: {train_acc:.4f} | " +
              f"Val Acc: {val_acc:.4f} | Recall: {val_recall:.4f} | Precision: {val_precision:.4f} | F1: {val_f1:.4f}")
        
        # Save best model
        if val_recall > best_recall or (val_recall == best_recall and val_acc > best_acc):
            best_acc = val_acc
            best_recall = val_recall
            model_path = os.path.join(output_dir, 'resnet50_finetuned.pt')
            torch.save(model.state_dict(), model_path)
            print(f"   ✓ Saved best model (Acc: {best_acc:.4f}, Recall: {best_recall:.4f})")
        
        scheduler.step(val_recall)
    
    # Save history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print(f"   Best Accuracy: {best_acc:.4f}")
    print(f"   Best Recall: {best_recall:.4f}")
    print(f"   Model saved to: {os.path.join(output_dir, 'resnet50_finetuned.pt')}")
    print("="*70 + "\n")


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    train()
