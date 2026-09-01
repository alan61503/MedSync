"""
ResNet-50 Fine-tuning Module for Osteoporosis Detection
Implements training pipeline with data augmentation, weighted loss, and validation
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Avoid torchvision import issues
try:
    from torchvision import transforms
except RuntimeError:
    # Fallback if torchvision has compatibility issues
    class transforms:
        @staticmethod
        def Compose(transform_list):
            return TransformPipeline(transform_list)
        
        @staticmethod
        def RandomRotation(degrees):
            return RandomRotation(degrees)
        
        @staticmethod
        def RandomAffine(degrees, translate=None, scale=None):
            return RandomAffine(degrees, translate, scale)
        
        @staticmethod
        def GaussianBlur(kernel_size, sigma):
            return GaussianBlur(kernel_size, sigma)
        
        @staticmethod
        def RandomVerticalFlip(p):
            return RandomVerticalFlip(p)
        
        @staticmethod
        def Resize(size):
            return Resize(size)
        
        @staticmethod
        def ToTensor():
            return ToTensor()
        
        @staticmethod
        def Normalize(mean, std):
            return Normalize(mean, std)

class TransformPipeline:
    def __init__(self, transforms_list):
        self.transforms = transforms_list
    def __call__(self, x):
        for t in self.transforms:
            x = t(x) if hasattr(t, '__call__') else t
        return x

class Resize:
    def __init__(self, size): self.size = size
    def __call__(self, x): return x.resize(self.size, Image.Resampling.BILINEAR)

class ToTensor:
    def __call__(self, x): return torch.from_numpy(np.array(x, dtype=np.float32)).unsqueeze(0)

class Normalize:
    def __init__(self, mean, std): self.mean = torch.tensor(mean).view(-1, 1, 1); self.std = torch.tensor(std).view(-1, 1, 1)
    def __call__(self, x): return (x - self.mean) / self.std

class RandomRotation:
    def __init__(self, degrees): self.degrees = degrees
    def __call__(self, x):
        angle = np.random.uniform(-self.degrees, self.degrees)
        return x.rotate(angle)

class RandomAffine:
    def __init__(self, degrees, translate=None, scale=None): 
        self.degrees = degrees
        self.translate = translate or (0, 0)
        self.scale = scale or (1, 1)
    def __call__(self, x): return x

class GaussianBlur:
    def __init__(self, kernel_size, sigma): pass
    def __call__(self, x): return x

class RandomVerticalFlip:
    def __init__(self, p): self.p = p
    def __call__(self, x): return x.transpose(Image.Transpose.FLIP_TOP_BOTTOM) if np.random.random() < self.p else x

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Import model building functions
import torch.nn as nn

# Import the ResNet50Custom from xray_service
try:
    from xray_service import _get_resnet50_model
except ImportError:
    # Fallback: define model loading locally
    def _get_resnet50_model():
        """Load ResNet-50 model"""
        from xray_service import _get_resnet50_model as get_model
        return get_model()


class OsteoporosisXrayDataset(Dataset):
    """Custom Dataset for X-ray images with labels"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load grayscale image
        image = Image.open(img_path).convert('L')  # Grayscale
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor
            image = transforms.ToTensor()(image)
        
        return image, torch.tensor(label, dtype=torch.long)


def load_dataset_from_directory(dataset_path: str, split_ratios=(0.7, 0.15, 0.15)):
    """
    Load dataset from directory structure: dataset_path/normal/ and dataset_path/osteoporosis/
    Returns: train_loader, val_loader, test_loader
    """
    normal_dir = os.path.join(dataset_path, 'normal')
    osteoporosis_dir = os.path.join(dataset_path, 'osteoporosis')
    
    # Collect all image paths
    normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    osteo_images = [os.path.join(osteoporosis_dir, f) for f in os.listdir(osteoporosis_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Create labels (0=normal, 1=osteoporosis)
    all_images = normal_images + osteo_images
    all_labels = [0] * len(normal_images) + [1] * len(osteo_images)
    
    # Shuffle together
    indices = np.random.permutation(len(all_images))
    all_images = [all_images[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]
    
    # Split into train/val/test
    n = len(all_images)
    train_size = int(n * split_ratios[0])
    val_size = int(n * split_ratios[1])
    
    train_images, train_labels = all_images[:train_size], all_labels[:train_size]
    val_images, val_labels = all_images[train_size:train_size+val_size], all_labels[train_size:train_size+val_size]
    test_images, test_labels = all_images[train_size+val_size:], all_labels[train_size+val_size:]
    
    print(f"Dataset loaded: {len(all_images)} total images")
    print(f"  Train: {len(train_images)} | Val: {len(val_images)} | Test: {len(test_images)}")
    print(f"  Normal: {all_labels.count(0)} | Osteoporosis: {all_labels.count(1)}")
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale
    ])
    
    # No augmentation for val/test
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_dataset = OsteoporosisXrayDataset(train_images, train_labels, train_transform)
    val_dataset = OsteoporosisXrayDataset(val_images, val_labels, test_transform)
    test_dataset = OsteoporosisXrayDataset(test_images, test_labels, test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    return train_loader, val_loader, test_loader


def compute_class_weights(train_labels: List[int]) -> torch.Tensor:
    """Compute class weights to handle imbalance"""
    n_normal = sum(1 for l in train_labels if l == 0)
    n_osteo = sum(1 for l in train_labels if l == 1)
    
    # Weight inversely proportional to class frequency
    weight_normal = n_osteo / len(train_labels)
    weight_osteo = n_normal / len(train_labels)
    
    weights = torch.tensor([weight_normal, weight_osteo], dtype=torch.float32)
    print(f"Class weights: normal={weight_normal:.4f}, osteoporosis={weight_osteo:.4f}")
    
    return weights


def train_resnet50(dataset_path: str, output_dir: str = 'backend/models', epochs: int = 50, device: str = None):
    """
    Fine-tune ResNet-50 on osteoporosis dataset
    
    Args:
        dataset_path: Path to dataset with normal/ and osteoporosis/ subdirectories
        output_dir: Directory to save fine-tuned model
        epochs: Number of training epochs
        device: torch device (auto-detect if None)
    """
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    train_loader, val_loader, test_loader = load_dataset_from_directory(dataset_path)
    
    # Get model
    model = _get_resnet50_model()
    model.to(device)
    
    # Compute class weights for weighted loss
    train_labels = []
    for images, labels in train_loader:
        train_labels.extend(labels.numpy())
    class_weights = compute_class_weights(train_labels)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Optimizer - only fine-tune last layers
    optimizer = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 0.001},
        {'params': model.fc.parameters(), 'lr': 0.01}
    ])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_recall': [],
        'best_val_acc': 0.0, 'best_epoch': 0
    }
    
    print(f"\n{'='*60}")
    print(f"Starting ResNet-50 Fine-tuning for {epochs} epochs")
    print(f"{'='*60}\n")
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds, train_true = [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]")
        for images, labels in pbar:
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
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        train_acc = accuracy_score(train_true, train_preds)
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds, val_true = [], []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [VAL]")
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                logits, _ = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(labels.cpu().numpy())
                
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        val_acc = accuracy_score(val_true, val_preds)
        val_recall = recall_score(val_true, val_preds, zero_division=0)
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_recall'].append(val_recall)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Recall: {val_recall:.4f}\n")
        
        # Save best model
        if val_acc > history['best_val_acc']:
            history['best_val_acc'] = val_acc
            history['best_epoch'] = epoch + 1
            model_path = os.path.join(output_dir, 'resnet50_finetuned_best.pt')
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Saved best model (accuracy: {val_acc:.4f})\n")
        
        scheduler.step(val_acc)
    
    # Test phase
    print(f"\n{'='*60}")
    print("Evaluating on Test Set")
    print(f"{'='*60}\n")
    
    model.eval()
    test_preds, test_true = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            logits, _ = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(labels.numpy())
    
    test_acc = accuracy_score(test_true, test_preds)
    test_precision = precision_score(test_true, test_preds, zero_division=0)
    test_recall = recall_score(test_true, test_preds, zero_division=0)
    test_f1 = f1_score(test_true, test_preds, zero_division=0)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1: {test_f1:.4f}\n")
    
    # Save final model
    final_path = os.path.join(output_dir, 'resnet50_finetuned_final.pt')
    torch.save(model.state_dict(), final_path)
    
    # Save history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({k: v for k, v in history.items() if k != 'best_epoch'}, f, indent=2)
    
    print(f"✓ Models saved to {output_dir}")
    print(f"✓ Best model epoch: {history['best_epoch']} (Accuracy: {history['best_val_acc']:.4f})")
    
    return model, history


if __name__ == '__main__':
    dataset_path = 'dataset/expanded_benchmark'
    output_dir = 'backend/models'
    
    print("ResNet-50 Fine-tuning Pipeline")
    print("="*60)
    
    model, history = train_resnet50(dataset_path, output_dir, epochs=50)
    
    print("\n✓ Fine-tuning complete!")
