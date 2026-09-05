"""
Test fine-tuned ResNet-50 binary model on benchmark dataset
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class ResNet50Binary(nn.Module):
    """ResNet-50 for binary osteoporosis classification"""
    
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
    
    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(self.ResNetBlock(in_channels, out_channels * 4, stride, expansion=4))
        for _ in range(1, blocks):
            layers.append(self.ResNetBlock(out_channels * 4, out_channels * 4, stride=1, expansion=4))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits


def test_finetuned_model():
    """Run fine-tuned model on benchmark dataset"""
    
    print("\n" + "="*70)
    print("Testing Fine-tuned ResNet-50 Binary Model")
    print("="*70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load model
    model = ResNet50Binary(num_classes=2, in_channels=1)
    model_path = 'backend/models/resnet50_finetuned.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✓ Loaded fine-tuned model from {model_path}\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    model.to(device)
    model.eval()
    
    # Load dataset
    dataset_path = 'dataset/expanded_benchmark'
    normal_files = list(Path(os.path.join(dataset_path, 'normal')).glob('*'))
    osteo_files = list(Path(os.path.join(dataset_path, 'osteoporosis')).glob('*'))
    
    all_files = normal_files + osteo_files
    all_labels = [0] * len(normal_files) + [1] * len(osteo_files)
    
    print(f"📊 Dataset: {len(all_files)} images")
    print(f"   Normal: {len(normal_files)}")
    print(f"   Osteoporosis: {len(osteo_files)}\n")
    
    # Run inference
    preds = []
    probs_list = []
    
    print("Running inference...")
    with torch.no_grad():
        for img_path in tqdm(all_files, total=len(all_files)):
            try:
                # Load image
                img = Image.open(str(img_path)).convert('L')
                img = img.resize((224, 224))
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)
                
                # Normalize
                img_tensor = (img_tensor - 0.5) / 0.5
                
                # Predict
                logits = model(img_tensor)
                pred_prob = torch.softmax(logits, dim=1)
                osteo_prob = pred_prob[0, 1].item()  # Probability of osteoporosis class
                pred_class = torch.argmax(logits, dim=1).item()
                
                preds.append(pred_class)
                probs_list.append(osteo_prob)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                preds.append(0)
                probs_list.append(0.5)
    
    # Compute metrics
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70 + "\n")
    
    accuracy = accuracy_score(all_labels, preds)
    precision = precision_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, probs_list)
    
    tp = sum(1 for p, t in zip(preds, all_labels) if p == 1 and t == 1)
    tn = sum(1 for p, t in zip(preds, all_labels) if p == 0 and t == 0)
    fp = sum(1 for p, t in zip(preds, all_labels) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(preds, all_labels) if p == 0 and t == 1)
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"ROC-AUC:     {roc_auc:.4f}\n")
    
    print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}\n")
    
    # Improvement comparison
    print("="*70)
    print("IMPROVEMENT vs. Original Model (PureDenseNet121)")
    print("="*70)
    print(f"Accuracy:    49.55% → {accuracy*100:.2f}%  (+{(accuracy-0.4955)*100:.2f}%)")
    print(f"Recall:      5.06%  → {recall*100:.2f}%   (+{(recall-0.0506)*100:.2f}%)")
    print(f"Precision:   54.84% → {precision*100:.2f}%  ({(precision-0.5484)*100:+.2f}%)")
    print(f"F1 Score:    9.26%  → {f1*100:.2f}%   (+{(f1-0.0926)*100:.2f}%)")
    print("="*70 + "\n")
    
    # Save results
    results = {
        "model": "ResNet-50 (Fine-tuned)",
        "dataset": "Osteoporosis Knee X-ray (Expanded Benchmark)",
        "total_images": len(all_files),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "improvement": {
            "accuracy_delta": accuracy - 0.4955,
            "recall_delta": recall - 0.0506,
            "precision_delta": precision - 0.5484,
            "f1_delta": f1 - 0.0926,
        }
    }
    
    results_path = 'backend/evaluation/results/finetuned_model_results.json'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {results_path}")


if __name__ == '__main__':
    test_finetuned_model()
