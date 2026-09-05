#!/usr/bin/env python3
"""Inference module for fine-tuned ResNet-50 binary osteoporosis classifier
Uses exact same architecture as training script for guaranteed compatibility
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path


class ResNetBlock(nn.Module):
    """Bottleneck residual block with expansion=4"""
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


class ResNet50Binary(nn.Module):
    """ResNet-50 binary classifier for osteoporosis detection"""
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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Global model instance
_FINETUNED_MODEL = None
_FINETUNED_DEVICE = None


def load_finetuned_model():
    """Load fine-tuned ResNet-50 model once"""
    global _FINETUNED_MODEL, _FINETUNED_DEVICE
    if _FINETUNED_MODEL is not None:
        return _FINETUNED_MODEL, _FINETUNED_DEVICE
    
    try:
        device = torch.device('cpu')
        model = ResNet50Binary(num_classes=2, in_channels=1)
        model_path = Path(__file__).resolve().parent.parent / "models" / "resnet50_finetuned.pt"
        
        if model_path.exists():
            state_dict = torch.load(str(model_path), map_location='cpu')
            model.load_state_dict(state_dict)
            print(f"[+] Loaded fine-tuned model from {model_path}")
        else:
            print(f"[!] Model file not found: {model_path}, using untrained model")
        
        model.eval()
        model.to(device)
        _FINETUNED_MODEL = model
        _FINETUNED_DEVICE = device
        return model, device
    
    except Exception as e:
        print(f"[-] Error loading fine-tuned model: {e}")
        return None, None


def get_osteoporosis_score(image_path_or_array) -> float:
    """Get osteoporosis probability [0, 1] from image
    
    Args:
        image_path_or_array: Path to image file or numpy array (grayscale [0,1])
    
    Returns:
        float: Probability of osteoporosis [0, 1], or None if error
    """
    try:
        # Load image if path provided
        if isinstance(image_path_or_array, (str, Path)):
            gray_norm = np.array(Image.open(str(image_path_or_array)).convert('L'), dtype=np.float32) / 255.0
        else:
            gray_norm = image_path_or_array
        
        # Load model
        model, device = load_finetuned_model()
        if model is None:
            return None
        
        # Preprocess: resize and normalize
        pil_img = Image.fromarray((gray_norm * 255).astype(np.uint8)).resize((224, 224))
        img_array = np.array(pil_img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)
        
        # Normalize to [-1, 1] (critical - training used this)
        img_tensor = (img_tensor - 0.5) / 0.5
        
        # Inference
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            osteo_prob = float(probs[0, 1].item())
        
        return osteo_prob
    
    except Exception as e:
        print(f"✗ Inference error: {e}")
        return None


if __name__ == "__main__":
    # Test
    from pathlib import Path
    test_file = list(Path("dataset/expanded_benchmark/osteoporosis").glob("*"))[0]
    score = get_osteoporosis_score(str(test_file))
    print(f"Test image osteoporosis score: {score:.4f}")
