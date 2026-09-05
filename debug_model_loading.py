#!/usr/bin/env python3
"""Debug fine-tuned model loading"""

import numpy as np
from PIL import Image
from pathlib import Path
from backend.services.xray_service import _get_finetuned_osteoporosis_score

# Load test image
test_osteo = list(Path("dataset/expanded_benchmark/osteoporosis").glob("*"))[0]
img = Image.open(str(test_osteo)).convert("L")
img = img.resize((224, 224))
gray_norm = np.array(img, dtype=np.float32) / 255.0

# Test with debug
print("Testing fine-tuned model...")
print(f"Input shape: {gray_norm.shape}")
print(f"Input range: [{gray_norm.min():.4f}, {gray_norm.max():.4f}]")

try:
    import torch
    import torch.nn as nn
    
    device = torch.device('cpu')
    
    # Define ResNetBlock (bottleneck with expansion=4)
    class ResNetBlock(nn.Module):
        expansion = 4
        
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            mid_channels = out_channels // self.expansion
            self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(mid_channels)
            self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(mid_channels)
            self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
        
        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv3(out)
            out = self.bn3(out)
            
            identity = self.shortcut(x)
            out += identity
            out = self.relu(out)
            return out
    
    # Define ResNet50Binary
    class ResNet50Binary(nn.Module):
        def __init__(self, num_classes=2, in_channels=1):
            super().__init__()
            self.in_channels = in_channels
            self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            
            self.layer1 = self._make_layer(64, 64, 3, stride=1)
            self.layer2 = self._make_layer(256, 128, 4, stride=2)
            self.layer3 = self._make_layer(512, 256, 6, stride=2)
            self.layer4 = self._make_layer(1024, 512, 3, stride=2)
            
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(2048, num_classes)
        
        def _make_layer(self, in_ch, out_ch, blocks, stride):
            layers = []
            layers.append(ResNetBlock(in_ch, out_ch * 4, stride=stride))
            for _ in range(1, blocks):
                layers.append(ResNetBlock(out_ch * 4, out_ch * 4, stride=1))
            return nn.Sequential(*layers)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # Load model
    print("\n1. Creating model...")
    model = ResNet50Binary(num_classes=2, in_channels=1)
    print("   ✓ Model created")
    
    print("\n2. Loading weights...")
    model_path = Path("backend/models/resnet50_finetuned.pt")
    if model_path.exists():
        state_dict = torch.load(str(model_path), map_location='cpu')
        print(f"   ✓ Weights loaded ({len(state_dict)} keys)")
        model.load_state_dict(state_dict)
        print("   ✓ Weights loaded into model")
    else:
        print(f"   ✗ Model file not found: {model_path}")
    
    print("\n3. Preparing input...")
    model.eval()
    model.to(device)
    
    # Prepare input
    pil_img = Image.fromarray((gray_norm * 255).astype(np.uint8)).resize((224, 224))
    img_array = np.array(pil_img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)
    print(f"   ✓ Tensor shape: {img_tensor.shape}")
    
    # Normalize to [-1, 1] (critical - training used this normalization)
    img_tensor = (img_tensor - 0.5) / 0.5
    print(f"   ✓ Normalized to [-1, 1]")
    print(f"     Range: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")
    
    print("\n4. Running inference...")
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        osteo_prob = float(probs[0, 1].item())  # Probability of osteoporosis class
    
    print(f"   ✓ Inference complete")
    print(f"   Prediction: {osteo_prob:.4f}")

except Exception as e:
    print(f"   ✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
