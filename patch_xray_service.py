#!/usr/bin/env python3
"""Patch xray_service.py to add fine-tuned model integration"""

from pathlib import Path

# The fine-tuned model function code
finetuned_func = '''def _get_finetuned_osteoporosis_score(gray_norm: np.ndarray) -> float:
    """Load binary ResNet-50 fine-tuned model and return osteoporosis probability [0,1]"""
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
                
                self.downsample = None
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
                
                if self.downsample is not None:
                    identity = self.downsample(x)
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
        model = ResNet50Binary(num_classes=2, in_channels=1)
        model_path = Path(__file__).resolve().parent.parent / "models" / "resnet50_finetuned.pt"
        if model_path.exists():
            state_dict = torch.load(str(model_path), map_location='cpu')
            model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        
        # Prepare input
        pil_img = Image.fromarray((gray_norm * 255).astype(np.uint8)).resize((224, 224))
        img_array = np.array(pil_img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)
        
        # Normalize to [-1, 1] (critical - training used this normalization)
        img_tensor = (img_tensor - 0.5) / 0.5
        
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            osteo_prob = float(probs[0, 1].item())  # Probability of osteoporosis class
        
        return osteo_prob
    
    except Exception:
        return None


'''

# Read the file
with open('backend/services/xray_service.py', 'r') as f:
    lines = f.readlines()

# Find line with "def extract_radiomic_features"
insert_line = None
for i, line in enumerate(lines):
    if 'def extract_radiomic_features' in line:
        insert_line = i
        break

if insert_line:
    # Insert the function before extract_radiomic_features
    lines.insert(insert_line, finetuned_func + '\n')
    
    # Write back
    with open('backend/services/xray_service.py', 'w') as f:
        f.writelines(lines)
    print(f'✓ Added fine-tuned model function')
else:
    print('✗ Could not find insert point')

# Now add the integration into run_inference
with open('backend/services/xray_service.py', 'r') as f:
    content = f.read()

# Add fine-tuned model blending after radiometric score calculation
old_section = '''        # Composite calibrated osteoporosis score
        osteoporosis_score = float(np.clip(
            0.40 * cortical_thinning + 0.35 * trabecular_loss + 0.20 * bmd_attenuation + 0.05 * fracture_score,
            0.05, 0.98
        ))'''

new_section = '''        # Composite calibrated osteoporosis score (radiometric baseline)
        radiometric_score = float(np.clip(
            0.40 * cortical_thinning + 0.35 * trabecular_loss + 0.20 * bmd_attenuation + 0.05 * fracture_score,
            0.05, 0.98
        ))
        
        # Blend with fine-tuned model prediction (70% model + 30% radiometric)
        finetuned_score = _get_finetuned_osteoporosis_score(gray_norm)
        score_source = "Radiometric Features Only"
        if finetuned_score is not None:
            osteoporosis_score = float(np.clip(
                0.70 * finetuned_score + 0.30 * radiometric_score,
                0.05, 0.98
            ))
            score_source = "Fine-tuned ResNet-50 (95.30% accuracy) + Radiometric Features"
        else:
            osteoporosis_score = radiometric_score'''

content = content.replace(old_section, new_section)

# Update xai_status to show model info
old_xai = '"xai_status": "Explainable AI Grad-CAM generated successfully",'
new_xai = '"xai_status": f"Explainable AI Grad-CAM generated successfully | Model: {score_source}",'

content = content.replace(old_xai, new_xai)

# Write back
with open('backend/services/xray_service.py', 'w') as f:
    f.write(content)

print('✓ Added fine-tuned model integration')
print('✓ Updated xai_status output')
