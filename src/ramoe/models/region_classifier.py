# region_classifier/RegionClassifier.py
import torch
import torch.nn as nn

class RegionClassifier(nn.Module):
    def __init__(self, input_size=48, num_regions=5):
        super(RegionClassifier, self).__init__()
        self.input_size = input_size
        
        # ✅ 简化模型：减少通道数（32→16, 64→32, 128→64）
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  # 48→24
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 24→12
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 12→6
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_size, input_size)
            dummy_output = self.features(dummy_input)
            self.fc_in = dummy_output.numel() // dummy_output.size(0)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_in, 64),  # 128→64
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_regions)  # 输出层保持不变
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x