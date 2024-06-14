import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SpatialPath(nn.Module):
    def __init__(self):
        super(SpatialPath, self).__init__()
        # Simplified SpatialPath architecture
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

class ContextPath(nn.Module):
    def __init__(self):
        super(ContextPath, self).__init__()
        # Use ResNet-101 as the backbone
        resnet101 = models.resnet101(pretrained=True)
        
        # Remove the fully connected layers at the end
        self.resnet_features = nn.Sequential(*list(resnet101.children())[:-2])

    def forward(self, x):
        # Forward pass through the ResNet-101 backbone
        x = self.resnet_features(x)
        return x

class BiSeNet(nn.Module):
    def __init__(self, num_classes):
        super(BiSeNet, self).__init__()
        self.spatial_path = SpatialPath()
        self.context_path = ContextPath()
        
        # Joint convolutional layers
        self.global_context = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.arms = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(768, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0)
            ),
            nn.Sequential(
                nn.Conv2d(2048, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0)
            )
        ])

    def forward(self, x):
        spatial_output = self.spatial_path(x)
        context_output = self.context_path(x)
        global_context = self.global_context(context_output)

        # Upsample and concatenate
        global_context_upsampled = F.interpolate(global_context, size=spatial_output.size()[2:], mode='bilinear', align_corners=True)
        spatial_context_concat = torch.cat([spatial_output, global_context_upsampled], 1)

        # Branches
        arm1 = self.arms[0](spatial_context_concat)
        arm2 = self.arms[1](context_output)
        # final_prediction = (arm1 + arm2) / 2

        return arm1, arm2

# Instantiate the model
# num_classes = 21  # Number of classes in the dataset
# bisenet_model = BiSeNet(num_classes)

# # Print the model architecture
# print(bisenet_model)

# if __name__ == "__main__":
#     model = BiSeNet(20)
#     print(model)
