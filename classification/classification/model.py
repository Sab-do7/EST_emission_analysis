

import torch
from torchvision.models import resnet50

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = resnet50(pretrained=True)

# modify first and final layer
model.conv1 = torch.nn.Conv2d(12, 64, kernel_size=(3, 3),
                              stride=(2, 2), padding=(3, 3), bias=False)
model.fc = torch.nn.Linear(4*512, 1)

model.to(device)
