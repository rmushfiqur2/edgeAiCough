import torch
from torchvision.models import resnet18

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = resnet18().to(device)
data = torch.rand(1, 3, 64, 64).to(device)
labels = torch.rand(1, 1000).to(device)

prediction = model(data) # forward pass

loss = (prediction - labels).sum()
loss.backward() # backward pass
print(loss)