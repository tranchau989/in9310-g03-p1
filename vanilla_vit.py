import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import STL10
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm

source_folder = 'source'

# Set seed
seed = 1

# Get STL10 data
data_dir = '/fp/projects01/ec517/data' # Update this to fit your setup if not on Educloud

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
])

# Get STL10 data
train_set = STL10(root=data_dir, split='train', download=True, transform=transform)
test_set = STL10(root=data_dir, split='test', download=True, transform=transform)

# Create data loaders
batch_size = 128
train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, tensor):
        # tensor: C x H x W
        return tensor * self.std[:, None, None] + self.mean[:, None, None]

# Useful for showing images
untransform = transforms.Compose([
    UnNormalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    transforms.ToPILImage()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The number of classes in the dataset we want to finetune on
num_classes = 10

# Load ViT model with pretrained weighs
model_lora = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes).to(device)



# Train ViT with LoRA wrapper

optimizer = SGD(model_lora.parameters(), lr=lr)
loss_criterion = nn.CrossEntropyLoss()

# Train for a few epochs
model_lora.train()
for epoch in range(epochs):
    train_loss, total_samples = 0.0, 0.0

    train_bar = tqdm(iterable=train_loader)
    for i, batch in enumerate(train_bar):
        x, y = batch
        x, y = x.to(device), y.to(device) #F.one_hot(y.to(device), num_classes)
        y_hat = model_lora(x)
        loss = loss_criterion(y_hat, y)

        train_loss += loss.item()
        total_samples += len(x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_bar.set_description(f'Epoch [{epoch+1}/{epochs}]')
        train_bar.set_postfix(loss = train_loss / total_samples)

# Test model on test data
model_lora.eval()
with torch.no_grad():
    test_loss, correct_pred, total_samples = 0.0, 0.0, 0.0

    test_bar = tqdm(iterable=test_loader)
    for i, batch in enumerate(test_bar):
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_hat = model_lora(x)
        loss = loss_criterion(y_hat, y)

        test_loss += loss.detach().cpu().item()
        correct_pred += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
        total_samples += len(x)

        test_bar.set_description(f'Test loop')
        test_bar.set_postfix(loss = test_loss / total_samples)