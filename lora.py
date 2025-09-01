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

class LoRAWrapper(nn.Module):
    
    def __init__(self, linear, rank):
        super().__init__()
        assert isinstance(linear, nn.Linear)

        # Save original weight and bias
        self.register_buffer('orig_weight', linear.weight.data.detach().clone())
        if linear.bias is not None:
            self.register_buffer('orig_bias', linear.bias.data.detach().clone())
        else:
            self.register_buffer('orig_bias', None)

        # Save the parameters you might need...
        self.in_dim = linear.in_features
        self.out_dim = linear.out_features
        self.bias = linear.bias is not None
        self.rank = min(rank, self.in_dim, self.out_dim)

        # Initialize the A and B weights. You can do this with nn.Linear()
        device = linear.weight.device # Place A and B on same device as W0
        self.A = nn.Linear(self.rank, self.out_dim, bias=False, device=device) # No bias in original paper (see Section 4.2)
        self.B = nn.Linear(self.in_dim, self.rank, bias=False, device=device) # No bias in original paper (see Section 4.2)

        # Make sure the A weights are initialized with Gaussian random noise
        nn.init.normal_(self.A.weight)

        # Make sure the B weights are initialized with zeros
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        W0x = F.linear(x, self.orig_weight, self.orig_bias)
        deltaWx = self.A(self.B(x))

        return W0x + deltaWx

# Freeze all layers except classification head
for name, param in model_lora.named_parameters():
    if name in ['norm.weight', 'norm.bias', 'head.weight', 'head.bias']:
        param.requires_grad = True
    else:
        param.requires_grad = False


# LoRA wrapper
for block in model_lora.blocks:

    # Wrap linear layers in the attention block
    block.attn.qkv = LoRAWrapper(block.attn.qkv, rank=4) # Rank = 4 matches paper
    block.attn.proj = LoRAWrapper(block.attn.proj, rank=4) # Rank = 4 matches paper

    # Unfreeze the attention block
    block.attn.requires_grad_(True)

    # Unfreeze LayerScale layers as well
    block.ls1.requires_grad_(True)
    block.ls2.requires_grad_(True)

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