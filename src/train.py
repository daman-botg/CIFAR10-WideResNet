from fastai.vision.all import *
import torch
import torch.nn.functional as F
from src.model import wrn22
from src.data import get_cifar10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
trainset, valset, _, _ = get_cifar10(bs=256)

# Wrap datasets for export
train_ds = Datasets(trainset, tfms=[lambda x,y: (x,y)])
valid_ds = Datasets(valset, tfms=[lambda x,y: (x,y)])
dls = DataLoaders(train_ds, valid_ds, bs=256, device=device)

# Load model
model = wrn22().to(device)

# Create learner
learn = Learner(dls, model, loss_func=F.cross_entropy, metrics=accuracy)
learn.clip = 0.1

# Find learning rate (optional)
learn.lr_find()

# Train
learn.fit_one_cycle(10, 5e-3, wd=1e-4)

# Save checkpoint and export for inference
learn.save('outputs/wrn_cifar10_checkpoint')
learn.export('outputs/wrn_cifar10.pkl')
