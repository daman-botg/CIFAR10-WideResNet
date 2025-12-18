import torchvision
import torchvision.transforms as tt
from torch.utils.data import DataLoader

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_transform = tt.Compose([
    tt.RandomCrop(32, padding=4, padding_mode='reflect'),
    tt.RandomHorizontalFlip(),
    tt.ToTensor(),
    tt.Normalize(*stats)
])

valid_transform = tt.Compose([
    tt.ToTensor(),
    tt.Normalize(*stats)
])

def get_cifar10(bs=256):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=valid_transform)
    train_dl = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(valset, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    return trainset, valset, train_dl, val_dl
