from typing import Callable

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision.models import resnet18
from tqdm import tqdm


def load_data(dataset_path: str, split: str = 'val', transform: Callable = None, batch_size: int = 1,
              num_workers: int = 1) -> DataLoader:
    dataset = ImageNet(root=dataset_path, split=split, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return dataloader


def train_one_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                    scheduler: LRScheduler, criterion: nn.Module) -> float:
    model.train()

    epoch_train_loss = 0.0
    batch_counter = 0

    pbar = tqdm(train_loader)

    for batch in pbar:
        images, labels = batch

        if torch.cuda.is_available():
            images = images.to(device)
            labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        train_loss = criterion(outputs, labels)

        train_loss.backward()

        optimizer.step()

        last_lr = scheduler.get_last_lr()[0]

        pbar.set_description(f'Train loss: {train_loss.item():.6f} | LR: {last_lr:.6f}')

        epoch_train_loss += train_loss.item()
        batch_counter += 1

    scheduler.step()

    epoch_train_loss /= batch_counter

    return epoch_train_loss


def train(model: nn.Module, train_loader: DataLoader, num_epochs: int = 100) -> None:
    optmizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9, nesterov=False)
    scheduler = StepLR(optimizer=optmizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(disable=True)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model=model, train_loader=train_loader, optimizer=optmizer, scheduler=scheduler,
                                     criterion=criterion)
        pbar.write(f"Epoch {epoch + 1}/{num_epochs} - Training loss: {train_loss:.6}")
        pass

    pass


def main() -> None:
    dataset_path = '../datasets/classification/ILSVRC2012'

    # Аугментации
    train_transform = T.Compose([
        T.Resize(size=(224, 224)),
        T.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.15, hue=0.05),
        T.RandomHorizontalFlip(p=0.45),
        T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    train_loader = load_data(dataset_path=dataset_path, transform=train_transform, batch_size=16, num_workers=10)

    model = resnet18(weights=None)
    model.to(device)

    train(model=model, train_loader=train_loader, num_epochs=90)
    pass


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
