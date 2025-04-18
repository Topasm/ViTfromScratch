import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from torchvision import transforms, datasets
from model.model import Vit
import wandb

device = "cuda"
save_dir = './checkpoints'
save_path = './cifar_vit.pth'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 256
total_epoch = 200
warmup_epochs = 10

trainset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False,
                           download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2)

wandb.init(project="vit_fromscratch")


def train():
    configs = {
        "patch_size": 4,
        "num_class": 10
    }

    model = Vit(**configs).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-4,
                      betas=(0.9, 0.999), weight_decay=0.05)

    # Warmup + Cosine Annealing Scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epoch - warmup_epochs)

    # Warmup manual setting
    def warmup_lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        return 1.0

    warmup_scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=warmup_lr_lambda)

    for epoch in range(total_epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(trainloader)
        train_acc = 100. * correct / total

        wandb.log({
            "loss": train_loss,
            "train_accuracy": train_acc,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"]
        })

        print(
            f"Epoch [{epoch+1}/{total_epoch}] - Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}%")

        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_path = f"{save_dir}/cifar_vit_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            wandb.save(ckpt_path)

        # Scheduler step
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()


def main():
    train()


if __name__ == "__main__":
    main()
