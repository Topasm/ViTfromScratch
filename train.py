import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from torchvision import transforms, datasets
from model.model import Vit
import wandb

device = "cuda"
path = './cifar_vit.pth'

"https://tutorials.pytorch.kr/beginner/basics/data_tutorial.html"
"https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 256
total_epoch = 200

trainset = datasets.CIFAR10(root='./data', train=True,
                            download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False,
                           download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

wandb.init(project="vit_fromscratch")


def train():

    configs = {
        "patch_size": 4,
        "num_class": 10}

    model = Vit(**configs).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3,
                     betas=(0.9, 0.999), weight_decay=0.1)

    """We train all models, including ResNets, using Adam (Kingma & Ba,
2015) with β1 = 0.9, β2 = 0.999, a batch size of 4096 and apply a high weight decay of 0.1, """

    for epoch in range(total_epoch):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            inputs, lables = data
            inputs, lables = inputs.to(device), lables.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, lables)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            wandb.log({"loss": loss.item(), "epoch": epoch})

        print(
            f"Epoch [{epoch+1}/{total_epoch} - Loss: {running_loss/len(trainloader)}]")

        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"./cifar_vit_epoch{epoch+1}.pth")
        wandb.save(path)


def main():
    train()


if __name__ == "__main__":
    main()
