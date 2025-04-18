import torch
import torchvision
import torchvision.transforms as transforms
from model.model import Vit  # make sure this matches your file structure
import os

"https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html?highlight=cifar based on this code"

classes = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False,
    download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64,
    shuffle=False, num_workers=2
)

model = Vit().to(device)

checkpoint_path = 'checkpoints/cifar_vit_epoch10.pth'
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()


correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}


with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for label, prediction in zip(labels, predicted):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# 각 분류별 정확도(accuracy)를 출력합니다
with open('accuracy_log.txt', 'w') as log_file:
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] > 0:
            accuracy = 100 * float(correct_count) / total_pred[classname]
            log_file.write(
                f'Accuracy for class: {classname:5s} is {accuracy:.1f} %\n')
        else:
            log_file.write(f'No predictions for class: {classname}\n')
