import argparse
import numpy as np
import torch
import torch.nn
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from model import mymodel

train_transform = transforms.Compose([transforms.ToTensor()])
test_transform = train_transform
test_set = MNIST('./data/mnist', train=False, download=True,
                 transform=test_transform)

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--weight_file", type=str)
args = parser.parse_args()

try:
    index = int(input("Enter an integer index between 0 and 59999: "))
except ValueError:
    print("Invalid input. Please enter a valid integer.")
    exit(1)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=2048, shuffle=False)


def test(model, test_img, device='cpu'):
    for imgs, labels in test_loader:
        model.eval()
        with torch.no_grad():
            model.load_state_dict(torch.load('MLP.8.pth'))
            imgs = imgs.view(imgs.size(0), -1)
            # Change type to float32
            imgs = imgs.to(dtype=torch.float32)
            outputs = model(imgs)
            # Images side by side
            f = plt.figure()
            f.add_subplot(1, 3, 1)
            plt.imshow(test_img.data[index].squeeze(), cmap='gray')

            noise = torch.rand(imgs.size())*0.5
            noisy_image = imgs + noise
            noisy_image = torch.clamp(noisy_image, 0, 1)
            f.add_subplot(1, 3, 2)
            plt.imshow(noisy_image[index].view(28,28).numpy(), cmap='gray')

            f.add_subplot(1, 3, 3)
            plt.imshow(outputs[index].view(28,28).numpy(), cmap='gray')
            plt.show()
        return


def interpolate(model, img1, img2, n_steps):
    model.eval()
    interpolated_tensors = []
    img1 = img1.to(dtype=torch.float32)
    img2 = img2.to(dtype=torch.float32)
    fig = plt.figure()
    plt.subplot(1, 10, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('Original1')

    plt.subplot(1, 10, 10)
    plt.imshow(img2, cmap='gray')
    plt.title('Original2')

    img1 = torch.reshape(img1, (1, 784))
    img2 = torch.reshape(img2, (1, 784))

    interp1 = model.encode(img1)
    interp2 = model.encode(img2)
    for i in range(1, n_steps + 1):
        alpha = (i-1) / (n_steps + 1)
        interpolated_img = model.decode((alpha * interp2) + (1 - alpha) * interp1)
        #interpolated_tensors.append(interpolated_img)
        image_array = interpolated_img.detach().numpy()
        fig.add_subplot(1, 10, i+1)
        plt.imshow(image_array.reshape(28, 28), cmap='gray')
        #plt.imshow(image_array.view(28, 28), cmap='gray')
    plt.show()
    exit()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = mymodel()
    model.load_state_dict(torch.load('MLP.8.pth'))
    test(model=model, test_img=test_set, device=device)
    n_steps = 8
    img1 = test_set.data[index]
    img2 = test_set.data[index+1]

    interpolate(model, img1, img2, n_steps)


if __name__ == '__main__':
    main()
