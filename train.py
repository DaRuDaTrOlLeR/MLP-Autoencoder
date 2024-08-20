import datetime
import argparse
import torch.optim as optim
import torch
import torch.nn
from torch import nn
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import torchsummary
from model import mymodel

train_transform = transforms.Compose([transforms.ToTensor()])
test_transform = train_transform
train_set = MNIST('./data/mnist', train=True, download=True,
                  transform=train_transform)
test_set = MNIST('./data/mnist', train=False, download=True,
                 transform=test_transform)

parser = argparse.ArgumentParser()
parser.add_argument("-z", "--bottleneck", type=int)
parser.add_argument("-e", "--epoch", type=int)
parser.add_argument("-b", "--batch_size", type=int)
parser.add_argument("-s", "--weight_file", type=str)
parser.add_argument("-p", "--loss_plot", type=str)
args = parser.parse_args()


def train(n_epochs, optimizer, model, loss_fn, scheduler, training_set, save_file=None, plotfile=None, device='cpu'):
    print('training...')
    torchsummary.summary(model, (1, 28 * 28))
    losses_train = []

    for epoch in range(1, n_epochs + 1):
        print('epochs ', epoch)
        loss_train = 0.0
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)

        for imgs, labels in train_loader:

            model.train()
            imgs = imgs.to(device=device)
            imgs = imgs.view(imgs.size(0), -1)
            outputs = model(imgs)
            loss = loss_fn(outputs, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        losses_train += [loss_train / len(train_loader)]
        print('Loss: ', losses_train[epoch-1])

    plt.figure(2, figsize=(12, 7))
    plt.clf()
    plt.plot(losses_train, label='train')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc=1)
    plt.show()
    plt.savefig(args.loss_plot)

    # plt.show(loss_train)
    return losses_train


def main():
    device = torch.device('cpu')
    model_t = mymodel(n_bottleneck=args.bottleneck)   # For training
    plt.savefig(args.loss_plot)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(params=model_t.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    train(n_epochs=args.epoch, optimizer=optimizer, model=model_t, loss_fn=loss_fn,
          training_set=train_set,
          scheduler=scheduler, device=device)

    torch.save(model_t.state_dict(), args.weight_file)


if __name__ == '__main__':
    main()
