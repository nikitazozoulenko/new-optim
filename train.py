import torch
import numpy as np
from new_optim import SGD, OlegOptim
from models import ResNet32x32, SimpleCNN
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets


def get_cifar10_data(data_dir, seed=13):
    #90% data is train, 10% is validation. USE RANDOM SEED
    cifar10 = datasets.CIFAR10(data_dir, train=True, transform=None, target_transform=None, download=True)
    train_size = int(len(cifar10)*0.9)
    np.random.seed(seed)
    indices = np.arange(len(cifar10))
    train_data = [[np.array(cifar10[idx][0]), cifar10[idx][1]] for idx in indices[0:train_size]]
    val_data = [[np.array(cifar10[idx][0]), cifar10[idx][1]] for idx in indices[train_size:]]
    return train_data, val_data


def get_batch(data, batch_size, device):
    indices = np.random.randint(0, len(data), size=batch_size)
    images = np.array([data[idx][0].transpose(2,0,1) for idx in indices])
    labels = np.array([data[idx][1] for idx in indices])
    images = torch.tensor(images, dtype=torch.float, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    return images, labels


def train_iteration(model, optim, loss_fn, writer, train_data, batch_size, global_step, device):
    images, labels = get_batch(train_data, batch_size, device)
    pred = model(images).squeeze()
    loss = loss_fn(pred, labels)
    loss.backward()
    optim.step()
    optim.zero_grad()

    #summary
    writer.add_scalar('loss/train', loss, global_step)
    #only OlegOptim
    writer.add_scalar('delta', optim.delta, global_step)
    writer.add_scalar('grad', optim.delta, global_step)



def val_iteration(model, loss_fn, writer, val_data, val_batch_size, global_step, device):
    with torch.no_grad():
        print(global_step)
        images, labels = get_batch(val_data, val_batch_size, device)
        pred = model(images).squeeze()
        loss = loss_fn(pred, labels)
        max_indices = torch.argmax(pred, dim=1)
        acc = torch.mean((max_indices==labels).float())

        #summary
        writer.add_scalar('loss/val', loss, global_step)
        writer.add_scalar('acc', acc, global_step)


def decrease_lr(optimizer, factor=10):
    for param_group in optimizer.param_groups:
        print("updated learning rate: new lr:", param_group['lr']/factor)
        param_group['lr'] = param_group['lr']/factor


def train():
    #parameters
    device = torch.device("cuda") #"cuda" or "cpu"
    n_iterations = 50
    when_decrease_lr = [10000, 15000]
    learning_rate=0.01
    momentum=0
    log_dir = "logs/cifar10-OlegOptim/"
    data_dir = "/hdd/Data/"
    batch_size=100
    val_batch_size=1000
    val_iter = 25

    #model, optimizer, loss_fn, data, and writer
    model = ResNet32x32().to(device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean").to(device)
    optim = OlegOptim(model.parameters(), lr=learning_rate, momentum=momentum)
    writer = SummaryWriter(log_dir=log_dir) # RUN THIS: tensorboard --logdir=/home/nikita/Code/new-optim/logs/ --host=127.0.0.1
    train_data, val_data = get_cifar10_data(data_dir)

    for global_step in range(n_iterations):
        train_iteration(model, optim, loss_fn, writer, train_data, batch_size, global_step, device)
        if global_step%val_iter==0:
            val_iteration(model, loss_fn, writer, val_data, val_batch_size, global_step, device)
        if global_step in when_decrease_lr:
            decrease_lr(optim)


if __name__=="__main__":
    train()