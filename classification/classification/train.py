
import numpy as np
import torch
from torch import nn, optim
from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
import argparse

from model import *
from data import create_dataset

def train_model(model, epochs, opt, loss, batch_size):
    """Wrapper function for model training.

    :param model: model instance
    :param epochs: (int) number of epochs to be trained
    :param opt: optimizer instance
    :param loss: loss function instance
    :param batch_size: (int) batch size"""

    # create dataset
    data_train = create_dataset(
        datadir = '../../images/images_/train',
        balance='upsample', mult=1)

    data_val = create_dataset(
        datadir='../../images/images_/val',
        # path to val data
        balance='upsample', mult=1)

    # draw random subsamples
    train_sampler = RandomSampler(data_train, replacement=True,
                                  num_samples=int(2*len(data_train)/3))
    val_sampler = RandomSampler(data_val, replacement=True,
                                  num_samples=int(2*len(data_val)/3))

    # initialize data loaders
    train_dl = DataLoader(data_train, batch_size=batch_size, num_workers=4,
                          pin_memory=True, sampler=train_sampler)
    val_dl = DataLoader(data_val, batch_size=batch_size, num_workers=4,
                         pin_memory=True, sampler=val_sampler)

    # start training process
    for epoch in range(epochs):

        model.train()

        train_loss_total, train_acc_total = 0, 0
        progress = tqdm(enumerate(train_dl), desc="Train Loss: ",
                        total=len(train_dl))
        for i, batch in progress:
            x = batch['img'].float().to(device)
            y = batch['lbl'].float().to(device)

            output = model(x)

            # derive binary output
            output_binary = np.zeros(output.shape)
            output_binary[output.cpu().detach().numpy() >= 0] = 1

            # derive accuracy score
            acc = accuracy_score(y.cpu().detach().numpy(), output_binary)
            train_acc_total += acc

            # calculate loss
            loss_epoch = loss(output, y.reshape(-1, 1))
            train_loss_total += loss_epoch.item()
            progress.set_description("Train Loss: {:.4f}".format(
                train_loss_total/(i+1)))

            # learning
            opt.zero_grad()
            loss_epoch.backward()
            opt.step()

        # logging
        writer.add_scalar("training loss", train_loss_total/(i+1), epoch)
        writer.add_scalar("training acc", train_acc_total/(i+1), epoch)
        writer.add_scalar('learning_rate', opt.param_groups[0]['lr'], epoch)

        torch.cuda.empty_cache()

        # evaluation based on validation sample
        model.eval()
        val_loss_total, val_acc_total = 0, 0
        progress = tqdm(enumerate(val_dl), desc="val Loss: ",
                        total=len(val_dl))
        for j, batch in progress:
            x, y = batch['img'].float().to(device), batch['lbl'].float().to(device)

            output = model(x)

            # calculate loss
            loss_epoch = loss(output, y.reshape(-1, 1))
            val_loss_total += loss_epoch.item()
            progress.set_description("val Loss: {:.4f}".format(
                val_loss_total/(j+1)))

            # derive binary output
            output_binary = np.zeros(output.shape)
            output_binary[output.cpu().detach().numpy() >= 0] = 1

            # derive accuracy score
            acc = accuracy_score(y.cpu().detach().numpy(), output_binary)
            val_acc_total += acc

        # logging
        writer.add_scalar("val loss", val_loss_total/(j+1), epoch)
        writer.add_scalar("val accuracy", val_acc_total/(j+1), epoch)

        # screen output
        print(("Epoch {:d}: train loss={:.3f}, val loss={:.3f}, "
               "train acc={:.3f}, val acc={:.3f}").format(
                   epoch+1, train_loss_total/(i+1), val_loss_total/(j+1),
                   train_acc_total/(i+1), val_acc_total/(j+1)))
      
        # save model checkpoint
        if epoch % 1 == 0:
            torch.save(model.state_dict(),
            'ep{:0d}_lr{:.0e}_bs{:02d}_mo{:.1f}_{:03d}.model'.format(
                args.ep, args.lr, args.bs, args.mo, epoch))

        writer.flush()
        scheduler.step(epoch)
        torch.cuda.empty_cache()

    return model


# setup argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-ep', type=int, default=100,
                    help='Number of epochs')
parser.add_argument('-bs', type=int, nargs='?',
                    default=30, help='Batch size')
parser.add_argument('-lr', type=float,
                    nargs='?', default=0.3, help='Learning rate')
parser.add_argument('-mo', type=float,
                    nargs='?', default=0.7, help='Momentum')
args = parser.parse_args()

# initialize tensorboard writer
writer = SummaryWriter('runs/'+"ep{:0d}_lr{:.0e}_bs{:03d}_mo{:.1f}/".format(
    args.ep, args.lr, args.bs, args.mo))

# initialize loss, optimizer, and scheduler
loss = nn.BCEWithLogitsLoss()
opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mo)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min',
                                                 factor=0.5, threshold=1e-4,
                                                 min_lr=1e-6)

# run model training
train_model(model, args.ep, opt, loss, args.bs)

writer.close()

