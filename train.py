import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models

# import lera

from data_loader import AVADataset

from model import *


def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    val_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.ToTensor()])

    base_model = models.vgg16(pretrained=True)
    model = NIMA(base_model)

    if config.warm_start:
        model.load_state_dict(torch.load(os.path.join(config.ckpt_path, 'epoch-%d.pkl' % config.warm_start_epoch)))
        print('Successfully loaded model epoch-%d.pkl' % config.warm_start_epoch)

    if config.multi_gpu:
        model.features = torch.nn.DataParallel(model.features, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)

    conv_base_lr = config.conv_base_lr
    dense_lr = config.dense_lr
    optimizer = optim.SGD([
        {'params': model.features.parameters(), 'lr': conv_base_lr},
        {'params': model.classifier.parameters(), 'lr': dense_lr}],
        momentum=0.9
        )

    # send hyperparams
    '''
    lrs.send({
        'title': 'EMD Loss',
        'train_batch_size': config.train_batch_size,
        'val_batch_size': config.val_batch_size,
        'optimizer': 'SGD',
        'conv_base_lr': config.conv_base_lr,
        'dense_lr': config.dense_lr,
        'momentum': 0.9
        })
    '''

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    if config.train:
        trainset = AVADataset(csv_file=config.train_csv_file, root_dir=config.train_img_path, transform=train_transform)
        valset = AVADataset(csv_file=config.val_csv_file, root_dir=config.val_img_path, transform=val_transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
            shuffle=True, num_workers=config.num_workers)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=config.val_batch_size,
            shuffle=False, num_workers=config.num_workers)
        # for early stopping
        count = 0
        init_val_loss = float('inf')
        train_losses = []
        val_losses = []
        for epoch in range(config.warm_start_epoch, config.epochs):
            # lrs.send('epoch', epoch)
            batch_losses = []
            for i, data in enumerate(train_loader):
                images = data['image'].to(device)
                labels = data['annotations'].to(device).float()
                outputs = model(images)
                outputs = outputs.view(-1, 10, 1)

                optimizer.zero_grad()

                loss = emd_loss(labels, outputs)
                batch_losses.append(loss.item())

                loss.backward()

                optimizer.step()

                # lrs.send('train_emd_loss', loss.item())

                print('Epoch: %d/%d | Step: %d/%d | Training EMD loss: %.4f' % (epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size + 1, loss.data[0]))

            avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size + 1)
            train_losses.append(avg_loss)
            print('Epoch %d averaged training EMD loss: %.4f' % (epoch + 1, avg_loss))

            # exponetial learning rate decay
            if (epoch + 1) % 10 == 0:
                conv_base_lr = conv_base_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
                dense_lr = dense_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
                optimizer = optim.SGD([
                    {'params': model.features.parameters(), 'lr': conv_base_lr},
                    {'params': model.classifier.parameters(), 'lr': dense_lr}],
                    momentum=0.9
                )

                # send decay hyperparams
                # lrs.send({
                #    'lr_decay_rate': config.lr_decay_rate,
                #    'lr_decay_freq': config.lr_decay_freq,
                #    'conv_base_lr': config.conv_base_lr,
                #    'dense_lr': config.dense_lr
                #    })

            # do validation after each epoch
            batch_val_losses = []
            for data in val_loader:
                images = data['image'].to(device)
                labels = data['annotations'].to(device).float()
                with torch.no_grad():
                    outputs = model(images)
                outputs = outputs.view(-1, 10, 1)
                val_loss = emd_loss(labels, outputs)
                batch_val_losses.append(val_loss.item())
            avg_val_loss = sum(batch_val_losses) / (len(valset) // config.val_batch_size + 1)
            val_losses.append(avg_val_loss)

            # lrs.send('val_emd_loss', avg_val_loss)

            print('Epoch %d completed. Averaged EMD loss on val set: %.4f.' % (epoch + 1, avg_val_loss))

            # Use early stopping to monitor training
            if avg_val_loss < init_val_loss:
                init_val_loss = avg_val_loss
                # save model weights if val loss decreases
                print('Saving model...')
                if not os.path.exists(config.ckpt_path):
                    os.makedirs(config.ckpt_path)
                torch.save(model.state_dict(), os.path.join(config.ckpt_path, 'epoch-%d.pkl' % (epoch + 1)))
                print('Done.\n')
                # reset count
                count = 0
            elif avg_val_loss >= init_val_loss:
                count += 1
                if count == config.early_stopping_patience:
                    print('Val EMD loss has not decreased in %d epochs. Training terminated.' % config.early_stopping_patience)
                    break

        print('Training completed.')

        if config.save_fig:
            # plot train and val loss
            epochs = range(1, epoch + 2)
            plt.plot(epochs, train_losses, 'b-', label='train loss')
            plt.plot(epochs, val_losses, 'g-', label='val loss')
            plt.title('EMD loss')
            plt.legend()
            plt.savefig('./loss.png')

    if config.test:
        model.eval()
        # compute mean score
        test_transform = val_transform
        testset = AVADataset(csv_file=config.test_csv_file, root_dir=config.test_img_path, transform=val_transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size, shuffle=False, num_workers=config.num_workers)

        mean_preds = []
        std_preds = []
        for data in test_loader:
            image = data['image'].to(device)
            output = model(image)
            output = output.view(10, 1)
            predicted_mean, predicted_std = 0.0, 0.0
            for i, elem in enumerate(output, 1):
                predicted_mean += i * elem
            for j, elem in enumerate(output, 1):
                predicted_std += elem * (j - predicted_mean) ** 2
            mean_preds.append(predicted_mean)
            std_preds.append(predicted_std)
        # Do what you want with predicted and std...


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--train_img_path', type=str, default='/home/yunxiao/ava_data/train')
    parser.add_argument('--val_img_path', type=str, default='/home/yunxiao/ava_data/val')
    parser.add_argument('--test_img_path', type=str, default='/home/yunxiao/ava_data/test')
    parser.add_argument('--train_csv_file', type=str, default='../train_labels.csv')
    parser.add_argument('--val_csv_file', type=str, default='../val_labels.csv')
    parser.add_argument('--test_csv_file', type=str, default='../test_labels.csv')

    # training parameters
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--conv_base_lr', type=float, default=3e-7)
    parser.add_argument('--dense_lr', type=float, default=3e-6)
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_decay_freq', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='../ckpts')
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--warm_start', type=bool, default=False)
    parser.add_argument('--warm_start_epoch', type=int, default=0)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--save_fig', type=bool, default=False)

    config = parser.parse_args()

    main(config)
