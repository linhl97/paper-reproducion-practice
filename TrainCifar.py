import os
import argparse
import time
from utils.dataset import *
from utils.utils import get_network, WarmUpLR, most_recent_folder, best_acc_weights, DATE_FORMAT, TIME_NOW,\
    most_recent_weights, last_epoch, print_options
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import logging
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

def train(epoch):

    start = time.time()
    model.train()
    for batch_idx, (images, labels) in enumerate(trainloader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(trainloader) + batch_idx + 1

        print("Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}".format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_idx * args.batch_size + len(images),
            total_samples=len(trainloader.dataset)
        ))

        logging.info(
            "Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}".format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_idx * args.batch_size + len(images),
                total_samples=len(trainloader.dataset)
            )
        )

        writer.add_scalar('Loss', loss.item(), n_iter)
    end = time.time()

    print("Epoch {} training time consumed: {:.2f} ".format(epoch, end-start))

    for name, param in model.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

@torch.no_grad()
def eval(epoch, tb=True):

    start = time.time()
    model.eval()

    test_loss = 0.0
    correct = 0

    print('Evaluating modelwork.....')

    for images, labels in testloader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        output = model(images)
        loss = criterion(output, labels)
        test_loss += loss.item()
        _, preds = torch.max(output, 1)
        correct += preds.eq(labels).sum()

    end = time.time()
    accuracy = correct.float() / len(testloader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(testloader),
        accuracy,
        end - start
    ))

    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(testloader), epoch)
        writer.add_scalar('Test/Accuracy', accuracy, epoch)

    return accuracy

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, required=True, help='model type')
    parser.add_argument('--gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--warm", type=int, default=2, help="warm up epoch")
    parser.add_argument("--save_epoch", type=int, default=10, help="interval between image samples")
    parser.add_argument('--resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--outf', type=str, default='./outf', help='resume training')
    parser.add_argument('--pre_dir', type=str, help='pretrained model path')

    args = parser.parse_args()

    # preparrion
    cudnn.benchmark = True

    path_model = os.path.join(args.outf, 'model', args.net)
    path_runs = os.path.join(args.outf, 'runs', args.net)

    os.makedirs(args.outf, exist_ok=True)
    os.makedirs(path_model, exist_ok=True)
    os.makedirs(path_runs, exist_ok=True)
    writer = SummaryWriter(os.path.join(path_runs, TIME_NOW))




    # data preprocessing
    trainloader, testloader = cifar100_dataloader(
        CIFAR100_MEAN,
        CIFAR100_STD,
        num_workers=8,
        batch_size=args.batch_size,
        shuffle=True
    )

    # model
    model = get_network(args)

    if args.resume:
        if not os.path.exists(args.pre_dir):
            print("Pre-trained model path doesn't exist")
            print("Loading form recent folder")
            recent_folder = most_recent_folder(path_model, fmt=DATE_FORMAT)
            if not recent_folder:
                raise Exception('no recent folder were found')
            ckpt_path = os.path.join(path_model, recent_folder)

            # get best_weights and recent weights
            best_weights = best_acc_weights(ckpt_path)
            if best_weights:
                weights_path = os.path.join(ckpt_path, best_weights)
                print('found best acc weights file:{}'.format(weights_path))
                print('load best training file to test acc...')
                model.load_state_dict(torch.load(weights_path))
                best_acc = eval(tb=False)
                print('best acc is {:0.2f}'.format(best_acc))

            recent_weights_file = most_recent_weights(ckpt_path)
            if not recent_weights_file:
                raise Exception('no recent weights file were found')
            weights_path = os.path.join(ckpt_path, recent_weights_file)
            print('loading weights file {} to resume training.....'.format(weights_path))
            model.load_state_dict(torch.load(weights_path))

            resume_epoch = last_epoch(ckpt_path)

        else:
            print("Loading pretrained model at %s" % args.pre_dir)
            model.load_state_dict(torch.load(args.pre_dir))
            ckpt_path = os.path.join(path_model, TIME_NOW)
    else:
        print("Training from scratch")
        ckpt_path = os.path.join(path_model, TIME_NOW)

    os.makedirs(ckpt_path, exist_ok=True)
    print_options(args, ckpt_path)
    logging.basicConfig(filename=ckpt_path + '/logger.log', level=logging.INFO)
    input_tensor = torch.Tensor(1, 3, 32, 32).cuda()
    writer.add_graph(model, input_tensor)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4) for vgg, hard to converge
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 130, 160, 190], gamma=0.2)
    iter_per_epoch = len(trainloader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    # train
    best_acc = 0.0
    for epoch in range(1, args.epochs):
        if epoch > args.warm:
            train_scheduler.step()

        if args.resume:
            if epoch < resume_epoch:
                continue

        train(epoch)
        acc = eval(epoch)

        if epoch % args.save_epoch == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_path, '{model}-{epoch}-{type}.pth'.format(
                model=args.model, epoch=epoch, type='regular')))
            if acc > best_acc:
                torch.save(model.state_dict(), os.path.join(ckpt_path, '{model}-{epoch}-{type}.pth'.format(
                    model=args.model, epoch=epoch, type='best')))
                best_acc = acc

    writer.close()





