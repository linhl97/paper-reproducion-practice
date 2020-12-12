import argparse
import torch
from utils.dataset import cifar100_dataloader, CIFAR100_STD, CIFAR100_MEAN
from utils.utils import get_network, WarmUpLR, most_recent_folder, best_acc_weights, DATE_FORMAT, TIME_NOW,\
    most_recent_weights, last_epoch, print_options

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True, help='net type')
    parser.add_argument('--weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('--gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()

    model = get_network(args)

    trainloader, testloader = cifar100_dataloader(
        CIFAR100_MEAN,
        CIFAR100_STD,
        num_workers=0,
        batch_size=args.batch_size,
    )

    model.load_state_dict(torch.load(args.weights))
    print(model)
    model.eval()

    correct_1 = 0
    correct_5 = 0

    with torch.no_grad():
        for niter, (images, labels) in enumerate(testloader):
            print("iteration: {}\ttotal {} iterations".format(niter + 1, len(testloader)))

            if args.gpu:
                images = images.cuda()
                labels = labels.cuda()

            output = model(images)
            _, pred = torch.topk(output, 5, dim=1)

            labels = labels.view(labels.size()[0], -1).expand_as(pred)
            correct = labels.eq(pred).float()

            correct_1 += correct[:, 0].sum()
            correct_5 += correct[:, :5].sum()

        print("Top 1 err: ", 1 - correct_1 / len(testloader.dataset))
        print("Top 5 err: ", 1 - correct_5 / len(testloader.dataset))
        print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))