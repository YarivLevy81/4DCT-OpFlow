import torch
#from utils.torch_utils import init_seed
import argparse

from data.dataset import get_dataset
from models.pwc3d import get_model
from losses.flow_loss import get_loss
from trainer.get_trainer import get_trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='4DCT Optical Flow Net')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='number of workers (default: 4)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--data-path', type=str, default='./data/raw',
                        help='Location of dataset')
    parser.add_argument('--pretrained-model', type=str, default=None,
                        help='Location of a PWC3D pretrained model')
    args = parser.parse_args()

    train_set = get_dataset(root=args.data_path)

    print('{} samples found'.format(len(train_set)))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=1,
        num_workers=args.num_workers, pin_memory=False, shuffle=True
    )

    valid_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, shuffle=True
    )

    model = get_model(args)
    loss = get_loss(args)
    trainer = get_trainer()(
        train_loader, valid_loader, model, loss, args 
    )

    trainer.train()