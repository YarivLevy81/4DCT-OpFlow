import torch
# from utils.torch_utils import init_seed
import argparse

from data.dataset import get_dataset
from models.pwc3d import get_model
from losses.flow_loss import get_loss
from trainer.get_trainer import get_trainer
import json
import os
from easydict import EasyDict
from utils.misc import VERBOSE


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='4DCT Optical Flow Net')
    parser.add_argument('-c', '--config', default='configs/base.json', help="Path (absolute or relative) for 4DCT data")
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose logs") 
    parser.add_argument('-p', '--plot', action='store_true', help="Plot samples along training")
    parser.add_argument('-l', '--load', help="Model .pth.tar file")
    args = parser.parse_args()

    VERBOSE = args.verbose
    load = args.load
    with open(args.config) as f:
        args = EasyDict(json.load(f))
    args.load = load

    print(f'<<<<< Init experiement >>>>>')
    print(f'args={args}')


    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    train_set = get_dataset(root=args.data_path, w_aug=True)
    valid_set = get_dataset(root=args.valid_path, w_aug=False, data_type=args.valid_type)

    print('{} training samples found'.format(len(train_set)))
    print('{} validation samples found'.format(len(valid_set)))

    # TODO: change batch size to args.batch_size
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=False, shuffle=True
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=1,
        num_workers=8, pin_memory=True, shuffle=False
    )

    model = get_model(args)
    loss = get_loss(args)
    trainer = get_trainer()(
        train_loader, valid_loader, model, loss, args
    )

    trainer.train()
