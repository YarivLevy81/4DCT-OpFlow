import torch
# from utils.torch_utils import init_seed
import argparse

from data.dataset import get_dataset
from models.pwc3d import get_model
from losses.flow_loss import get_loss
from trainer.get_trainer import get_trainer
import json
from easydict import EasyDict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='4DCT Optical Flow Net')
    parser.add_argument('-c', '--config', default='configs/base.json') 
    args = parser.parse_args()

    with open(args.config) as f:
        args = EasyDict(json.load(f))
    print(f'<<<<< Init experiement >>>>>')
    print(f'args={args}')

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
        train_loader, valid_loader, model, loss, args)

    trainer.train()
