import torch
#from utils.torch_utils import init_seed

from data.dataset import get_dataset
from models.pwc3d import get_model
from losses.flow_loss import get_loss
from trainer.get_trainer import get_trainer


def main(cfg, _log):
    #init_seed(cfg.seed)

    _log.info("=> fetching img pairs.")
    train_set = get_dataset(cfg)

    _log.info('{} samples found'.format(
        len(train_set)))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.train.batch_size,
        num_workers=cfg.train.workers, pin_memory=True, shuffle=True)

    if cfg.train.epoch_size == 0:
        cfg.train.epoch_size = len(train_loader)

    cfg.train.epoch_size = min(cfg.train.epoch_size, len(train_loader))

    model = get_model()#cfg.model)
    loss = get_loss()#cfg.loss)
    trainer = get_trainer()(
        train_loader, model, loss, _log, cfg.save_root, cfg.train)

    trainer.train()
