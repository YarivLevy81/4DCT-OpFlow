import argparse
from models import pwc3d
import torch
from utils.torch_utils import load_checkpoint
from data.dataset import get_dataset
from random import randint
from utils.visualization_utils import plot_image, plot_flow


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='4DCT-Net demo')
    parser.add_argument('-p', '--pretrained-model', default='./models/dir/4DCT_2021-02-13 00:57_ckpt.pth.tar',
                        help="Model .pth.tar file")
    parser.add_argument('-d', '--data-path', default='./data/raw',
                        help="Path of patients images")
    args = parser.parse_args()

    model = pwc3d.PWC3D(args)

    # Load pretrainged model
    epoch, weights = load_checkpoint(args.pretrained_model)

    from collections import OrderedDict

    new_weights = OrderedDict()
    model_keys = list(model.state_dict().keys())
    weight_keys = list(weights.keys())
    for a, b in zip(model_keys, weight_keys):
        new_weights[a] = weights[b]
    weights = new_weights
    model.load_state_dict(weights)

    train_set = get_dataset(root=args.data_path, w_aug=True)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=1,
        num_workers=4, pin_memory=False, shuffle=True
    )
    rnd_sample = randint(0, len(train_set)-1)

    for i_step, data in enumerate(train_loader):
        if i_step != rnd_sample:
            continue

        img1, img2, _ = data
        flow = model(img1, img2)[0]

        img1 = img1[0].unsqueeze(1).float()  # Add channel dimension
        img2 = img2[0].unsqueeze(1).float()  # Add channel dimension

        # Image 1 plot
        plot_image(img1)

        # Image 2 plot
        plot_image(img2)

        # Flow plot
        plot_flow(flow.unsqueeze(0).float().detach())
        break
