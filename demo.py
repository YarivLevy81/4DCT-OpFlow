import argparse
from models import pwc3d
import torch
import imageio
import datetime
from utils.torch_utils import load_checkpoint
from data.dataset import get_dataset
from random import randint
from utils.visualization_utils import plot_image, plot_flow, plot_training_fig


# @torch.no_grad()
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='4DCT-Net demo')
    parser.add_argument('-p', '--pretrained-model', default='./models/dir/4DCT_2021-02-13 00:57_ckpt.pth.tar',
                        help="Model .pth.tar file")
    parser.add_argument('-d', '--data-path', default="/mnt/storage/datasets/4DCT/041516 New Cases/training_data",
                        help="Path of patients images")
    parser.add_argument('-v', '--valid-path', default="/mnt/storage/datasets/4DCT/041516 New Cases/training_data",
                        help="Path of validation patients images")
    parser.add_argument('-s', '--synthetic', action='store_true',
                        help="Whether to use synthetic deformation")
    args = parser.parse_args()

    model = pwc3d.PWC3D(args)
    big_flows = True
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

    # train_set = get_dataset(root=args.data_path, w_aug=True)
    inference_set = get_dataset(
        root=args.valid_path, w_aug=False, data_type="variance_valid")

    loader = None
    if not args.synthetic:
        loader = torch.utils.data.DataLoader(
            inference_set, batch_size=1,
            num_workers=4, pin_memory=False, shuffle=False
        )
    else:
        loader = torch.utils.data.DataLoader(
            inference_set, batch_size=1,
            num_workers=8, pin_memory=True, shuffle=False
        )
    filenames = []
    suff = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    # rnd_sample = randint(0, len(train_set)-1)
    for i_step, data in enumerate(loader):
        # if i_step != rnd_sample:
        #   continue
        if i_step < 28:
            continue
        if i_step == 42:
            break
        img1, img2, name = data
        print(name)
        vox_dim = img1[1]
        img1, img2 = img1[0], img2[0]
        img1 = img1.unsqueeze(1).float()  # Add channel dimension
        img2 = img2.unsqueeze(1).float()  # Add channel dimension

        flow_net = model(img1, img2, vox_dim=vox_dim,w_bk=False)['flows_fw'][0].squeeze(0).float()

        # img1 = img1[0].unsqueeze(1).float()  # Add channel dimension
        # img2 = img2[0].unsqueeze(1).float()  # Add channel dimension

        # Image 1 plot
        # plot_image(img1)

        # Image 2 plot
        # plot_image(img2)

        if big_flows:
            flow_net=torch.where(flow_net.detach().double() < 0.25, 0.0, flow_net.detach().double())

        if args.synthetic:
            # Real flow plot
            # plot_flow(flow[0].float().detach())
            print(1)
        plot_training_fig(img1, img2,flow_net.unsqueeze(0), output_path=f'./demo_pics/pic_{i_step}_{suff}.jpg', show=False)
        filenames.append(f'./demo_pics/pic_{i_step}_{suff}.jpg')
        # Net's flow plot
        # plot_flow(flow_net.unsqueeze(0).float().detach())

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(f'./demo_pics/movie{suff}.gif', images, fps=0.5)
