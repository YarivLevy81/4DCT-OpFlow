import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.warp_utils import flow_warp
from utils.misc import log
from .admm.admm import ADMMSolverBlock, MaskGenerator


def get_model(args):
    model = PWC3D(args)

    return model


class PWC3D(nn.Module):
    def __init__(self, args, upsample=True, reduce_dense=True, search_range=4):
        super(PWC3D, self).__init__()
        self.search_range = search_range
        # TODO: num_chs starts from 1 because grayscale?
        self.num_chs = [1, 16, 32, 64, 96, 128, 192]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)
        self.n_frames = 2
        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)

        self.upsample = upsample
        self.reduce_dense = reduce_dense  # CHANGED DEFAULT TO true

        self.corr = Correlation(pad_size=self.search_range, kernel_size=1,
                                max_displacement=self.search_range, stride1=1,
                                stride2=1, corr_multiply=1)

        self.dim_corr = (self.search_range * 2 + 1) ** 3
        # ^3 because we have another dimension
        self.num_ch_in = 32 + (self.dim_corr + 2) * (self.n_frames - 1) + 1
        # Added +1 because it fits the model lol

        if self.reduce_dense:
            self.flow_estimators = FlowEstimatorReduce(self.num_ch_in)
        else:
            self.flow_estimators = FlowEstimatorDense(self.num_ch_in)

        self.context_networks = ContextNetwork(
            (self.flow_estimators.feat_dim + 2) * (self.n_frames - 1) + 1)
        # Added +1 because it fits the model lol

        self.admm_block = ADMMSolverBlock(rho=args.admm_args.rho, lamb=args.admm_args.lamb, eta=args.admm_args.eta, 
                grad=args.admm_args.grad, T=args.admm_args.T)
        self.mask_gen = MaskGenerator(alpha=args.admm_args.alpha, learn_mask=args.admm_args.learn_mask)
        self.apply_admm = args.admm_args.apply_admm

        self.conv_1x1 = nn.ModuleList([conv(192, 32, kernel_size=1,
                                            stride=1, dilation=1),
                                       conv(128, 32, kernel_size=1,
                                            stride=1, dilation=1),
                                       conv(96, 32, kernel_size=1,
                                            stride=1, dilation=1),
                                       conv(64, 32, kernel_size=1,
                                            stride=1, dilation=1),
                                       conv(32, 32, kernel_size=1,
                                            stride=1, dilation=1)])

    def num_parameters(self):
        return sum(
            [p.data.nelement() if p.requires_grad else 0 for p in self.parameters()])

    def init_weights(self, layer):
        if isinstance(layer, nn.Conv3d):
            log(f'Visit nn.Conv3d')
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.ConvTranspose3d):
            log(f'Visit nn.ConvTranspose3d')
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, vox_dim, w_bk=True):

        x1_p = self.feature_pyramid_extractor(x1) + [x1]
        x2_p = self.feature_pyramid_extractor(x2) + [x2]

        masks = []
        masks.append(self.mask_gen(x1, scale=1/4))
        masks.append(self.mask_gen(x2, scale=1/4))

        log(f'Pyramidized inputs')
        res_dict={}
        res_dict['flows_fw'] = self.forward_2_frames(x1_p, x2_p, mask=masks[0])
        if w_bk:
            res_dict['flows_bk'] = self.forward_2_frames(x2_p, x1_p, mask=masks[1])
        return res_dict

    def forward_2_frames(self, x1_p: torch.Tensor, x2_p: torch.Tensor, mask):
        # init
        flows = []
        aux_vars = {
                "q":        [],
                "c":        [],
                "betas":    [],
                "masks":    mask
            }
        
        N, C, H, W, D = x1_p[0].size()
        log(f'Got batch of size {N}')
        init_dtype = x1_p[0].dtype
        init_device = x1_p[0].device
        flow12 = torch.zeros(N, 3, H, W, D, dtype=init_dtype, device=init_device).float()

        log(flow12.size())
        log(f'forward init complete')

        for l, (_x1, _x2) in enumerate(zip(x1_p, x2_p)):
            log(f'Level {l + 1} flow...')
            # warping
            if l == 0:
                x2_warp = _x2
            else:
                flow12 = F.interpolate(flow12 * 2, scale_factor=2,
                                     mode='trilinear', align_corners=True)
                x2_warp = flow_warp(_x2, flow12)

            # correlation
            out_corr = self.corr(_x1, x2_warp)
            out_corr_relu = self.leakyRELU(out_corr)

            # concat and estimate flow
            x1_1by1 = self.conv_1x1[l](_x1)
            log(f'Sizes - x1={_x1.size()}, x2={_x2.size()}, x1_1b1y={x1_1by1.size()}, out_corr_relu = {out_corr_relu.size()}, flow={flow12.size()}')

            x_intm, flow_res = self.flow_estimators(
                torch.cat([out_corr_relu, x1_1by1, flow12], dim=1))
            flow12 = flow12 + flow_res
            log(f'Completed flow estimation')

            log(f'Sizes - x_intm={x_intm.size()}, flow = {flow12.size()}')
            flow_fine = self.context_networks(torch.cat([x_intm, flow12], dim=1))
            log(f'Completed forward of context_networks')
            log(f'Sizes - flow={flow12.size()}, flow_fine={flow_fine.size()}')
            flow12 = flow12 + flow_fine
            flows.append(flow12)
            
            if self.apply_admm[l]:
                Q, C, Betas = self.admm_block(flow12, aux_vars["masks"])
                aux_vars["q"].append(Q)
                aux_vars["c"].append(C)
                aux_vars["betas"].append(Betas)

            if l == self.output_level:
                log(f'Broke flow construction at level {l+1}')
                break

            log(f'Ended iteration of flows')

        if self.upsample:
            flows = [F.interpolate(flow * 4, scale_factor=4,
                                   mode='trilinear', align_corners=True) for flow in flows]

        return flows[::-1], aux_vars


class Correlation(nn.Module):
    def __init__(self, max_displacement=4, *args, **kwargs):
        super(Correlation, self).__init__()
        self.max_displacement = max_displacement
        self.output_dim = 2 * self.max_displacement + 1
        self.pad_size = self.max_displacement

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        N, C, H, W, D = x1.size()

        log(x1.size(), x2.size())
        x2 = F.pad(x2, [self.pad_size] * 6)  # 6 because of 3D
        log(x2.size())
        cv = []
        iter = 0
        log(f'output_dim={self.output_dim}')
        for i in range(self.output_dim):
            for j in range(self.output_dim):
                for k in range(self.output_dim):
                    log(iter)
                    iter += 1
                    log(x2[:, :, i:(i + H), j:(j + W), k:(k + D)].size())
                    cost = x1 * x2[:, :, i:(i + H), j:(j + W), k:(k + D)]
                    cost = torch.mean(cost, 1, keepdim=True)
                    cv.append(cost)

        log("Bye")
        return torch.cat(cv, 1)


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )


class ContextNetwork(nn.Module):
    def __init__(self, ch_in):
        super(ContextNetwork, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 2),
            conv(128, 128, 3, 1, 4),
            conv(128, 96, 3, 1, 8),
            conv(96, 64, 3, 1, 16),
            conv(64, 32, 3, 1, 1),
            conv(32, 3, isReLU=False)
        )

    def forward(self, x):
        return self.convs(x)


class FeatureExtractor(nn.Module):
    def __init__(self, num_chs):
        super(FeatureExtractor, self).__init__()
        self.num_chs = num_chs
        self.convs = nn.ModuleList()

        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            layer = nn.Sequential(
                conv(ch_in, ch_out, stride=2),
                conv(ch_out, ch_out)
            )
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]


class FlowEstimatorDense(nn.Module):
    def __init__(self, ch_in):
        super(FlowEstimatorDense, self).__init__()
        log(f'ch_in={ch_in}')
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(ch_in + 128, 128)
        self.conv3 = conv(ch_in + 256, 96)
        self.conv4 = conv(ch_in + 352, 64)
        self.conv5 = conv(ch_in + 416, 32)
        self.feat_dim = ch_in + 448
        self.conv_last = conv(ch_in + 448, 3, isReLU=False)

    def forward(self, x):
        log(f'Dense estimator')
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out


class FlowEstimatorReduce(nn.Module):
    # can reduce 25% of training time.
    def __init__(self, ch_in):
        super(FlowEstimatorReduce, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(128, 128)
        self.conv3 = conv(128 + 128, 96)
        self.conv4 = conv(128 + 96, 64)
        self.conv5 = conv(96 + 64, 32)
        self.feat_dim = 32
        self.predict_flow = conv(64 + 32, 3, isReLU=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x2, x3], dim=1))
        x5 = self.conv5(torch.cat([x3, x4], dim=1))
        flow = self.predict_flow(torch.cat([x4, x5], dim=1))
        return x5, flow
