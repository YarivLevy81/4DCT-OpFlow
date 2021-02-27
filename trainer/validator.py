import sys
import os
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)
import torchio as tio
import numpy as np

import torch
import SimpleITK as sitk
from data.dataset import get_dataset
from data.dicom_utils import validation_to_npz
from utils.warp_utils import flow_warp
from utils.visualization_utils import plot_image, plot_flow
from trainer.deformations import RandomNormalElasticDeformation

import random


class Validator(object):

    def __init__(self):
        # self.dataset = get_dataset(
        #     root="./data/raw", w_aug=False)
        self.dataset = get_dataset(
            root="/mnt/storage/datasets/4DCT/041516 New Cases/training_data", w_aug=False)
        self.validation_batch_size = 4

    @staticmethod
    def make_validation_sample(subject):
        rescale = tio.RescaleIntensity((0, 1))
        crop_or_pad = tio.CropOrPad(target_shape=(256, 256, 128), )
        #transformer = tio.RandomElasticDeformation()
        transformer = RandomNormalElasticDeformation(num_control_points=4, max_displacement=1, locked_borders=1)
        pipe = tio.Compose([rescale, crop_or_pad, transformer])

        _transformed = pipe(subject)
        _transformation_data = _transformed.history
        return _transformed, _transformation_data

    def get_validation_batch(self):
        validation_batch = []

        for _ in range(self.validation_batch_size):
            num_samples = len(self.dataset)
            next_index = random.randint(0, num_samples - 1)

            new_img = self.dataset[next_index]
            data = new_img[0][0][np.newaxis]
            vox_dim = new_img[0][1]

            tim = tio.Image(tensor=data, spacing=vox)
            tim_itk = tim.as_sitk()
            subj = tio.Subject({'img': tim})

            transformed, transformation_data = validator.make_validation_sample(
                subj)

            t = transformation_data[-1]
            control_points = t.control_points

            itk_transform = t.get_bspline_transform(
                subj['img'].as_sitk(), control_points)
            x, y, z = itk_transform.GetCoefficientImages()
            displ = sitk.TransformToDisplacementField(itk_transform,
                                                      sitk.sitkVectorFloat64,
                                                      tim_itk.GetSize(),
                                                      tim_itk.GetOrigin(),
                                                      tim_itk.GetSpacing(),
                                                      tim_itk.GetDirection())

            transformed_img = transformed.get_images()
            aug_im1 = transformed_img[0].data
            vectors = sitk.GetArrayFromImage(displ).T
            vectors[2, :, :, :] *= (-1)
            v_as_torch = torch.from_numpy(vectors).unsqueeze(0).float()


def create_and_save_validation_triplet(img1, vox, name, target_dir):
    tim = tio.Image(tensor=img1, spacing=vox)
    tim_itk = tim.as_sitk()
    subj = tio.Subject({'img': tim})

    transformed, transformation_data = Validator.make_validation_sample(subj)
    t = transformation_data[-1]
    control_points = t.control_points
    itk_transform = t.get_bspline_transform(
        subj['img'].as_sitk(), control_points)
    x, y, z = itk_transform.GetCoefficientImages()
    displ = sitk.TransformToDisplacementField(itk_transform,
                                              sitk.sitkVectorFloat64,
                                              tim_itk.GetSize(),
                                              tim_itk.GetOrigin(),
                                              tim_itk.GetSpacing(),
                                              tim_itk.GetDirection())
    vectors = sitk.GetArrayFromImage(displ).T
    vectors[2, :, :, :] *= (-1)
    v_as_torch = torch.from_numpy(vectors).unsqueeze(0).float()
    transformed_recons = flow_warp(data.unsqueeze(0).float(), v_as_torch)
    out_img1 = data.squeeze(0).numpy()
    out_img2 = transformed_recons.squeeze(0).squeeze(0).numpy()
    validation_to_npz(target_dir, name, out_img1, vox, out_img2, vox, vectors)
    return True


def analyze_coor(coor):
    print(control_points.shape)
    u = control_points[..., 0].T
    v = control_points[..., 1].T
    w = control_points[..., 2].T
    size_i, size_j, size_k = x.GetSize()
    points = []
    for i in range(size_i):
        for j in range(size_j):
            for k in range(size_k):
                point = coor.TransformIndexToPhysicalPoint((i, j, k))
                points.append(point)

    points = np.array(points)
    u = u.flatten()
    v = v.flatten()
    w = w.flatten()
    print(points.shape)
    print(u.shape, v.shape, w.shape)

    for i in range(u.shape[0]):
        print(f'Point -> {points[i]}: u -> {u[i]}, v -> {v[i]}, w -> {w[i]}')


def create_synt_data(shape=(128, 128, 64)):
    res = np.ones(shape) * .9
    cpts = 7
    y = [i * 18 + 9 for i in range(cpts)]
    x = [i * 9 + 4 for i in range(cpts * 2)]
    xz = [i * 7 + 3 for i in range(cpts + 2)]
    for i in x:
        for j in y:
            for k in xz:
                if i % 2 == 0:
                    res[i - 2:i + 2, :, :] = 0.05
                else:
                    res[i:i + 1, :, :] = 0.05
                res[:, :, k:k + 1] = 0.05
                res[:, j:j + 1, :] = 0.05

    return res


def find_zer_coors(img):
    H = W = 128
    D = 64
    for i in range(H):
        for j in range(W):
            for k in range(D):
                if img[0, 0, i, j, k] <= 0.3:
                    print(f'({i},{j},{k})')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Validator of 4DCT-Net')
    parser.add_argument('-s', '--synthetic', action='store_true',
                        help="Whether to use synthetic or real data")
    args = parser.parse_args()

    validator = Validator()
    num_of_valid_samples_to_create = 50
    target_dir = '/mnt/storage/datasets/4DCT/041516 New Cases/validation_normal_data'
    for idx in range(num_of_valid_samples_to_create):
        print(f'creating sample no\'{idx+1}')
        if args.synthetic:
            img = create_synt_data()
            data = torch.from_numpy(img[np.newaxis])
            vox = (1, 1, 1)
        else:
            i=idx*int(len(validator.dataset)/num_of_valid_samples_to_create)
            img = validator.dataset[i]
            data = img[1][0][np.newaxis]
            vox = img[0][1]
        create_and_save_validation_triplet(data, vox, img[2], target_dir)
    # tim = tio.Image(tensor=data, spacing=vox)
    # tim_itk = tim.as_sitk()
    # subj = tio.Subject({'img': tim})
    #
    # transformed, transformation_data = validator.make_validation_sample(subj)
    # subj.plot()
    # transformed.plot()
    #
    # t = transformation_data[-1]
    # control_points = t.control_points
    #
    # itk_transform = t.get_bspline_transform(subj['img'].as_sitk(), control_points)
    # x, y, z = itk_transform.GetCoefficientImages()
    # displ = sitk.TransformToDisplacementField(itk_transform,
    #                                           sitk.sitkVectorFloat64,
    #                                           tim_itk.GetSize(),
    #                                           tim_itk.GetOrigin(),
    #                                           tim_itk.GetSpacing(),
    #                                           tim_itk.GetDirection())
    #
    # transformed_img = transformed.get_images()
    # aug_im1 = transformed_img[0].data
    # vectors = sitk.GetArrayFromImage(displ).T
    # vectors[2,:,:,:]*=(-1)
    # # vectors = np.zeros(vectors.shape) * 2
    # # vectors[2,:,:,:]=2
    # v_as_torch = torch.from_numpy(vectors).unsqueeze(0).float()
    # transformed_recons = flow_warp(data.unsqueeze(0).float(), v_as_torch)
    # # plot_image(aug_im1)
    # # transformed_recons = flow_warp(aug_im1.unsqueeze(0).float(), v_as_torch)
    # plot_image(transformed_recons)
    # plot_flow(v_as_torch)
    # # find_zer_coors(data.unsqueeze(0).float())
    # # print('looking in transformed')
    # # find_zer_coors(transformed_recons)
    # # tim = tio.Image(tensor=transformed_recons.squeeze(0), spacing=vox)
    # # subj = tio.Subject({'img': tim})
    # # subj.plot()
    # out_img1=data.squeeze(0).numpy()
    # out_img2= data.squeeze(0).squeeze(0).numpy()
    # print('done')
    # root = "./data/validation/"
    # root_dir = pathlib.Path(root)
    # # validation_to_npz(root,"4_2",out_img1,vox,out_img2,vox,vectors)
