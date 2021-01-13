import torchio as tio
import numpy as np


def no_transform(x):
    return x


def get_transforms():
    rescale = tio.RescaleIntensity((0, 1))
    crop_or_pad = tio.CropOrPad(target_shape=(256, 256, 128),)
    transforms_dict = {tio.RandomAffine(): 0.4, tio.RandomElasticDeformation(): 0.4, tio.RandomFlip(): 0.1,
                       tio.Lambda(no_transform): 0.1}
    pipe = tio.Compose([rescale, crop_or_pad, tio.OneOf(transforms_dict)])

    return pipe


def get_transformed_images(trans_subj: tio.Subject, org_vox_dims, plot=False):
    transformed_imgs = trans_subj.get_images()
    aug_im1 = transformed_imgs[0].data.squeeze()
    aug_im2 = transformed_imgs[1].data.squeeze()
    if plot: trans_subj.plot()
    return aug_im1, aug_im2, org_vox_dims


def pre_augmentor(img1, img2, vox, plot=False):
    tim1 = tio.Image(tensor=img1[np.newaxis], spacing=vox)
    tim2 = tio.Image(tensor=img2[np.newaxis], spacing=vox)
    subj = tio.Subject({'one image': tim1, 'two image': tim2})
    if plot: subj.plot()
    transforms = get_transforms()
    trans_subj = transforms(subj)
    aug_img1, aug_img2, aug_vox = get_transformed_images(trans_subj, vox)
    return (aug_img1, aug_vox), (aug_img2, aug_vox)
