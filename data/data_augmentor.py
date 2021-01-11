import torchio as tio
import numpy as np


def get_transforms():
    transforms = [tio.RandomAffine(), tio.RandomElasticDeformation(), tio.RandomFlip()]
    return tio.OneOf(transforms)


def get_transformed_images(trans_subj: tio.Subject, org_vox_dims):
    transformed_imgs = trans_subj.get_images()
    aug_im1 = transformed_imgs[0].data.squeeze()
    aug_im2 = transformed_imgs[1].data.squeeze()
    trans_subj.plot()
    return aug_im1, aug_im2, org_vox_dims


def augmentor(img1, img2, vox):
    tim1 = tio.Image(tensor=img1[np.newaxis], spacing=vox)
    tim2 = tio.Image(tensor=img2[np.newaxis], spacing=vox)
    subj = tio.Subject({'one image': tim1, 'two image': tim2})
    transforms = get_transforms()
    trans_subj = transforms(subj)
    aug_img1, aug_img2, aug_vox = get_transformed_images(trans_subj, vox)
    return (aug_img1, aug_vox), (aug_img2, aug_vox)
