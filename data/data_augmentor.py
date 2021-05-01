import torchio as tio
import numpy as np


def no_transform(x):
    return x


def get_transforms(w_aug):
    rescale = tio.RescaleIntensity((0, 1))
    crop_or_pad = tio.CropOrPad(target_shape=(192, 192, 192), )
    if w_aug:
        transforms_dict = {tio.RandomAffine(): 0.4, tio.RandomElasticDeformation(): 0.4,
                           tio.RandomFlip(): 0.1, tio.Lambda(no_transform): 0.1}
        pipe = tio.Compose([rescale, crop_or_pad, tio.OneOf(transforms_dict)])
    else:
        pipe = tio.Compose([rescale, crop_or_pad])
    return pipe


def get_pair_transformed_images(trans_subj: tio.Subject, org_vox_dims, plot=False):
    transformed_imgs = trans_subj.get_images()
    aug_im1 = transformed_imgs[0].data.squeeze()
    aug_im2 = transformed_imgs[1].data.squeeze()
    if plot: trans_subj.plot()
    return aug_im1, aug_im2, org_vox_dims


def get_transformed_images(trans_subj: tio.Subject, org_vox_dims, num_of_images, plot=False):
    transformed_imgs = trans_subj.get_images()
    images = [(transformed_imgs[i].data.squeeze(), org_vox_dims) for i in range(num_of_images)]
    if plot: trans_subj.plot()
    return images


def pre_augmentor(img1, img2, vox, w_aug: bool, plot=False):
    tim1 = tio.Image(tensor=img1[np.newaxis], spacing=vox)
    tim2 = tio.Image(tensor=img2[np.newaxis], spacing=vox)
    subj = tio.Subject({'one image': tim1, 'two image': tim2})
    if plot: subj.plot()
    transforms = get_transforms(w_aug)
    trans_subj = transforms(subj)
    aug_img1, aug_img2, aug_vox = get_pair_transformed_images(trans_subj, vox)
    return (aug_img1, aug_vox), (aug_img2, aug_vox)


def pre_validation_set(image_tuples, vox, w_aug: bool, plot=False):
    tio_imgs = [tio.Image(tensor=tup[0][np.newaxis], spacing=tup[1]) for tup in image_tuples]
    subj = tio.Subject({f'{idx} image': img for idx, img in enumerate(tio_imgs)})
    if plot: subj.plot()
    transforms = get_transforms(w_aug)
    trans_subj = transforms(subj)
    images = get_transformed_images(trans_subj, vox, len(image_tuples))
    return images
