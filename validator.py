import torchio as tio
import numpy as np
from data.dataset import get_dataset
import torch


class Validator(object):

    def __init__(self):
        self.dataset = get_dataset(root="./data/raw")

    @staticmethod
    def make_validation_sample(subject):
        rescale = tio.RescaleIntensity((0, 1))
        crop_or_pad = tio.CropOrPad(target_shape=(256, 256, 128), )
        transformer = tio.RandomElasticDeformation()
        pipe = tio.Compose([rescale, crop_or_pad, transformer])

        _transformed = pipe(subject)
        _transformation_data = _transformed.history
        return _transformed, _transformation_data


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


if __name__ == '__main__':
    validator = Validator()
    img = validator.dataset[0]
    data = img[0][0][np.newaxis]
    vox = img[0][1]

    tim = tio.Image(tensor=data, spacing=vox)
    subj = tio.Subject({'img': tim})

    transformed, transformation_data = validator.make_validation_sample(subj)
    # transformed.plot()
    # subj.plot()

    # check data
    print(torch.all(transformed['img'].data.eq(subj['img'].data)))
    a = transformed['img'].numpy()
    print((a >= 0).all() and (a <= 1).all())

    # Continue
    t = transformation_data[-1]

    control_points = t.control_points

    itk_transform = t.get_bspline_transform(subj['img'].as_sitk(), control_points)
    x, y, z = itk_transform.GetCoefficientImages()

    print(f'>>>>> Analyzing \n')
    analyze_coor(x)
