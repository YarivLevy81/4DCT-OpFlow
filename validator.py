import torchio as tio
import numpy as np
from data.dataset import get_dataset
import torch
import SimpleITK as sitk


class Validator(object):

    def __init__(self):
        self.dataset = get_dataset(root="./data/raw")

    @staticmethod
    def make_validation_sample(subject):
        rescale = tio.RescaleIntensity((0, 1))
        crop_or_pad = tio.CropOrPad(target_shape=(256, 256, 128), )
        transformer = tio.RandomElasticDeformation()
        pipe = tio.Compose([rescale, transformer])

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


def create_synt_data(shape=(128, 128, 64)):
    res = np.ones(shape) * .9
    cpts = 7
    xy = [i * 18 + 9 for i in range(cpts)]
    xz = [i * 9 + 4 for i in range(cpts)]
    for i in xy:
        for j in xy:
            for k in xz:
                res[i:i+1, :, :] = 0.05
                res[:, :,k:k+1] = 0.05
                res[:, j:j+1, :] = 0.05

    return res


if __name__ == '__main__':
    validator = Validator()
    # img = validator.dataset[0]
    img = create_synt_data()
    data = img[np.newaxis]
    vox = (1, 1, 1)

    tim = tio.Image(tensor=data, spacing=vox)
    subj = tio.Subject({'img': tim})

    transformed, transformation_data = validator.make_validation_sample(subj)
    transformed.plot()
    subj.plot()

    # check data
    print(torch.all(transformed['img'].data.eq(subj['img'].data)))
    a = transformed['img'].numpy()
    print((a >= 0).all() and (a <= 1).all())

    # Continue
    t = transformation_data[-1]

    control_points = t.control_points

    itk_transform = t.get_bspline_transform(subj['img'].as_sitk(), control_points)
    x, y, z = itk_transform.GetCoefficientImages()
    displ = sitk.TransformToDisplacementField(itk_transform,
                                              sitk.sitkVectorFloat64,
                                              x.GetSize(),
                                              x.GetOrigin(),
                                              x.GetSpacing(),
                                              x.GetDirection())
    vectors = sitk.GetArrayFromImage(displ)
    print(f'image size:{x.GetSize()}, direction:{x.GetDirection()}, spacing:{x.GetSpacing()}, origin:{x.GetOrigin()}')
    print(f'>>>>> Analyzing \n')
    analyze_coor(x)
