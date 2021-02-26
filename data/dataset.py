import numpy as np
from scipy.ndimage import zoom
import pathlib
from torch.utils.data import Dataset
import torch.nn.functional as F

from .dicom_utils import npz_to_ndarray_and_vox_dim as file_processor
from .dicom_utils import npz_valid_to_ndarrays_flow_vox as vld_file_processor

from .data_augmentor import pre_augmentor, pre_validation_set


class CT_4DDataset(Dataset):
    def __init__(self, root: str, w_aug=False):
        print(pathlib.Path.cwd())
        root_dir = pathlib.Path(root)
        if not root_dir.exists() or not root_dir.is_dir:
            raise FileExistsError(
                f"{str(root_dir)} doesn't exist or isn't a directory")

        self.root = root_dir
        self.w_augmentations = w_aug

        # Traverse the root directory and count it's size
        self.patient_directories = []
        self.patient_samples = []
        self.collect_samples()

    def __len__(self):
        # We return pairs -> there are len - 1 pairs from len files
        return len(self.patient_samples)

    def __getitem__(self, index):
        img1, vox_dim1 = file_processor(self.patient_samples[index]['img1'])
        img2, vox_dim2 = file_processor(self.patient_samples[index]['img2'])
        sample_name = self.patient_samples[index]['name']
        if img1.shape[2] > 128:
            print('non_mat')
            # todo implement solution

        if self.patient_samples[index]['dim'] == 512:
            img1, img2 = crop_512_imgs_to_256(img1, img2)

        p1, p2 = pre_augmentor(img1, img2, vox_dim1, self.w_augmentations)
        return p1, p2, sample_name

    def collect_samples(self):
        for entry in self.root.iterdir():
            if entry.is_dir():
                self.patient_directories.append(entry)

        self.patient_directories = sorted(self.patient_directories)

        # import zipfile
        for directory in self.patient_directories:
            dir_files = []
            for file in directory.iterdir():
                # print(file)
                # z = zipfile.ZipFile(file)
                # if z.testzip() is not None:
                #     print(file)
                if file.is_file() and file.suffix == '.npz':
                    dir_files.append(file)
            dir_files.sort(key=take_name)
            if len(list(directory.glob('*(256, 256*'))) > 0:
                dim = 256
            else:
                dim = 512
            for idx in range(len(dir_files) - 1):
                sample_name = dir_files[idx].name
                sample_name = sample_name[sample_name.index(
                    '_'):sample_name.index('(')]
                name = dir_files[idx].parent.name + sample_name
                self.patient_samples.append(
                    {'name': name, 'img1': dir_files[idx], 'img2': dir_files[idx + 1], 'dim': dim})


class CT_4DValidationset(Dataset):
    def __init__(self, root: str):
        print(pathlib.Path.cwd())
        root_dir = pathlib.Path(root)
        if not root_dir.exists() or not root_dir.is_dir:
            raise FileExistsError(
                f"{str(root_dir)} doesn't exist or isn't a directory")

        self.root = root_dir

        # Traverse the root directory and count it's size
        self.validation_tuples = []
        self.collect_samples()

    def __len__(self):
        return len(self.validation_tuples)

    def __getitem__(self, index):
        p1, p2, flow12 = vld_file_processor(
            self.validation_tuples[index])
        return p1, p2, flow12

    def collect_samples(self):
        for entry in self.root.iterdir():
            dir_files = []
            if entry.is_file():
                dir_files.append(entry)
            dir_files.sort()
            for idx in range(len(dir_files)):
                self.validation_tuples.append(dir_files[idx])


class CT_4D_Variance_Valid_set(Dataset):
    def __init__(self, root: str, w_aug=False, set_length=5, num_of_sets=15):
        print(pathlib.Path.cwd())
        root_dir = pathlib.Path(root)
        if not root_dir.exists() or not root_dir.is_dir:
            raise FileExistsError(
                f"{str(root_dir)} doesn't exist or isn't a directory")

        self.root = root_dir
        self.w_augmentations = w_aug
        self.set_length = set_length
        self.num_of_sets = num_of_sets

        # Traverse the root directory and count it's size
        self.patient_directories = []
        self.patient_sets = []
        self.collect_samples()

    def __len__(self):
        return len(self.patient_sets)

    def __getitem__(self, index):
        idx = [i + 1 for i in range(self.set_length)]
        image_tuples = []
        for i in idx:
            image_tuples.append(file_processor(self.patient_sets[index][f'img{i}']))

        sample_name = self.patient_sets[index]['name']
        if image_tuples[0][0].shape[2] > 128:
            print('non_mat')
            # todo implement solution

        if self.patient_sets[index]['dim'] == 512:
            for idx in range(len(image_tuples)):
                image_tuples[idx] = resize_512_to_256(image_tuples[idx])
        image_tuples = pre_validation_set(image_tuples, self.w_augmentations)
        return image_tuples, sample_name

    def collect_samples(self):
        for entry in self.root.iterdir():
            if entry.is_dir():
                self.patient_directories.append(entry)

        self.patient_directories = sorted(self.patient_directories)

        for directory in self.patient_directories:
            dir_files = []
            for file in directory.iterdir():
                if file.is_file() and file.suffix == '.npz':
                    dir_files.append(file)
            dir_files.sort(key=take_name)

            if len(list(directory.glob('*(256, 256*'))) > 0:
                dim = 256
            else:
                dim = 512
            if len(dir_files) < self.set_length:
                continue
            idx = [i for i in range(self.set_length)]
            name = dir_files[idx[0]].parent.name
            set_dict = {f'img{i + 1}': dir_files[i] for i in idx}
            set_dict['name'] = name
            set_dict['dim'] = dim

            self.patient_sets.append(set_dict)


def pad_img_to_128(img):
    pad = np.zeros((img.shape[0], img.shape[1], 128))
    pad[:img.shape[0], :img.shape[1], :img.shape[2]] = img
    return pad


def crop_512_imgs_to_256(image1, image2):
    # returns a random quarter of an image
    i = np.random.randint(2)
    j = np.random.randint(2)
    return (
        image1[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256, :],
        image2[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256, :])


def resize_512_to_256(img_tup):
    img = img_tup[0]
    z = img.shape[2]
    vox_dim = img_tup[1]
    img = F.interpolate(img, size=[256, 256, z])
    vox_dim[0] = vox_dim[0] / 2
    vox_dim[1] = vox_dim[1] / 2

    return img, vox_dim


def get_dataset(root="./raw", w_aug=False, data_type='train'):
    if data_type == 'train':
        return CT_4DDataset(root=root, w_aug=w_aug)
    if data_type == 'valid':
        return CT_4DDataset(root=root, w_aug=w_aug)
        # return CT_4DValidationset(root)
    if data_type == 'variance_valid':
        return CT_4D_Variance_Valid_set(root=root, w_aug=w_aug)


def take_name(file_path):
    name = file_path.name
    name = name[name.index('_') + 1:(name.index('(') - 1)]
    return int(name)
