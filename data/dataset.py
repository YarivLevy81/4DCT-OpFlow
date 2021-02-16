import numpy as np
from scipy.ndimage import zoom
import pathlib
from torch.utils.data import Dataset
from .dicom_utils import npz_to_ndarray_and_vox_dim as file_processor
from .dicom_utils import npz_valid_to_ndarrays_flow_vox as vld_file_processor

from .data_augmentor import pre_augmentor


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

        #import zipfile
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


def get_dataset(root="./raw", w_aug=False, data_type='train'):
    if data_type == 'train':
        return CT_4DDataset(root=root, w_aug=w_aug)
    if data_type == 'valid':
        return CT_4DValidationset(root)


def take_name(file_path):
    name = file_path.name
    name = name[name.index('_')+1:(name.index('(')-1)]
    return int(name)
