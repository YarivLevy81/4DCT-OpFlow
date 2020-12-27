import numpy as np
from scipy.ndimage import zoom
import pathlib
from torch.utils.data import Dataset
from .dicom_utils import npz_to_ndarray_and_vox_dim as file_processor


class CT_4DDataset(Dataset):
    def __init__(self, root: str):
        print(pathlib.Path.cwd())
        root_dir = pathlib.Path(root)
        if not root_dir.exists() or not root_dir.is_dir:
            raise FileExistsError(f"{str(root_dir)} doesn't exist or isn't a directory")

        self.root = root_dir

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
        if img1.shape[2] > 128:
            print('non_mat')
            # todo implement solution
        else:
            img1 = pad_img_to_128(img1)
            img2 = pad_img_to_128(img2)
        if self.patient_samples[index]['dim'] == 512:
            img1, img2 = crop_512_imgs_to_256(img1, img2)

        # normalizing data to 0-1
        img1 = img1 / 4196
        img2 = img2 / 4196
        # img1 = zoom(img1, (0.25, 0.25, 0.25))
        # img2 = zoom(img2, (0.25, 0.25, 0.25))
        return (img1, vox_dim1), (img2, vox_dim2)

    def collect_samples(self):
        for entry in self.root.iterdir():
            if entry.is_dir():
                self.patient_directories.append(entry)

        self.patient_directories = sorted(self.patient_directories)

        for directory in self.patient_directories:
            dir_files = []
            for file in directory.iterdir():
                if file.is_file():
                    dir_files.append(file)
            dir_files.sort()
            if len(list(directory.glob('*(256, 256*'))) > 0:
                dim = 256
            else:
                dim = 512
            for idx in range(len(dir_files) - 1):
                self.patient_samples.append({'img1': dir_files[idx], 'img2': dir_files[idx + 1], 'dim': dim})


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


def get_dataset(root="./raw"):
    return CT_4DDataset(root=root)
