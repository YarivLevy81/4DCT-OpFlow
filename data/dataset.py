import numpy
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
        for entry in root_dir.iterdir():
            if entry.is_dir():
                self.patient_directories.append(entry)

        self.patient_directories = sorted(self.patient_directories)

        self.patient_files = []
        for dir in self.patient_directories:
            for file in dir.iterdir():
                if file.is_file():
                    self.patient_files.append(file)

    def __len__(self):
        # We return pairs -> there are len - 1 pairs from len files
        return len(self.patient_files) - 1

    def __getitem__(self, index):
        return file_processor(self.patient_files[index]), file_processor(self.patient_files[index + 1])


def get_dataset(root="./raw"):
    return CT_4DDataset(root=root)
