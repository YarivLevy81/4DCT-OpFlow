import numpy
import pathlib
from torch.utils.data import Dataset
from .dicom_utils import npz_to_ndarray_and_vox_dim as file_processor

class CT_4DDataset(Dataset):
    def __init__(self, root: str):
        root_dir = pathlib.Path(root)
        if not root_dir.exists() or not root_dir.is_dir:
            raise FileExistsError(f"{str(root_dir)} doesn't exist or isn't a directory")

        self.root = root_dir
    
        # Traverse the root directory and count it's size
        self.files = [f for f in self.root.rglob('*')]
        self._len = len(self.files)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        current_file = self.files[index]
        return file_processor(current_file)