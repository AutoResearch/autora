import torch
from torch.utils.data import Dataset
from cnnmod.StroopNet import StroopNet

class StroopDataset(Dataset):
    """Stroop model data set."""

    def __init__(self, method='random', num_patterns = 100):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self._length = num_patterns

        stimulus = torch.tensor([1, 0, 0, 1], dtype=torch.long)
        task = torch.tensor([1, 0], dtype=torch.long)

        model = StroopNet()
        out = model(stimulus, task)
        print(out)


    def __len__(self):
        return self._length

    def __getitem__(self, idx):


        sample = {'image': image, 'landmarks': landmarks}

        return sample