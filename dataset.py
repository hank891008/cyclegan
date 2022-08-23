from torch.utils.data.dataset import Dataset
from PIL import Image
import os
import config

class dataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.images = os.listdir(root)
        self.length = len(self.images)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.images[idx]
        path = os.path.join(self.root, img)
        img = Image.open(path).convert('RGB')
        return self.transform(img)

if __name__ == '__main__':
    dataset = dataset(f'inputs/trainA', config.transform)
    print(dataset[0].shape)
    from torchvision.utils import save_image
    save_image((dataset[0] + 1) / 2, '1.png')