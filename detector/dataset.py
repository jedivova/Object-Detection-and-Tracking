from torch.utils.data import Dataset, DataLoader
from transforms import pre_transforms

class Eval_dataset(Dataset):
    def __init__(self, data_dict, augmentations=None, preprocessing=None) -> None:
        self.augmentations = augmentations
        self.preprocessing = preprocessing
        self.Data = data_dict
        self.keys = list(data_dict.keys())

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        key = self.keys[idx]
        image, frame_num = self.Data[key], key
        result = {"image": image}

        if self.augmentations is not None:
            result = self.augmentations(**result)

        if self.preprocessing:
            result = self.preprocessing(**result)

        return result['image'], frame_num

    def get_image(self, key: int):
        return self.Data[key]


def get_loader(frames_dict):
    valid_dataset = Eval_dataset(frames_dict, preprocessing=pre_transforms(image_size=300))
    valid_loader = DataLoader(valid_dataset, batch_size=20, shuffle=False, num_workers=4, drop_last=False)
    return valid_loader, valid_dataset
