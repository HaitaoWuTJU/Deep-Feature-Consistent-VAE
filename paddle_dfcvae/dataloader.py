import paddle
import yaml
from paddle.io import Dataset, DataLoader
from paddle.vision import transforms
from paddle.vision.transforms import BaseTransform
from PIL import Image
import pandas
import os
from functools import partial


# class Lambda(BaseTransform):
#     def __init__(self, lambdaStr=None, keys=None):
#         super(Lambda, self).__init__(keys)
#         self.lambdaStr=lambdaStr
class SetRange(BaseTransform):
    def __init__(self, keys=None):
        super(SetRange, self).__init__(keys)

    def _apply_image(self, img):
        return 2 * img - 1


class CelebA(Dataset):
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    def __init__(self, root, split="train", target_type="attr", transform=None):
        super(CelebA, self).__init__()
        self.root = os.path.join(os.path.split(os.path.realpath(__file__))[0],'../dataset')
        self.transform = transform
        self.base_folder = "celeba"
        self.split = split
        self.target_type = target_type
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split = split_map[split]

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)

        mask = slice(None) if split is None else (splits[1] == split)

        self.filename = splits[mask].index.values

        self.attr = paddle.to_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)

    def __getitem__(self, index):
        # print('-----')


        path=os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index])
        # print(path)
        X = Image.open(path)
        # X= paddle.to_tensor(X,'float32')
        # print('-----')
        target = self.attr[index, :]
        if self.transform is not None:
            X = self.transform(X)
        return X, target

    def __len__(self):
        return len(self.attr)


def get_dataloader():
    DIR = os.path.split(os.path.realpath(__file__))[0]
    with open(os.path.join(DIR,'dfc_vae.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    setRange = SetRange()
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.CenterCrop(148),
                                    transforms.Resize(config['exp_params']['img_size']),
                                    transforms.ToTensor(),
                                    setRange])
    trainset = CelebA(root=config['exp_params']['data_path'],
                      split="train",
                      transform=transform)

    testset = CelebA(root=config['exp_params']['data_path'],
                     split="test",
                     transform=transform)
    trainloader = DataLoader(trainset, batch_size=144, num_workers=0, shuffle=True, drop_last=False,use_buffer_reader=True)
    testloader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False, drop_last=False,use_buffer_reader=True)
    msg = 'load dataloader success'
    return trainloader, testloader,len(trainset),msg


if __name__ == "__main__":
    trainloader, testloader, msg = get_dataloader()
    print(len(trainloader))
    for batch in trainloader():
        print(batch[0].shape)
