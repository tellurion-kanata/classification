import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional

import os
import json
from glob import glob
import PIL.Image as Image
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
def get_transforms(opt, grayscale=False):
    transform_list = []
    if not opt.no_resize:
        transform_list += [transforms.Resize((opt.load_size, opt.load_size))]
    if not opt.no_crop:
        transform_list += [transforms.RandomCrop((opt.crop_size, opt.crop_size))]
    if not opt.no_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    if not opt.no_rotate:
        transform_list += [transforms.RandomRotation((0, 360))]

    transform_list += [transforms.ToTensor()]
    if grayscale:
        transform_list += [transforms.Normalize((0.5,), (0.5,))]
    else:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


def normalize(img):
    img = transforms.ToTensor()(img)
    img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
    return img


class ImageDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        if not os.path.exists(self.opt.dataroot):
            raise FileNotFoundError('data file is not found.')
        self.num_classes = self.opt.num_classes
        self.image_root = os.path.join(self.opt.dataroot, 'color')
        self.tags_root = os.path.join(self.opt.dataroot, 'tags')
        self.image_files = glob(os.path.join(self.image_root, '*/*.jpg'))
        self.transforms = get_transforms(self.opt)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file).convert('RGB')
        image = self.transforms(image)

        tags_file = image_file.replace(self.image_root, self.tags_root).replace('jpg', 'json')
        with open(tags_file, 'r') as f:
            img_dict = json.load(f)
        class_vector = torch.zeros([self.num_classes])
        class_vector[img_dict['tags']] = 1

        return {'image': image,
                'class': class_vector,
                'index': index}

    def __len__(self):
        return len(self.image_files)

class TestDataset():
    def __init__(self, opt):
        self.opt = opt
        self.opt.no_flip = True
        self.num_classes = self.opt.num_classes
        self.incl_label = self.opt.gt_label

        if self.incl_label:
            self.image_root = os.path.join(self.opt.dataroot, 'image')
            self.tags_root = os.path.join(self.opt.dataroot, 'tags')
            self.image_files = glob(os.path.join(self.image_root, '*.jpg'))
        else:
            self.image_files = glob(os.path.join(self.opt.dataroot, '*.jpg'))

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file).convert('RGB')
        image = normalize(image)

        ret_dict = {}
        ret_dict['image'] = image
        ret_dict['path'] = image_file

        if self.incl_label:
            tags_file = image_file.replace(self.image_root, self.tags_root).replace('jpg', 'json')
            with open(tags_file, 'r') as f:
                img_dict = json.load(f)
            class_vector = torch.zeros([self.num_classes])
            for tag in img_dict['tags']:
                class_vector[tag] = 1.
            ret_dict['class'] = class_vector

        return ret_dict

    def __len__(self):
        return len(self.image_files)

class CustomDataLoader():
    def initialize(self, opt):
        if not opt.eval:
            self.dataset = ImageDataset(opt)
        else:
            self.dataset = TestDataset(opt)

        self.dataLoader = data.DataLoader(
            dataset = self.dataset,
            batch_size = opt.batch_size,
            shuffle = not opt.no_shuffle and not opt.eval,
            num_workers = opt.num_threads)

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for data in self.dataLoader:
            yield data