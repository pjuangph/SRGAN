import os
from os.path import join

from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from io import StringIO
from get_image_size import get_image_size
from tqdm import trange 
torch.manual_seed(17)



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])

def recursive_search(dataset_dir):        
    image_filenames = list()
    subfolders = list()
    for f in os.scandir(dataset_dir):
        if f.is_file():
            if is_image_file(f.path):
                image_filenames.append(f.path)
        elif f.is_dir():
            subfolders.append(f)
    for dir in subfolders:
        files, subF = recursive_search(dir)
        subfolders.extend(subF)
        image_filenames.extend(files)
    
    return image_filenames, subfolders


class TrainDatasetFromFolder(Dataset):


    def __init__(self, dataset_dir, crop_size, upscale_factor):
        '''
            Folder to grab all images from. If images are nested inside the folders, this will look inside a single directory. 
        '''
        super(TrainDatasetFromFolder, self).__init__()        
        self.image_filenames, _ = recursive_search(dataset_dir)
        keep_images = list()
        
        # Get Images large than crop_size
        for i in trange(len(self.image_filenames),desc='Selecting images that fit crop size'):
            w,h = get_image_size(self.image_filenames[i])
            if w>crop_size and h >crop_size:
                keep_images.append(i)
        self.image_filenames = [self.image_filenames[i] for i in keep_images]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        p = Image.open(self.image_filenames[index])
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

    
    

class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames, _ = recursive_search(dataset_dir)


    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)                      # This is initializes the centercrop class and forwards an image to it
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames,_ = recursive_search(dataset_dir)
        self.hr_filenames,_ = recursive_search(dataset_dir)

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
