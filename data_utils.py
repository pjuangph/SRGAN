import os
import pandas as pd
from typing import List
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, InterpolationMode
from io import StringIO
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

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
    '''
        Recursively search a directory for Images 

        Args
            dataset_dir - directory to recursively search 
        
        Returns
            image_filenames, subfolders (do not use)
    ''' 
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

class DataFrameQuery:
    '''
        Class to help query the dataframe for a specific height and width
    '''
    def __init__(self,df:pd.DataFrame,df_grouping:pd.DataFrame):
        '''
            This saves memory by retaining df and df_grouping
            df - image dataframe with path to all images and their height, width, channels
            df_grouping - a dataframe containing height, width, count_of_images
        '''
        self.df = df
        self.df_grouping =df_grouping

    def query_func(self,row_indx):
        '''
            Query the dataframe for matching all values height and width
        '''
        row = self.df_grouping.iloc[row_indx]
        return self.df[(self.df['height']==row['height']) & (self.df['width']==row['width'])]

    

def dataframe_find_similar_images(df:pd.DataFrame,batch_size=1):
    '''
        Finds images in dataframe df - columns height, width, channels that are similar in dimensions
        https://stackoverflow.com/questions/35268817/unique-combinations-of-values-in-selected-columns-in-pandas-data-frame-and-count

        Returns
            df_grouping - height, width, count
            dataframes - a list containing dataframes of common height and width 
    '''

    df_grouping = df.groupby(['height','width']).size().reset_index().rename(columns={0:'count'})
    df_grouping.sort_values('count',ascending=False, inplace=True)

    dfq = DataFrameQuery(df,df_grouping)
    dataframes = None
    n_cpu = 12 
    tqdm.pandas()
    with ProcessPoolExecutor(max_workers=n_cpu) as executor:
        dataframes = list(
            tqdm(executor.map(dfq.query_func, range(df_grouping.shape[0]), chunksize=4096*2),
                    desc=f"Processing {len(df_grouping)} examples on {n_cpu} cores",
                    total=len(df_grouping)))
    dataframes = [df for df in dataframes if len(df)>batch_size]
    return df_grouping,dataframes


def create_validation_groups(df:pd.DataFrame):
    '''
        Create groups based on validation data, same height, width
    '''
    # Loop for all heights that are not duplicates
    # find images with similar widths 

class TrainDatasetFromList(Dataset):
    def __init__(self, dataset_list:List[str], crop_size:int, upscale_factor:int):
        '''
            Folder to grab all images from. If images are nested inside the folders, this will look inside a single directory. 
        '''
        super(TrainDatasetFromList, self).__init__()        
        self.image_filenames = dataset_list
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]).convert('RGB'))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

class ValDatasetFromList(Dataset):
    def __init__(self, dataset_list:List[str], upscale_factor:int):
        super(ValDatasetFromList, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = dataset_list


    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index]).convert('RGB')
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=InterpolationMode.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=InterpolationMode.BICUBIC)
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
