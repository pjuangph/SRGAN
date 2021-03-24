'''
    Reads through the dataset folder, gets a list of images, h,w,depth
'''

from data_utils import recursive_search
import imageio 
import pandas as pd
from tqdm import trange
from multiprocessing import Process, Manager
import os.path as osp

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_image_size(filename:str):
    h = 0
    w = 0
    if (not osp.exists(filename)):
        return {'filename':filename,'height':0,'width':0,'depth':0}

    img = imageio.imread(filename)
    if len(img.shape)>2:
        h, w, d = img.shape
        return {'filename':filename,'height':h,'width':w,'channels':d}
    else:
        h, w = img.shape
        d=1
        return {'filename':filename,'height':h,'width':w,'channels':d}

def evaluate_chunk(L,images):
    image_data = list()
    for img in images:
        image_data.append(get_image_size(img))
    L.extend(image_data)                                    # append data to master list L

num_cores = 16 

if __name__ == '__main__':
    folder = r'D:/datasets/imagenet-object-localization-challenge/imagenet_object_localization_patched2019/ILSVRC\Data/CLS-LOC/'
    # folder = r'D:/datasets/VOC2012/JPEGImages'
    images,_ = recursive_search(folder)

    image_chunks = list(chunks(images,10000))                # Breaks the list into chunks of a particular size
    image_info = list()
    for i in trange(0,len(image_chunks),num_cores):
        with Manager() as manager:
            L = manager.list()  # <-- can be shared between processes.
            processes = list()
            for core in range(num_cores):
                if (i+core)<len(image_chunks):
                    p = Process(target=evaluate_chunk, args=(L,image_chunks[i+core]))  # Each process runs on a chunk of data
                    p.start()
                    processes.append(p)

            for p in processes:
                p.join()                                    # Joins all the data 
            image_info.extend(L)                            # Adds the data to image_info which will be converted to a dataframe
            processes.clear()
    df = pd.DataFrame(image_info)
    df.to_pickle('data/dataset_files.pickle')
    df.to_csv('data/dataset_files.pickle')
