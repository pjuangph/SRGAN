'''
    Reads through the dataset folder, gets a list of images, h,w,depth
'''

from data_utils import recursive_search
import imageio 
import pandas as pd

def get_image_size(filename):
    h = 0
    w = 0
    img = imageio.imread(filename)
    if len(img.shape)>2:
        h, w, d = img.shape
        return {'filename':filename,'height':h,'width':w,'depth':d}
    else:
        h, w = img.shape
        d=1
        return {'filename':filename,'height':h,'width':w,'depth':d}

if __name__ == '__main__':
    folder = r'D:/datasets/imagenet-object-localization-challenge/imagenet_object_localization_patched2019/ILSVRC\Data/CLS-LOC/'
    images,_ = recursive_search(folder)
    image_data = [get_image_size(x) for x in images]
    df = pd.DataFrame(image_data)    
    df.to_csv('data/dataset_files.csv')
