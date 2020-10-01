import os, math
import numpy as np 
import pandas as pd

from torchvision import models, transforms
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image

from vocab import Vocabulary
from constants import *

class CustomDataset(Dataset):
    """ COCO dataset """
    def __init__(self, root_dir, captions_path, instances_ann_name, image_dir, mode, transform):
        self.transform = transform
        self.mode = mode
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.inst_ann_name = instances_ann_name
        self.coco = COCO(os.path.join(self.root_dir, self.inst_ann_name))

        self.caps_data = pd.read_csv(os.path.join(captions_path))
        print("Captions file read.")

        self.data_ids = self.get_ids()

        self.vocab = Vocabulary(root_dir, captions_path)
        print("Vocab fetch done")
    
    def get_ids(self):
        ids = []
        for _, cap_data in self.caps_data.iterrows():
            for i in range(MAX_CAPTIONS):
                if isinstance(cap_data['caption'+ str(i)], str):
                    ids.append(str(cap_data['img_id']) + ' ' + str(i))
        return ids
    
    def __len__(self):
        return len(self.data_ids)
    
    def __getitem__(self, idx):
        
        img_cap_id = idx.split(' ')
        img_id = img_cap_id[0]
        cap_num = img_cap_id[1]
        
        image = self.coco.loadImgs([int(img_id)])
        image = Image.open(os.path.join(self.root_dir, self.image_dir, image[0]['file_name'])).convert('RGB')
        image = self.transform(image)

        cap_data = self.caps_data[self.caps_data['img_id'] == int(img_id)]
        captions = cap_data['caption' + str(cap_num)]

        caption = ''
        for _, cap in captions.to_dict().items():
            caption = cap        
        tokens = self.vocab.convert_to_tensor(caption)

        return image, tokens