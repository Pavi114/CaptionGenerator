from pycocotools.coco import COCO
import numpy as np 
import os, requests
import pandas as pd

"""
    Download image and caption annotations file from COCO site
"""

direc = '../COCO/' # COCO Direc
data_type = 'train2014' # Data file to be used

img_ann_path = '{}annotations_trainval2014/annotations/instances_{}.json'.format(direc, data_type)
coco_imgs = COCO(img_ann_path)

caps_ann_path = '{}annotations_trainval2014/annotations/captions_{}.json'.format(direc, data_type)
coco_caps = COCO(caps_ann_path)

def fetch_images():
    categories = ['bed', 'wine glass', 'teddy bear', 'sports ball']

    images = []

    if len(categories) > 0:
        for cat in categories:
            cat_ids = coco_imgs.getCatIds(catNms=cat)
            img_ids = coco_imgs.getImgIds(catIds=cat_ids)
            images += coco_imgs.loadImgs(img_ids)
    else:
        pass

    # getting unique set of images
    unique_images = []

    for image in images:
        if image not in unique_images:
            unique_images.append(image)

    images = unique_images
    # print(len(unique_images))
    return images

def download_images(image_info):
    caption_data = {
        'img_id': [],
        'caption0': [],
        'caption1': [],
        'caption2': [],
        'caption3': [],
        'caption4': []
    }

    for image in image_info:
        # print(image['id'])

        # fetching image
        img_content = requests.get(image['coco_url']).content
        with open(os.path.join(direc, 'train', 'images', image['file_name']), 'wb') as f:
            f.write(img_content)
        
        # fetching captions
        cap = coco_caps.getAnnIds(imgIds=image['id'])
        anns = coco_caps.loadAnns(cap)
        caption_data['img_id'].append(image['id'])
        for i, ann in enumerate(anns):
            if i <= 4:
                caption_data['caption' + str(i)].append(ann['caption'])
        if i < 4:
            for j in range(i + 1, 5):
                caption_data['caption' + str(j)].append(None)
        
    return caption_data

def convert_dict_to_df(caption_data, csv_path):
    df = pd.DataFrame(caption_data, columns=['img_id', 'caption0', 'caption1', 'caption2', 'caption3', 'caption4'])
    df.to_csv('./captions.csv', index=False) # replace with csv path


image_info = fetch_images()
image_caption_data = download_images(image_info)
convert_dict_to_df(image_caption_data, './captions.csv')

