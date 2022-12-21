from PIL import Image
import os
from tqdm import tqdm
import splitfolders
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser(
        description='preprocess pics')
    parser.add_argument('--img-path', required=True, help='images file path')
    parser.add_argument('--ann-path',required=True, help='annotations file path')
    parser.add_argument('--resized-img-path',required=True, help='resized image file path')
    parser.add_argument('--resized-ann-path', required=True, help='resized annotation file path')
    parser.add_argument('--split-img-path', required=True, help='split image file path')
    parser.add_argument('--split-ann-path', required=True, help='split annotation file path')
    parser.add_argument('--seed', default=4832, help='choose your seed!')
    parser.add_argument('--split-ratio', nargs='+', type=float, help='train, test, val ratios')
    args = parser.parse_args()
    return args


def validate_size(img_path, ann_path):
    img_dir = os.listdir(img_path)
    ann_dir = os.listdir(ann_path)
    for item in img_dir:
        if os.path.isfile(img_path+item):
            im = Image.open(img_path+item)
            im_width, im_height = im.size
            ann = Image.open(ann_path+item)
            ann_width, ann_height = ann.size
            if (ann_width != im_width or ann_height != im_height):
                return False
    return True

def resize_images(path, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    dirs = os.listdir(path)
    for item in dirs:
        input_file = path + '/' + item
        output = output_path + '/' + item
        if os.path.isfile(input_file):
            im = Image.open(input_file)
            width, height = im.size
            new_height = int((600 * height)/width)
            resize_factor = (600, new_height)
            imResize = im.resize(resize_factor, Image.ANTIALIAS)
            width, height = imResize.size
            imResize.save(output)
            
def split_train_val(all_images, all_annotations, output_dir_img, output_dir_ann, seed, split=(.8, 0.1,0.1)):
    folder_above_img = os.path.dirname(all_images)
    img_destination = os.path.join(folder_above_img, 'nested_img')
    if not os.path.exists(img_destination):
        os.mkdir(img_destination)
    shutil.move(all_images, img_destination) 
    
    folder_above_ann = os.path.dirname(all_annotations)
    ann_destination = os.path.join(folder_above_ann, 'nested_ann')
    if not os.path.exists(ann_destination):
        os.mkdir(ann_destination)
    shutil.move(all_annotations, ann_destination) 
             
    if not os.path.exists(output_dir_img):
        os.mkdir(output_dir_img)
    if not os.path.exists(output_dir_ann):
        os.mkdir(output_dir_ann)
        
    dirs = [output_dir_img, output_dir_ann]
    subdirs = ['train', 'test', 'val']
    
    for d in dirs:
        for s in subdirs:
            output_path = os.path.join(d, s)
            if not os.path.exists(output_path):
                os.mkdir(output_path)
    
    splitfolders.ratio(img_destination, output=output_dir_img, seed=seed, ratio=split) 
    splitfolders.ratio(ann_destination, output=output_dir_ann, seed=seed, ratio=split)
    
 
    for s in subdirs:
        dest_path = os.path.join(output_dir_img, s)
        inner_path = os.path.join(dest_path, 'resized_img')
        for img in os.listdir(inner_path):
            img_path = os.path.join(inner_path, img)
            shutil.copy(img_path, dest_path)
        shutil.rmtree(inner_path)
    for s in subdirs:
        dest_path = os.path.join(output_dir_ann, s)
        inner_path = os.path.join(dest_path, 'resized_ann')
        for ann in os.listdir(inner_path):
            ann_path = os.path.join(inner_path, ann)
            shutil.copy(ann_path, dest_path)
        shutil.rmtree(inner_path)
        
    

def change_jpg(path, output):
    dirs = os.listdir(path)
    for item in dirs:
        input_file = path+'/'+item
        if os.path.isfile(input_file):
            im = Image.open(input_file)
            im = im.convert('RGB')
            filename = item.split('.')[0]
            full_output = output+'/'+filename+'.jpg'
            im.save(full_output)
            os.remove(input_file)

if __name__ == '__main__':
    args = parse_args()
    img_path = args.img_path
    ann_path = args.ann_path
    img_resize_path = args.resized_img_path
    ann_resize_path = args.resized_ann_path
    split_img_out = args.split_img_path
    split_ann_out = args.split_ann_path
    seed = args.seed
    if args.split_ratio != None:
        split = tuple(args.split_ratio)
    else:
        split = (.8, 0.1,0.1)
             
    
    assert validate_size(img_path, ann_path)
    resize_images(img_path, img_resize_path)
    resize_images(ann_path, ann_resize_path)
    change_jpg(img_resize_path, img_resize_path)
    split_train_val(img_resize_path, ann_resize_path, split_img_out, split_ann_out, seed, split)
             
    
    
    
    
    
    
    
    
    




