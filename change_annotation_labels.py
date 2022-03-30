from PIL import Image
import os
from tqdm import tqdm

path = '/home/gridsan/kxu/mmsegmentation/data/dataset_mmseg/ann_dir_old_labels/test/'
new_path = '/home/gridsan/kxu/mmsegmentation/data/dataset_mmseg/ann_dir_new/test/'

label_map = {0: 10, 1: 9, 2: -1, 3: -1, 4: 19, 5: -1, 6: -1, 7: -1, 8: 11, 9: 3, 10: -1, 11: 1, 12: 15, 13: 2, 14: 12, 15: -1, 16: -1, 17: 8, 18: -1, 19: -1, 20: 17, 21: 5, 22: -1, 23: -1, 24: -1, 25: -1, 26: -1, 27: -1, 28: -1, 29: -1, 30: -1, 31: -1, 32: 13, 33: -1, 34: 6, 35: -1, 36: -1, 37: -1, 38: -1, 39: -1, 40: -1, 41: -1, 42: -1, 43: -1, 44: -1, 45: -1, 46: 4, 47: -1, 48: -1, 49: -1, 50: -1, 51: -1, 52: -1, 53: -1, 54: -1, 55: -1, 56: -1, 57: -1, 58: -1, 59: -1, 60: -1, 61: -1, 62: -1, 63: -1, 64: 22, 65: -1, 66: -1, 67: -1, 68: -1, 69: -1, 70: -1, 71: -1, 72: -1, 73: -1, 74: -1, 75: -1, 76: -1, 77: -1, 78: -1, 79: -1, 80: -1, 81: -1, 82: -1, 83: -1, 84: -1, 85: -1, 86: -1, 87: -1, 88: -1, 89: 20, 90: -1, 91: -1, 92: -1, 93: 14, 94: -1, 95: -1, 96: -1, 97: -1, 98: -1, 99: -1, 100: -1, 101: -1, 102: -1, 103: -1, 104: -1, 105: -1, 106: -1, 107: -1, 108: -1, 109: 7, 110: -1, 111: -1, 112: -1, 113: -1, 114: -1, 115: -1, 116: -1, 117: -1, 118: -1, 119: -1, 120: -1, 121: -1, 122: -1, 123: 23, 124: 21, 125: -1, 126: 16, 127: 18, 128: -1, 129: -1, 130: -1, 131: -1, 132: -1, 133: -1, 134: 0, 135: -1, 136: -1, 137: -1, 138: -1, 139: -1, 140: -1, 141: -1, 142: -1, 143: -1, 144: -1, 145: -1, 146: -1, 147: -1, 148: -1, 149: -1}

aerial_to_ade20k_label_map = dict()
for key, val in label_map.items():
    if val != -1:
        aerial_to_ade20k_label_map[val] = key
print(aerial_to_ade20k_label_map)
print(len(aerial_to_ade20k_label_map))

dirs = os.listdir(path)
for item in tqdm(dirs):
    if os.path.isfile(path+item):
        im = Image.open(path+item)
        image = im.load()
        width, height = im.size
        new_im = Image.new(mode="L", size=(width, height))
        new_image = new_im.load()
        for x in range(width):
            for y in range(height):
                pixel = image[x, y]
                if pixel > 23:
                    pixel = 23
                new_pixel = aerial_to_ade20k_label_map[pixel]
                new_image[x, y] = new_pixel
        new_im.save(new_path+item)