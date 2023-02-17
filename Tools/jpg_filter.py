import os
import shutil
from tqdm import tqdm

# filter.txt 파일에 적힌 이름의 도면만 분리하여 저장
filter_file = "C:/Users/DongwonJeong/Desktop/HyundaiPNID/Data/2023/filter.txt"
drawing_img_dir = "C:/Users/DongwonJeong/Desktop/HyundaiPNID/Data/2023/VisualizedJPG"
output_img_dir = "C:/Users/DongwonJeong/Desktop/HyundaiPNID/Data/2023/NoSymJPG"

with open(filter_file, 'r') as f:
    raw_names = f.readlines()
    img_names = [f'{name.strip()}.jpg' for name in raw_names]

for img_name in img_names:
    shutil.copyfile(f'{drawing_img_dir}/{img_name}', f'{output_img_dir}/{img_name}')