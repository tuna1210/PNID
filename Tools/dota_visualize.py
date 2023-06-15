import os
import cv2
import json
from multiprocessing import Pool
from tqdm import tqdm
from Misc.rotated_box import rotated_box
from Common.pnid_xml import xml_reader

# DOTA 데이터 검증을 위한 가시화 코드

def parse_annfiles(annfile_path: str) -> list:
    annfile = open(annfile_path, 'r')

    lines = annfile.readlines()

    rboxes = []
    for ind, l in enumerate(lines):
        line = l.split()
        points = line[0:8]
        class_name = line[8]
        difficulty = line[9]

        rboxes.append(rotated_box(vertices=points, text=class_name))
    
    return rboxes
        
root_dir = r"E:\PNID_Data\split_dataset\test"
annfiles_dir = root_dir + r'\annfiles'
images_dir = root_dir + r'\images'
visualized_images_dir = r"E:\PNID_Data\split_dataset\visualized_test"

width = 9933
height = 7016
resize_factor = 0.125
annfile_names = sorted(os.listdir(annfiles_dir))

for ann_file in tqdm(annfile_names):
    name_only = ann_file.split(".")[0]
    drawing_path = os.path.join(images_dir, name_only + ".jpg")

    annfile_path = os.path.join(annfiles_dir, name_only + ".txt")

    rboxes = parse_annfiles(annfile_path)
    
    img = cv2.imread(drawing_path)
    # img = cv2.resize(img, (int(width * resize_factor), int(height * resize_factor)))

    for rbox in rboxes:
        rbox.draw(img)

    out_path = os.path.join(visualized_images_dir, f"{name_only}" + ".jpg")

    cv2.imwrite(out_path, img)