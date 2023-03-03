import os
import cv2
import json
from multiprocessing import Pool
from tqdm import tqdm
from Misc.rotated_box import rotated_box
from Common.pnid_xml import xml_reader

# XML 데이터 검증을 위한 가시화 코드

xml_dir = r"E:\PNID_Data\2023_0228\XML"
drawing_img_dir = r"E:\PNID_Data\2023_0228\JPG"
output_img_dir = r"E:\PNID_Data\2023_0228\VisualizedJPG"

start_xml_name = "" # with suffix

xml_filenames = sorted(os.listdir(xml_dir))

error_dict = {}

for xml_filename in tqdm(xml_filenames):
    if xml_filename < start_xml_name:
        continue
    name_only = xml_filename.split(".")[0]
    drawing_path = os.path.join(drawing_img_dir, name_only + ".jpg")

    xml_path = os.path.join(xml_dir, name_only + ".xml")
    xml_reader_obj = xml_reader(xml_path)
    filename, width, height, depth, object_list = xml_reader_obj.getInfo()
    error_list = xml_reader_obj.getErrorInfo()

    if len(error_list) != 0:
        error_dict[filename] = error_list

    img = cv2.imread(drawing_path)
    
    for obj in object_list:
        rotated_box(vertices=obj[2:], text=obj[1]).draw(img)

    out_path = os.path.join(output_img_dir, f"{name_only}" + ".jpg")

    cv2.imwrite(out_path, img)

if len(error_dict) == 0:
    print('There was no error while visualizing.')
else:
    with open(f'{output_img_dir}/error_info.json', 'w') as f: 
        json.dump(error_dict, f, indent=4)