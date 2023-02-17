import os
import cv2
import json
from tqdm import tqdm
from Misc.rotated_box import rotated_box
from Common.pnid_xml import xml_reader

# XML 데이터 검증을 위한 가시화 코드

#TODO: symbol & text xml 통합 후 xml2 관련 코드는 삭제 예정

xml_dir = "C:/Users/DongwonJeong/Desktop/HyundaiPNID/Data/2023/TrainingData_SYMBOL/SymbolXML"
xml_dir2 = "C:/Users/DongwonJeong/Desktop/HyundaiPNID/Data/2023/TextXML"
drawing_img_dir = "C:/Users/DongwonJeong/Desktop/HyundaiPNID/Data/2023/JPG"
output_img_dir = "C:/Users/DongwonJeong/Desktop/HyundaiPNID/Data/2023/VisualizedJPG_RangeCheck"
start_xml_name = ""

xml_filenames = sorted(os.listdir(xml_dir))

error_dict = {}

for xml_filename in tqdm(xml_filenames):
    if xml_filename < start_xml_name:
        continue
    name_only = xml_filename.split(".")[0]
    drawing_path = os.path.join(drawing_img_dir, name_only + ".jpg")

    xml_path = os.path.join(xml_dir, name_only + ".xml")
    xml_reader_obj = xml_reader(xml_path)
    filename, width, height, depth, object_list, error_list = xml_reader_obj.getInfo()

    xml2_path = os.path.join(xml_dir2, name_only + ".xml")
    xml2_reader_obj = xml_reader(xml2_path)
    filename2, width2, height2, depth2, object_list2, error_list2 = xml2_reader_obj.getInfo()

    total_error_list = error_list + error_list2 

    if len(total_error_list) != 0:
        error_dict[filename] = total_error_list

    img = cv2.imread(drawing_path)
    
    for obj in object_list:
        rotated_box(vertices=obj[2:], text=obj[1]).draw(img)

    for obj in object_list2:
        rotated_box(vertices=obj[2:], text=obj[1]).draw(img)

    out_path = os.path.join(output_img_dir, f"{name_only}" + ".jpg")

    cv2.imwrite(out_path, img)

with open(f'{output_img_dir}/error_info.json', 'w') as f: 
    json.dump(error_dict, f, indent=4)