import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Common.pnid_xml import xml_reader
from Visualize.image_drawing import draw_bbox_from_bbox_list

# XML 데이터 검증을 위한 가시화 코드
xml_dir = "C:/Users/DongwonJeong/Desktop/HyundaiPNID/Data/2023/TrainingData_SYMBOL/SymbolXML"
xml_dir2 = "C:/Users/DongwonJeong/Desktop/HyundaiPNID/Data/2023/TextXML"
drawing_img_dir = "C:/Users/DongwonJeong/Desktop/HyundaiPNID/Data/2023/JPG"
output_img_dir = "C:/Users/DongwonJeong/Desktop/HyundaiPNID/Data/2023/VisualizedJPG"

# xml_dir = "D:/Test_Models/PNID/HyundaiEng/210518_Data/Symbol_XML"
# drawing_img_dir = "D:/Test_Models/PNID/HyundaiEng/210518_Data/Drawing/JPG"
# is_text_xml = False

xml_filenames = os.listdir(xml_dir)

xml_info = []
entire_objects = []

for xml_filename in tqdm(xml_filenames):
    # print(xml_filename)
    name_only = xml_filename.split(".")[0]
    drawing_path = os.path.join(drawing_img_dir, name_only + ".jpg")

    xml_path = os.path.join(xml_dir, name_only + ".xml")
    xml_reader_obj = xml_reader(xml_path)
    filename, width, height, depth, object_list, error_list = xml_reader_obj.getInfo()

    xml2_path = os.path.join(xml_dir2, name_only + ".xml")
    xml2_reader_obj = xml_reader(xml2_path)
    filename2, width2, height2, depth2, object_list2, error_list2 = xml2_reader_obj.getInfo()

    object_list.extend(object_list2)

    symbol_list = [x for x in object_list if x[0] != 'text']
    text_list = [['text', 'text'] + x[2:] for x in object_list if x[0] == 'text']

    bbox = symbol_list + text_list

    xml_info.append({
        'filename': name_only,
        'symbols': len(symbol_list), 
        'texts': len(text_list) 
        })

    entire_objects.extend(bbox)

unique_labels = set([x[1] for x in entire_objects])
unique_labels.add('text')
occurences = {}

for unique_label in tqdm(unique_labels):
    current_labels = [x[1] for x in entire_objects if x[1] == unique_label]
    occurences[unique_label] = len(current_labels)
    # current_bboxes = [x[1:] for x in entire_objects if x[0] == unique_label]
    # current_bboxes_array = np.array(current_bboxes)
    # mean_diagonal_lengths[unique_label] = np.mean(np.sqrt(current_bboxes_array[:,2] ** 2 + current_bboxes_array[:,3] ** 2))

print(f"Number of symbol instances in entire dataset: {len(entire_objects)}")
print(f"Number of unique symbol labels in entire dataset: {len(unique_labels)}")
# entire_bboxes_array = np.array([x[1:] for x in entire_objects])
# sorted_by_size = sorted(mean_diagonal_lengths.items(), key=lambda item: item[1])

with open("symbol_statistics.csv",'w') as f:
    col_info = 'class' + ',' + 'occurences\n'
    f.write(col_info)
    for cls, occr in occurences.items():
        data = str(cls) + "," + str(occr) + "\n"
        f.write(data)

with open("xml_statistics.csv", "w") as f:
    col_info = 'filename,symbols,texts,total\n'
    f.write(col_info)
    for xml in xml_info:
        data = str(xml['filename']) + "," + str(xml['symbols']) + "," + str(xml['texts']) + "," + str(xml['symbols']+xml['texts']) + "\n"
        f.write(data)

f.close()