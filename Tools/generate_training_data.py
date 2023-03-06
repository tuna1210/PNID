import os
import random
import pickle
from Data_Generator.generate_segmented_data import generate_segmented_data
from Common.symbol_io import read_symbol_txt
from Data_Generator.write_annotation import write_coco_annotation, write_dota_annotation
from multiprocessing import Manager
import parmap

# 학습 데이터 생성 코드. 도면을 train/test/val로 나누고, 각 set의 이미지를 분할하여 sub_img들로 만들어 저장함
# 이때, train의 경우 심볼(또는 옵션에 따라 심볼+텍스트)가 존재하지 않는 도면은 저장하지 않음
# 단 test/val 도면의 경우 심볼이 존재하지 않아도 저장함

base_dir = r"E:\PNID_Data\2023_0228"
drawing_dir = base_dir + r"\Drawing"
drawing_segment_dir = base_dir + r"\Drawing_Segment\800_800"
xml_dir = base_dir + r"\XML"

val_drawings = ['26071-200-M6-052-00025', '26071-200-M6-052-00039', '26071-200-M6-052-00046', '26071-200-M6-052-00073',
                '26071-200-M6-052-00091', '26071-200-M6-052-00097', '26071-200-M6-052-00543', '26071-200-M6-052-00904',
                '26071-200-M6-052-02002', '26071-200-M6-062-00013', '26071-200-M6-063-00020', '26071-203-M6-060-00010',
                '26071-203-M6-064-00010', '26071-203-M6-064-00012', '26071-203-M6-079-00011', '26071-203-M6-160-00211',
                '26071-203-M6-167-00009', '26071-203-M6-167-00202', '26071-203-M6-315-00002', '26071-203-M6-315-00006',
                '26071-203-M6-315-00012', '26071-203-M6-320-00015', '26071-203-M6-320-00020', '26071-203-M6-320-00049',
                '26071-203-M6-320-00051', '26071-203-M6-321-00202', '26071-203-M6-321-00502', '26071-203-M6-321-00517',
                '26071-203-M6-321-00522', '26071-203-M6-321-00608', '26071-203-M6-323-30024', '26071-203-M6-325-00005',
                '26071-203-M6-325-00009', '26071-203-M6-331-00002', '26071-203-M6-331-00025', '26071-203-M6-331-00035',
                '26071-203-M6-337-00018', '26071-203-M6-337-00035', '26071-203-M6-337-00042', '26071-300-M6-053-00006',
                '26071-300-M6-053-00009', '26071-300-M6-053-00017', '26071-300-M6-053-00023', '26071-300-M6-053-00271',
                '26071-300-M6-053-00307', '26071-450-M6-082-00010', '26071-450-M6-082-00231', '26071-500-M6-059-00018',
                '26071-500-M6-059-00056', '26071-500-M6-059-00108', '26071-500-M6-059-00235', '26071-550-M6-084-00001',
                '26071-550-M6-084-00015', '26071-550-M6-084-00263', '26071-600-M6-065-00028', '26071-600-M6-065-00042',
                '26071-600-M6-065-00260', '26071-625-M6-169-00231', '26071-675-M6-068-00025', '26071-675-M6-068-00031',
                '26071-675-M6-068-00043', '26071-700-M6-054-00051', '26071-700-M6-054-00055', '26071-700-M6-054-00261',
                '26071-750-M6-085-00221', '26071-750-M6-085-00305', '26071-750-M6-085-00316']
test_drawings = ['26071-200-M6-052-00001', '26071-200-M6-052-00002', '26071-200-M6-052-00003', '26071-200-M6-052-00080',
                '26071-200-M6-052-00090', '26071-200-M6-052-00093', '26071-200-M6-052-00150', '26071-200-M6-052-00539',
                '26071-200-M6-052-00540', '26071-200-M6-052-00544', '26071-200-M6-052-00552', '26071-200-M6-062-00276',
                '26071-200-M6-062-00277', '26071-200-M6-063-00010', '26071-203-M6-060-00001', '26071-203-M6-060-00006',
                '26071-203-M6-064-00271', '26071-203-M6-079-00001', '26071-203-M6-160-00291', '26071-203-M6-320-00008',
                '26071-203-M6-320-00042', '26071-203-M6-320-00044', '26071-203-M6-320-00060', '26071-203-M6-320-00061',
                '26071-203-M6-321-00225', '26071-203-M6-321-00604', '26071-203-M6-325-00012', '26071-203-M6-325-00013',
                '26071-203-M6-332-00001', '26071-203-M6-332-00008', '26071-203-M6-333-00010', '26071-203-M6-337-00007',
                '26071-203-M6-337-00031', '26071-300-M6-053-00008', '26071-300-M6-053-00010', '26071-300-M6-053-00012',
                '26071-300-M6-053-00309', '26071-450-M6-082-00009', '26071-450-M6-082-00031', '26071-450-M6-082-00221',
                '26071-500-M6-059-00011', '26071-500-M6-059-00013', '26071-500-M6-059-00017', '26071-500-M6-059-00040',
                '26071-500-M6-059-00044', '26071-500-M6-059-00071', '26071-500-M6-059-00116', '26071-500-M6-059-00248',
                '26071-550-M6-084-00026', '26071-550-M6-084-00248', '26071-600-M6-065-00021', '26071-600-M6-065-00037',
                '26071-600-M6-065-00231', '26071-600-M6-065-00259', '26071-600-M6-066-00002', '26071-600-M6-066-00003',
                '26071-600-M6-066-00012', '26071-675-M6-068-00034', '26071-675-M6-068-00041', '26071-675-M6-068-10002',
                '26071-700-M6-054-00013', '26071-700-M6-054-00036', '26071-700-M6-054-00246', '26071-750-M6-085-00007',
                '26071-750-M6-085-00012', '26071-750-M6-085-00286', '26071-750-M6-085-00309']
ignore_drawing = []
train_drawings = [x.split(".")[0] for x in os.listdir(xml_dir)
                  if x.split(".")[0] not in test_drawings and
                  x.split(".")[0] not in val_drawings and
                  x.split(".")[0] not in ignore_drawing ]

symbol_txt_path = base_dir + "\SymbolClass_Class.txt"

include_text_as_class = True # Text를 별도의 클래스로 포함할 것인지 {"text"}
# include_text_orientation_as_class = False # 세로 문자열을 또다른 별도의 클래스로 포함할 것인지 {"text_rotated"},

segment_params = [800, 800, 300, 300] # width_size, height_size, width_stride, height_stride
drawing_resize_scale = 0.5

symbol_dict = read_symbol_txt(symbol_txt_path, include_text_as_class)

train_xmls = [os.path.join(xml_dir, f"{x}.xml") for x in train_drawings]
val_xmls = [os.path.join(xml_dir, f"{x}.xml") for x in val_drawings]
test_xmls = [os.path.join(xml_dir, f"{x}.xml") for x in test_drawings]

# # Random Shuffle
# train_ratio = 0.9
#
# random.Random(1).shuffle(xml_paths_without_test)
# train_count = int(len(xml_paths_without_test)*train_ratio)
# train_xmls = xml_paths_without_test[0:train_count]
# val_xmls = xml_paths_without_test[train_count:]

def segment_and_write(prefix):
    ''' train/val/test셋 분할 함수 (병렬처리를 위해 함수 하나로 묶음)
    '''
    if prefix == 'train':
        annotation_data = generate_segmented_data(train_xmls, drawing_dir, drawing_segment_dir, segment_params,
                                                symbol_dict, drawing_resize_scale, prefix)
    if prefix == 'val':
        annotation_data = generate_segmented_data(val_xmls, drawing_dir, drawing_segment_dir, segment_params,
                                                symbol_dict, drawing_resize_scale, prefix)
    if prefix == 'test':
        annotation_data = generate_segmented_data(test_xmls, drawing_dir, drawing_segment_dir, segment_params,
                                                symbol_dict, drawing_resize_scale, prefix)
    write_dota_annotation(drawing_segment_dir, annotation_data, symbol_dict, segment_params, prefix)

if __name__ == '__main__':
    num_cores = 8
    manager = Manager()
    d = manager.dict()
    input_list = ['train', 'val', 'test']
    # segment_and_write('val')
    parmap.map(segment_and_write, input_list, pm_pbar=True, pm_processes=num_cores)

# val_annotation_data = generate_segmented_data(val_xmls, drawing_dir, drawing_segment_dir, segment_params,
#                                               symbol_dict, drawing_resize_scale, "val")
# write_dota_annotation(drawing_segment_dir, val_annotation_data, symbol_dict, segment_params, 'val')

# train_annotation_data = generate_segmented_data(train_xmls, drawing_dir, drawing_segment_dir, segment_params,
#                                                 symbol_dict, drawing_resize_scale, "train")
# write_dota_annotation(drawing_segment_dir, train_annotation_data, symbol_dict, segment_params, 'train')

# test_annotation_data = generate_segmented_data(test_xmls, drawing_dir, drawing_segment_dir, segment_params,
#                                                symbol_dict, drawing_resize_scale, "test")
# write_dota_annotation(drawing_segment_dir, test_annotation_data, symbol_dict, segment_params, 'test')

# val_annotation_data = generate_segmented_data(val_xmls, drawing_dir, drawing_segment_dir, segment_params,
#                                               symbol_dict, include_text_as_class, include_text_orientation_as_class, drawing_resize_scale, "val")
# write_coco_annotation(os.path.join(drawing_segment_dir,"val.json"), val_annotation_data, symbol_dict, segment_params)

# train_annotation_data = generate_segmented_data(train_xmls, drawing_dir, drawing_segment_dir, segment_params,
#                                                 symbol_dict, include_text_as_class, include_text_orientation_as_class, drawing_resize_scale, "train")
# write_coco_annotation(os.path.join(drawing_segment_dir,"train.json"), train_annotation_data, symbol_dict, segment_params)

# test_annotation_data = generate_segmented_data(test_xmls, drawing_dir, drawing_segment_dir, segment_params,
#                                                symbol_dict, include_text_as_class, include_text_orientation_as_class, drawing_resize_scale, "test")
# write_coco_annotation(os.path.join(drawing_segment_dir,"test.json"), test_annotation_data, symbol_dict, segment_params)