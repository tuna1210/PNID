import cv2
import os
import numpy as np
from Common.pnid_xml import xml_reader
from tqdm import tqdm

def generate_segmented_data(xml_list, drawing_dir, drawing_segment_dir, segment_params, symbol_dict,
                            drawing_resize_scale, prefix):
    """ 폴더 내 원본 이미지 도면들을 분할하고 분할 정보를 리스트로 저장

    Arguments:
        xml_list (list): xml 파일 리스트
        drawing_dir (string): 원본 도면 이미지가 존재하는 폴더
        drawing_segment_dir (string): 분할된 이미지를 저장할 폴더
        segment_params (list): 분할 파라메터 [가로 크기, 세로 크기, 가로 stride, 세로 stride]
        symbol_dict (dict): symbol 이름을 key로, id를 value로 갖는 dict
        include_text_as_class (bool): text 데이터를 class로 추가할 것인지
        drawing_resize_scale (float): 전체 도면 조정 스케일
        prefix (string): train/val/test 중 하나. 이미지 저장 폴더명 생성에 필요

    Return:
        xml에 있는 전체 도면에서 분할된 도면의 전체 정보 [sub_img_name, type, class(string), x1, y1, x2, y2, x3, y3, x4, y4]
    """
    entire_segmented_info = []
    no_sym_list = set()
    
    # print(f'Start segmentation of {prefix} set images & XMLs.')
    for xmlPath in xml_list:
        fname, ext = os.path.splitext(xmlPath)
        if ext.lower() != ".xml":
            continue

        xmlReader = xml_reader(xmlPath)
        img_filename, width, height, depth, object_list = xmlReader.getInfo()

        for i in range(len(object_list)):
            name = object_list[i][1] if object_list[i][0] != 'text' else 'text'
            # try:
            if name not in symbol_dict:
                no_sym_list.add((img_filename, name))
            #     object_list[i][0] = symbol_dict[name]
            # except:

        img_file_path = os.path.join(drawing_dir, img_filename)

        segmented_objects_info = segment_images(img_file_path, drawing_segment_dir, object_list,
                                                    symbol_dict, segment_params, drawing_resize_scale, prefix)

        entire_segmented_info.extend(segmented_objects_info)

    if len(no_sym_list) != 0:
        with open(os.path.join(drawing_segment_dir, f'no_sym_list_{prefix}.txt'), 'w') as nosym:
            nosym.write(str(no_sym_list))

    return entire_segmented_info

def segment_images(img_path, drawing_segment_dir, objects, symbol_dict, segment_params, drawing_resize_scale, prefix, ignore_exist=True):
    """ 각 도면에 대해 분할 도면을 생성하고 분할된 파일로 저장

    Arguments:
        img_path (string): 원본 이미지 파일의 경로
        drawing_segment_dir (string): 출력 분할 이미지 경로
        objects (list): 원본 이미지의 object list [type, class(string), x1, y1, x2, y2, x3, y3, x4, y4]
        symbol_dict (dict): symbol 이름을 key로, id를 value로 갖는 dict
        segment_params (list): 분할 파라메터 [가로 크기, 세로 크기, 가로 stride, 세로 stride]
        drawing_resize_scale (float): 도면 크기 조절 (분할 전)
        prefix (string): train/val/test 중 하나. 이미지 저장 폴더명 생성에 필요
        ignore_exist (bool): 분할 이미지가 이미 존재하면 symbol 정보만 취하고 이미지 저장은 수행하지 않을지에대한 여부 (True: overwrite)

    Return:
        seg_obj_info : 한 장의 도면에서 생성된 스케일이 적용된 분할 도면의 전체 정보 [sub_img_name, type, class(string), x1, y1, x2, y2, x3, y3, x4, y4]
    """
    if os.path.exists(os.path.join(drawing_segment_dir, prefix)) == False:
        os.mkdir(os.path.join(drawing_segment_dir, prefix))

    if os.path.exists(os.path.join(drawing_segment_dir, prefix, 'images')) == False:
        os.mkdir(os.path.join(drawing_segment_dir, prefix, 'images'))

    out_dir = os.path.join(drawing_segment_dir, prefix, 'images')

    width_size = segment_params[0]
    height_size = segment_params[1]
    width_stride = segment_params[2]
    height_stride = segment_params[3]

    bbox_array = np.zeros((len(objects),8))
    for ind in range(len(objects)):
        objects[ind] = [objects[ind][0],
                        objects[ind][1],
                        int(objects[ind][2]*drawing_resize_scale),
                        int(objects[ind][3]*drawing_resize_scale),
                        int(objects[ind][4]*drawing_resize_scale),
                        int(objects[ind][5]*drawing_resize_scale),
                        int(objects[ind][6]*drawing_resize_scale),
                        int(objects[ind][7]*drawing_resize_scale),
                        int(objects[ind][8]*drawing_resize_scale),
                        int(objects[ind][9]*drawing_resize_scale),
                        ]
        
        bbox_object = objects[ind]
        bbox_array[ind, :] = np.array([bbox_object[2], bbox_object[3], bbox_object[4], bbox_object[5],
                                       bbox_object[6], bbox_object[7], bbox_object[8], bbox_object[9]])

    # if txt_object_list is not None:
    #     txt_boox_array = np.zeros((len(txt_object_list), 4))
    #     for ind in range(len(txt_object_list)):
    #         txt_object_list[ind] = [txt_object_list[ind][0],
    #                                 int(txt_object_list[ind][1] * drawing_resize_scale),
    #                                 int(txt_object_list[ind][2] * drawing_resize_scale),
    #                                 int(txt_object_list[ind][3] * drawing_resize_scale),
    #                                 int(txt_object_list[ind][4] * drawing_resize_scale),
    #                                 int(txt_object_list[ind][5])]
    #         txt_bbox_object = txt_object_list[ind]
    #         txt_boox_array[ind, :] = np.array([txt_bbox_object[1], txt_bbox_object[2], txt_bbox_object[3], txt_bbox_object[4]])

    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=(0,0), fx=drawing_resize_scale, fy=drawing_resize_scale, interpolation=cv2.INTER_LINEAR)

    seg_obj_info = []
    start_height = 0
    h_index = 0

    while start_height < img.shape[0]: # 1픽셀때문에 이미지를 하나 더 만들 필요는 없음
        start_width = 0
        w_index = 0

        while start_width < img.shape[1]:
            x1_lb_in = start_width  < bbox_array[:, 0]
            y1_lb_in = start_height < bbox_array[:, 1]
            x2_lb_in = start_width  < bbox_array[:, 2]
            y2_lb_in = start_height < bbox_array[:, 3]
            x3_lb_in = start_width  < bbox_array[:, 4]
            y3_lb_in = start_height < bbox_array[:, 5]
            x4_lb_in = start_width  < bbox_array[:, 6]
            y4_lb_in = start_height < bbox_array[:, 7]

            x1_ub_in = bbox_array[:, 0] < start_width + width_size
            y1_ub_in = bbox_array[:, 1] < start_height + height_size
            x2_ub_in = bbox_array[:, 2] < start_width + width_size
            y2_ub_in = bbox_array[:, 3] < start_height + height_size
            x3_ub_in = bbox_array[:, 4] < start_width + width_size
            y3_ub_in = bbox_array[:, 5] < start_height + height_size
            x4_ub_in = bbox_array[:, 6] < start_width + width_size
            y4_ub_in = bbox_array[:, 7] < start_height + height_size
            
            is_bbox_in = x1_lb_in & y1_lb_in & x2_lb_in & y2_lb_in & x3_lb_in & y3_lb_in & x4_lb_in & y4_lb_in & \
                        x1_ub_in & y1_ub_in & x2_ub_in & y2_ub_in & x3_ub_in & y3_ub_in & x4_ub_in & y4_ub_in
            in_bbox_ind = [i for i, val in enumerate(is_bbox_in) if val == True]

            if len(in_bbox_ind) == 0 and prefix == "train":
                start_width += width_stride
                w_index += 1
                continue

            filename, _ = os.path.splitext(os.path.basename(img_path))
            sub_img_filename = f"{filename}_{h_index}_{w_index}.png"

            if not os.path.isfile(os.path.join(out_dir, sub_img_filename)) and ignore_exist:
                sub_img = np.ones((height_size, width_size, 3)) * 255
                if start_width+width_size > img.shape[1] and start_height+height_size > img.shape[0]:
                    sub_img[0:img.shape[0] - (start_height + height_size),
                            0:img.shape[1] - (start_width + width_size), :] = img[start_height:start_height + height_size,
                                                                                start_width:start_width + width_size, :]
                elif start_width+width_size > img.shape[1]:
                    sub_img[:, 0:img.shape[1]-(start_width+width_size), :] = img[start_height:start_height+height_size, start_width:img.shape[1],:]
                elif start_height+height_size > img.shape[0]:
                    sub_img[0:img.shape[0] - (start_height + height_size), : , :] = img[start_height:img.shape[0], start_width:start_width+width_size, :]
                else:
                    sub_img = img[start_height:start_height+height_size, start_width:start_width+width_size, :]
                    
                cv2.imwrite(os.path.join(out_dir, sub_img_filename), sub_img)

            for i in in_bbox_ind:
                seg_obj_info.append([sub_img_filename, objects[i][0], objects[i][1], 
                                     objects[i][2] - start_width, objects[i][3] - start_height, 
                                     objects[i][4] - start_width, objects[i][5] - start_height,
                                     objects[i][6] - start_width, objects[i][7] - start_height,
                                     objects[i][8] - start_width, objects[i][9] - start_height
                                     ])

            # TODO: Text의 경우 text string을 같이 넣도록 추가 구현?
            # for i in txt_in_bbox_ind:
            #     if include_text_orientation_as_class == True:
            #         if txt_object_list[i][5] == 0:
            #             seg_obj_info.append([sub_img_filename, symbol_dict["text"], txt_object_list[i][1] - start_width,
            #                                  txt_object_list[i][2] - start_height,
            #                                  txt_object_list[i][3] - start_width, txt_object_list[i][4] - start_height])
            #         if txt_object_list[i][5] == 90:
            #             seg_obj_info.append(
            #                 [sub_img_filename, symbol_dict["text_rotated"], txt_object_list[i][1] - start_width,
            #                  txt_object_list[i][2] - start_height,
            #                  txt_object_list[i][3] - start_width, txt_object_list[i][4] - start_height])
            #         if txt_object_list[i][5] == 45:
            #             seg_obj_info.append(
            #                 [sub_img_filename, symbol_dict["text_rotated_45"], txt_object_list[i][1] - start_width,
            #                  txt_object_list[i][2] - start_height,
            #                  txt_object_list[i][3] - start_width, txt_object_list[i][4] - start_height])
            #     else:
            #         seg_obj_info.append([sub_img_filename, symbol_dict["text"], txt_object_list[i][1] - start_width, txt_object_list[i][2] - start_height,
            #                              txt_object_list[i][3] - start_width, txt_object_list[i][4] - start_height])

            # if prefix != "train" and len(seg_obj_info) == 0: # test/val은 박스가 없어도 이미지 인덱스를 만들기 위해 추가
            if len(in_bbox_ind) == 0: # test/val은 박스가 없어도 이미지 인덱스를 만들기 위해 추가
                seg_obj_info.append([sub_img_filename, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0])

            start_width += width_stride
            w_index += 1

        start_height += height_stride
        h_index += 1

            # # Debug visualize
            # fig, ax = plt.subplots(1)
            # ax.imshow(sub_img)
            # for i in in_bbox_ind:
            #     symbolxmin = objects[i][1] - start_width
            #     symbolxmax = objects[i][3] - start_width
            #     symbolymin = objects[i][2] - start_height
            #     symbolymax = objects[i][4] - start_height
            #     rect = patches.Rectangle((symbolxmin, symbolymin),
            #                              symbolxmax - symbolxmin,
            #                              symbolymax - symbolymin,
            #                              linewidth=1, edgecolor='r', facecolor='none')
            # for i in txt_in_bbox_ind:
            #     symbolxmin = txt_object_list[i][1] - start_width
            #     symbolxmax = txt_object_list[i][3] - start_width
            #     symbolymin = txt_object_list[i][2] - start_height
            #     symbolymax = txt_object_list[i][4] - start_height
            #     rect = patches.Rectangle((symbolxmin, symbolymin),
            #                              symbolxmax - symbolxmin,
            #                              symbolymax - symbolymin,
            #                              linewidth=1, edgecolor='r', facecolor='none')
            #     ax.add_patch(rect)
            # plt.show()

    return seg_obj_info