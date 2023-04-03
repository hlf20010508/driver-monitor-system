import os
import torch
import cv2
from PIL import Image
import numpy as np
from model.dm_net import DMNet
from module.entity import COLORS, INFINITE, TRANSFORMS, BODY_HEATMAP_DICT, BODY_LIMB_DICT, THREADHOLD
from test_body_class import detect_class

base_model = os.environ.get('base_model', 'mnv3s')
video_path = os.environ['video_path']
model_load_path = os.environ.get('model_load_path', 'model.pth')

heatmap_num = len(BODY_HEATMAP_DICT)
paf_num = len(BODY_LIMB_DICT) * 2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = DMNet(
    base_model=base_model,
    heatmap_num=heatmap_num,
    paf_num=paf_num
)
model_dict = torch.load(model_load_path, map_location=device)
model.load_state_dict(model_dict)
model = model.to(device)
model.eval()

def load_img(img_path):
    ori_img = cv2.imread(img_path)
    ori_width = ori_img.shape[1]
    ori_height = ori_img.shape[0]
    image = Image.fromarray(cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB))
    image = TRANSFORMS(image)
    image = torch.unsqueeze(image, 0)
    return ori_img, image, ori_width, ori_height

def point_list_gen(heatmaps, ori_width, ori_height):
    point_list = [[] for i in range(heatmap_num)]
    scale_x = ori_width / heatmaps.shape[2]
    scale_y = ori_height / heatmaps.shape[1]
    for heatmap_index in range(heatmap_num):
        heatmap = heatmaps[heatmap_index]
        max_point = np.max(heatmap)
        if max_point > 0.2:
            point = np.where(heatmap == max_point)
            point = (int(point[1][0] * scale_x), int(point[0][0] * scale_y))
            point_list[heatmap_index].append(point)
    return point_list

# 计算两点间距离
def calc_distance(point_list):
    return ((point_list[0][0] - point_list[1][0])**2 + (point_list[0][1] - point_list[1][1])**2)**0.5

cap = cv2.VideoCapture(video_path)
count = 0
while(cap.isOpened()):
    ret, ori_img = cap.read() 
    if count % 5 == 0:
        ori_width = ori_img.shape[1]
        ori_height = ori_img.shape[0]
        image = Image.fromarray(cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB))
        image = TRANSFORMS(image)
        image = torch.unsqueeze(image, 0)

        image = image.to(device)

        heatmaps, pafs = model(image)

        heatmaps = heatmaps.to('cpu').detach().numpy()

        point_list = point_list_gen(heatmaps, ori_width, ori_height)

    for points in point_list:
        for point in points:
            cv2.circle(ori_img, point, 4, COLORS[0], -1)

    # left_shoulder = point_list[0]
    # left_elbow = point_list[1]
    # left_wrist = point_list[2]
    # right_shoulder = point_list[3]
    # right_elbow = point_list[4]
    # right_wrist = point_list[5]
    # head = point_list[6]
    # wheel = point_list[7]

    for limb in BODY_LIMB_DICT:
        start = limb[0]
        end = limb[1]
        if len(point_list[start]) > 0 and len(point_list[end]) > 0:
            cv2.line(ori_img, point_list[start][0], point_list[end][0], COLORS[1])
        # if len(left_shoulder) > 0 and len(head) > 0:
        #     cv2.line(ori_img, left_shoulder[0], head[0], COLORS[1])
        # if len(left_shoulder) > 0 and len(left_elbow) > 0:
        #     cv2.line(ori_img, left_shoulder[0], left_elbow[0], COLORS[1])
        # if len(left_elbow) > 0 and len(left_wrist) > 0:
        #     cv2.line(ori_img, left_elbow[0], left_wrist[0], COLORS[1])
        # if len(right_shoulder) > 0 and len(head) > 0:
        #     cv2.line(ori_img, right_shoulder[0], head[0], COLORS[2])
        # if len(right_shoulder) > 0 and len(right_elbow) > 0:
        #     cv2.line(ori_img, right_shoulder[0], right_elbow[0], COLORS[2])
        # if len(right_elbow) > 0 and len(right_wrist) > 0:
        #     cv2.line(ori_img, right_elbow[0], right_wrist[0], COLORS[2])
        # if len(wheel) > 0 and len(left_wrist) > 0:
        #     cv2.line(ori_img, wheel[0], left_wrist[0], COLORS[3])
        # if len(wheel) > 0 and len(right_wrist) > 0:
        #     cv2.line(ori_img, wheel[0], right_wrist[0], COLORS[3])
    
    if len(point_list[-1]) > 0: # wheel
        left_dis = INFINITE
        right_dis = INFINITE
        if len(point_list[2]) > 0: # left wrist
            left_dis = calc_distance([point_list[2][0], point_list[-1][0]])
        if len(point_list[5]) > 0: # right wrist
            right_dis = calc_distance([point_list[5][0], point_list[-1][0]])
        if left_dis < THREADHOLD and right_dis < THREADHOLD:
            cv2.putText(ori_img, 'safe', (0,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, COLORS[2])
        else:
            label = detect_class(ori_img)
            cv2.putText(ori_img, label, (0,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, COLORS[3])

    cv2.imshow('monitor', ori_img) 
    k = cv2.waitKey(1) 
    count += 1
    #q键退出
    if (k & 0xff == ord('q')): 
        break

cap.release() 
cv2.destroyAllWindows()
