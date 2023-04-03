import os
import torch
import cv2
from PIL import Image
import numpy as np
from model.dm_net import DMNet
from module.entity import COLORS, TRANSFORMS, FACE_HEATMAP_DICT, FACE_LIMB_DICT, FACE_LIMB_DICT_NEW

base_model = os.environ.get('base_model', 'mnv3s')
video_path = os.environ['video_path']
model_load_path = os.environ.get('model_load_path', 'model.pth')

heatmap_num = len(FACE_HEATMAP_DICT)
paf_num = len(FACE_LIMB_DICT) * 2

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

def point_list_gen(heatmaps, ori_width, ori_height):
    point_list = [[] for i in range(heatmap_num)]
    scale_x = ori_width / heatmaps.shape[2]
    scale_y = ori_height / heatmaps.shape[1]
    for heatmap_index in range(heatmap_num):
        heatmap = heatmaps[heatmap_index]
        max_point = np.max(heatmap)
        if max_point > 0.8:
            point = np.where(heatmap == max_point)
            point = (int(point[1][0] * scale_x), int(point[0][0] * scale_y))
            point_list[heatmap_index].append(point)
    return point_list

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
    
    for limb in FACE_LIMB_DICT_NEW:
        start = limb[0]
        end = limb[1]
        if len(point_list[start]) > 0 and len(point_list[end]) > 0:
            cv2.line(ori_img, point_list[start][0], point_list[end][0], COLORS[1])

    cv2.imshow('monitor', ori_img) 
    k = cv2.waitKey(1) 
    count += 1
    #q键退出
    if (k & 0xff == ord('q')): 
        break

cap.release() 
cv2.destroyAllWindows()
