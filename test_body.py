import os
import torch
import cv2
from PIL import Image
import numpy as np
from model.dm_net import DMNet
from module.entity import COLORS, TRANSFORMS, BODY_HEATMAP_DICT, BODY_LIMB_DICT, TIME_LEN
from test_body_class import detect_class
import cv2
import time

video_path = os.environ['video_path']
model_load_path = os.environ.get('model_load_path', 'model.pth')
time_len = int(os.environ.get('time_len', TIME_LEN))

heatmap_num = len(BODY_HEATMAP_DICT)
paf_num = len(BODY_LIMB_DICT) * 2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = DMNet(
    heatmap_num=heatmap_num,
    paf_num=paf_num
)
model_dict = torch.load(model_load_path, map_location=device)
model.load_state_dict(model_dict)
model = model.to(device)
model.eval()

def point_list_gen(heatmaps, ori_width, ori_height):
    point_list = [(-1, -1) for i in range(heatmap_num)]
    scale_x = ori_width / heatmaps.shape[2]
    scale_y = ori_height / heatmaps.shape[1]
    for heatmap_index in range(heatmap_num):
        heatmap = heatmaps[heatmap_index]
        max_point = np.max(heatmap)
        if max_point > 0.2:
            point = np.where(heatmap == max_point)
            point = (int(point[1][0] * scale_x), int(point[0][0] * scale_y))
            point_list[heatmap_index] = point
    return point_list

cap = cv2.VideoCapture(video_path)
count = 0
time_point_list = []
t = time.time()
max_fps = 0
min_fps = 10000
ave_fps = []
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
        # pafs = pafs.to('cpu').detach().numpy()

        point_list = point_list_gen(heatmaps, ori_width, ori_height)
        if len(time_point_list) == time_len:
            time_point_list = time_point_list[1:]
        time_point_list.append(point_list)

    for point in point_list:
        cv2.circle(ori_img, point, 4, COLORS[0], -1)

    for limb in BODY_LIMB_DICT:
        start = limb[0]
        end = limb[1]
        if point_list[start] != (-1, -1) and point_list[end] != (-1, -1):
            cv2.line(ori_img, point_list[start], point_list[end], COLORS[1])
    
    if len(time_point_list) == time_len:
        label = detect_class(np.array(time_point_list, dtype=np.float32))
        cv2.putText(ori_img, label, (0,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, COLORS[3])

    cv2.imshow('monitor', ori_img) 
    # k = cv2.waitKey()
    k = cv2.waitKey(1) 
    count += 1
    if count % 60 == 0:
        fps = 60 / (time.time() - t)
        if fps > max_fps:
            max_fps = fps
        if fps < min_fps:
            min_fps = fps
        ave_fps.append(fps)
        t = time.time()
        _ave_fps = sum(ave_fps) / len(ave_fps)
        print('max fps: %.3f'%max_fps)
        print('min fps: %.3f'%min_fps)
        print('ave fps: %.3f'%_ave_fps)
    #q键退出
    if (k & 0xff == ord('q')): 
        break

cap.release() 
cv2.destroyAllWindows()
