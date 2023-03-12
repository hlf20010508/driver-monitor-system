import os
import torch
import cv2
from PIL import Image
import numpy as np
import torchvision.transforms as tf
from model.dm_net import DMNet

base_model = os.environ.get('base_model', 'mnv3s')
img_root_path = os.environ['img_root_path']
model_load_path = os.environ.get('model_load_path', 'model.pth')

width = int(os.environ.get('width', 224))
height = int(os.environ.get('height', 224))
batch_size = int(os.environ.get('batch_size', 25))
heatmap_num = int(os.environ.get('heatmap_num', 6))
paf_num = int(os.environ.get('paf_num', 8))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

colors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
    [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
    [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
]

img_path_list = [ os.path.join(img_root_path, i) for i in os.listdir(img_root_path) if not i.startswith('.')]

transforms = tf.Compose([
    tf.Resize((width, height)),
    tf.ToTensor(),
    tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

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
    image = transforms(image)
    print(ori_width, ori_height)
    image = torch.unsqueeze(image, 0)
    return ori_img, image, ori_width, ori_height

def point_list_gen(heatmaps, ori_width, ori_height):
    point_list = [[] for i in range(heatmap_num)]
    scale_x = ori_width / 14
    scale_y = ori_height / 14
    for heatmap_index in range(heatmap_num):
        heatmap = heatmaps[heatmap_index]
        max_point = np.max(heatmap)
        if max_point > 0.2:
            point = np.where(heatmap == max_point)
            point = (int(point[1][0] * scale_x), int(point[0][0] * scale_y))
            point_list[heatmap_index].append(point)
    return point_list

for img_path in img_path_list:
    ori_img , image , ori_width, ori_height= load_img(img_path)

    image = image.to(device)

    heatmaps, pafs = model(image)

    heatmaps = heatmaps.to('cpu').detach().numpy()
    pafs = pafs.to('cpu').detach().numpy()

    point_list = point_list_gen(heatmaps, ori_width, ori_height)

    for heatmap_class in point_list:
        for points in point_list:
            for point in points:
                cv2.circle(ori_img, point, 4, colors[0], -1)
    cv2.imshow('monitor', ori_img)
    cv2.waitKey()
