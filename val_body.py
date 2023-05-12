import os
import torch
import numpy as np
from module.load_data import Train_Dataset, stride
from model.dm_net import DMNet
from module.entity import BODY_HEATMAP_DICT, BODY_LIMB_DICT

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


annotation_path = os.environ['annotation_path']
img_root_path = os.environ['img_root_path']
model_load_path = os.environ['model_load_path']

batch_size = int(os.environ.get('batch_size', 1))
num_epochs = int(os.environ.get('num_epochs', 5))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

heatmap_num = len(BODY_HEATMAP_DICT)
paf_num = len(BODY_LIMB_DICT) * 2

dataset = Train_Dataset(
    heatmap_num=heatmap_num,
    paf_num=paf_num,
    heatmap_dict=BODY_HEATMAP_DICT,
    limb_dict=BODY_LIMB_DICT,
    annotation_path=annotation_path,
    img_root_path=img_root_path,
)

train_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

model = DMNet(
    heatmap_num=heatmap_num,
    paf_num=paf_num
)
model_dict = torch.load(model_load_path, map_location=device)
model.load_state_dict(model_dict)
model = model.to(device)

model.eval()

ave_acc = 0
for epoch in range(num_epochs):
    total_acc = 0
    for batch in train_loader:
        images, labels = batch

        heatmaps_target = labels['heatmaps_target']
        heatmap_masks = labels['heatmap_masks']
        pafs_target = labels['pafs_target']
        paf_masks = labels['paf_masks']
        ori_points = labels['heatmap_points']
        ori_width = labels['img_width']
        ori_height = labels['img_height']

        ori_points = [(int(points[0] * stride), int(points[1] * stride)) if points else (-1, -1) for points in ori_points]

        images = images.to(device)
        heatmaps_target = heatmaps_target.to(device)
        heatmap_masks = heatmap_masks.to(device)
        pafs_target = pafs_target.to(device)
        paf_masks = paf_masks.to(device)

        heatmaps_pre, pafs_pre = model(images)

        heatmaps_pre = heatmaps_pre.to('cpu').detach().numpy()

        point_list = point_list_gen(heatmaps_pre, ori_width, ori_height)
        
        for i in range(len(BODY_HEATMAP_DICT)):
            total_acc += ((point_list[i][0] - ori_points[i][0]) ** 2 + (point_list[i][1] - ori_points[i][1]) ** 2) ** 0.5

    total_acc /= len(BODY_HEATMAP_DICT) * len(train_loader)
    ave_acc += total_acc
    print('%d/%d: %.4f'%(epoch + 1, num_epochs, total_acc))

ave_acc /= num_epochs
print('ave: %.2f'%ave_acc)
