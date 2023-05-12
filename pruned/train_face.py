import os
import torch
from module.load_data import Train_Dataset
from module.loss import Loss_Weighted
from model.dm_net import DMNet

annotation_path = os.environ['annotation_path']
img_root_path = os.environ['img_root_path']
model_save_dir = os.environ.get('model_save_dir', './')

num_epochs = int(os.environ.get('num_epochs', 300))
batch_size = int(os.environ.get('batch_size', 24))
learning_rate = float(os.environ.get('learning_rate', 5e-5))
weight_decay = float(os.environ.get('weight_decay', 5e-4))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

heatmap_dict = {
    'face_outline_0': 0,
    'face_outline_1': 1,
    'face_outline_2': 2,
    'face_outline_3': 3,
    'face_outline_4': 4,
    'face_outline_5': 5,
    'face_outline_6': 6,
    'face_outline_7': 7,
    'face_outline_8': 8,
    'face_outline_9': 9,
    'face_outline_10': 10,
    'face_outline_11': 11,
    'face_outline_12': 12,
    'face_outline_13': 13,
    'face_outline_14': 14,
    'face_outline_15': 15,
    'face_outline_16': 16,
    'right_eyebrow_17': 17,
    'right_eyebrow_18': 18,
    'right_eyebrow_19': 19,
    'right_eyebrow_20': 20,
    'right_eyebrow_21': 21,
    'left_eyebrow_22': 22,
    'left_eyebrow_23': 23,
    'left_eyebrow_24': 24,
    'left_eyebrow_25': 25,
    'left_eyebrow_26': 26,
    'nose_27': 27,
    'nose_28': 28,
    'nose_29': 29,
    'nose_30': 30,
    'nose_bellow_31': 31,
    'nose_bellow_32': 32,
    'nose_bellow_33': 33,
    'nose_bellow_34': 34,
    'nose_bellow_35': 35,
    'right_eye_36': 36,
    'right_eye_37': 37,
    'right_eye_38': 38,
    'right_eye_39': 39,
    'right_eye_40': 40,
    'right_eye_41': 41,
    'left_eye_42': 42,
    'left_eye_43': 43,
    'left_eye_44': 44,
    'left_eye_45': 45,
    'left_eye_46': 46,
    'left_eye_47': 47,
    'lip_outer_48': 48,
    'lip_outer_49': 49,
    'lip_outer_50': 50,
    'lip_outer_51': 51,
    'lip_outer_52': 52,
    'lip_outer_53': 53,
    'lip_outer_54': 54,
    'lip_outer_55': 55,
    'lip_outer_56': 56,
    'lip_outer_57': 57,
    'lip_outer_58': 58,
    'lip_outer_59': 59,
    'lip_inner_60': 60,
    'lip_inner_61': 61,
    'lip_inner_62': 62,
    'lip_inner_63': 63,
    'lip_inner_64': 64,
    'lip_inner_65': 65,
    'lip_inner_66': 66,
    'lip_inner_67': 67,
    'right_eye_center_68': 68,
    'left_eye_center_69': 69
}
limb_dict = [
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], # 輪郭
    [17, 18], [18, 19], [19, 20], [20, 21], # 右眉
    [22, 23], [23, 24], [24, 25], [25, 26], # 左眉
    [27, 28], [28, 29], [29, 30], # 鼻
    [31, 32], [32, 33], [33, 34], [34, 35], # 鼻下の横線
    [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36], # 右目
    [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42], # 左目
    [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [59, 48], # 唇外輪
    [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [67, 60] # 唇内輪
]

heatmap_num = len(heatmap_dict.keys())
paf_num = len(limb_dict) * 2

dataset = Train_Dataset(
    heatmap_num=heatmap_num,
    paf_num=paf_num,
    heatmap_dict=heatmap_dict,
    limb_dict=limb_dict,
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
model = model.to(device)

# 定义损失和优化器
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)

model.eval()

log_recorder = ''
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        images, labels = batch

        heatmaps_target = labels['heatmaps_target']
        heatmap_masks = labels['heatmap_masks']
        pafs_target = labels['pafs_target']
        paf_masks = labels['paf_masks']

        images = images.to(device)
        heatmaps_target = heatmaps_target.to(device)
        heatmap_masks = heatmap_masks.to(device)
        pafs_target = pafs_target.to(device)
        paf_masks = paf_masks.to(device)

        heatmaps_pre, pafs_pre = model(images)

        loss = Loss_Weighted()
        loss = loss.calc(heatmaps_pre, heatmaps_target, heatmap_masks) + loss.calc(pafs_pre, pafs_target, paf_masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    output = 'epoch: %d/%d loss: %f'%(epoch + 1, num_epochs, total_loss)
    print(output)
    log_recorder += output + '\n'

output_pre = 'mnv3s-ep%d-loss%.2f'%(num_epochs, total_loss)
torch.save(model.state_dict(), os.path.join(model_save_dir, output_pre + '.pth'))
with open(os.path.join(model_save_dir, output_pre + '.log'), 'w') as log:
    log.write(log_recorder)
