import os
import torch
from module.load_data import Train_Dataset
from module.loss import Loss_Weighted
from model.dm_net import DMNet

base_model = os.environ.get('base_model', 'mnv3s')
annotation_path = os.environ['annotation_path']
img_root_path = os.environ['img_root_path']
model_save_dir = os.environ.get('model_save_dir', './')

num_epochs = int(os.environ.get('num_epochs', 300))
batch_size = int(os.environ.get('batch_size', 24))
learning_rate = float(os.environ.get('learning_rate', 5e-5))
weight_decay = float(os.environ.get('weight_decay', 5e-4))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

heatmap_dict = {
    'left_eyebrow_out': 0,
    'right_eyebrow_out': 1,
    'left_eyebrow_in': 2,
    'right_eyebrow_in': 3,
    'left_eyebrow_center_top': 4,
    'left_eyebrow_center_bottom': 5,
    'right_eyebrow_center_top': 6,
    'right_eyebrow_center_bottom': 7,
    'left_eye_out': 8,
    'right_eye_out': 9,
    'left_eye_in': 10,
    'right_eye_in': 11,
    'left_eye_center_top': 12,
    'left_eye_center_bottom': 13,
    'right_eye_center_top': 14,
    'right_eye_center_bottom': 15,
    'left_eye_pupil': 16,
    'right_eye_pupil': 17,
    'left_nose_out': 18,
    'right_nose_out': 19,
    'nose_center_top': 20,
    'nose_center_bottom': 21,
    'left_mouth_out': 22,
    'right_mouth_out': 23,
    'mouth_center_top_lip_top': 24,
    'mouth_center_top_lip_bottom': 25,
    'mouth_center_bottom_lip_top': 26,
    'mouth_center_bottom_lip_bottom': 27,
    'chin': 28
    # 'left_ear_top': 28,
    # 'right_ear_top': 29,
    # 'left_ear_bottom': 30,
    # 'right_ear_bottom': 31,
    # 'left_ear_canal': 32,
    # 'right_ear_canal': 33,
    # 'chin': 34
}
limb_dict = [
    [0, 4], [4, 2], [2, 5], [5, 0], # left eyebrow
    [1, 6], [6, 3], [3, 7], [7, 1], # right eyebrow
    [8, 12], [12, 10], [10, 13], # left eye
    [9, 14], [14, 11], [11, 16], # right eye
    [18, 20], [20, 19], [19, 21], # nose
    [22, 24], [24, 23], [22, 25], [25, 23], [22, 26], [26, 23], [22, 27], [27, 23], # mouse
    [27, 28] # mouse to chin
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
    base_model=base_model,
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

output_pre = 'face-%s-ep%d-loss%.2f'%(base_model, num_epochs, total_loss)
torch.save(model.state_dict(), os.path.join(model_save_dir, output_pre + '.pth'))
with open(os.path.join(model_save_dir, output_pre + '.log'), 'w') as log:
    log.write(log_recorder)
