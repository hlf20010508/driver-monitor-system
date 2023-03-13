import os
import torch
import torchvision.transforms as transforms
from module.load_data import Train_Dataset
from module.loss import Loss_Weighted
from model.dm_net import DMNet

base_model = os.environ.get('base_model', 'mnv3s')
annotation_path = os.environ['annotation_path']
img_root_path = os.environ['img_root_path']
model_save_dir = os.environ.get('model_save_dir', './')

width = int(os.environ.get('width', 224))
height = int(os.environ.get('height', 224))
num_epochs = int(os.environ.get('num_epochs', 300))
batch_size = int(os.environ.get('batch_size', 24))
learning_rate = float(os.environ.get('learning_rate', 5e-5))
weight_decay = float(os.environ.get('weight_decay', 5e-4))
# heatmap_num = int(os.environ.get('heatmap_num', 8))
# paf_num = int(os.environ.get('paf_num', 14))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

heatmap_dict = {
    'left-shoulder': 0,
    'left-elbow': 1,
    'left-wrist': 2,
    'right-shoulder': 3,
    'right-elbow': 4,
    'right-wrist': 5,
    'head': 6,
    'wheel': 7
}

# 下标为起点，值为终点，-1表示不连通
limb_dict = [6, 0, 1, 6, 3, 4, 7, -1]
# 起点对应在pafs中的下标
paf_dict = [0, 1, 2, 3, 4, 5, 6, -1]

heatmap_num = len(heatmap_dict.keys())
paf_num = (max(paf_dict) + 1) * 2

dataset = Train_Dataset(
    heatmap_num=heatmap_num,
    paf_num=paf_num,
    heatmap_dict=heatmap_dict,
    limb_dict=limb_dict,
    paf_dict=paf_dict,
    annotation_path=annotation_path,
    img_root_path=img_root_path,
    width=width,
    height=height
)

train_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

model = DMNet(
    base_model = base_model,
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

        # T = transforms.Resize((height, width))

        # heatmaps_pre = T(heatmaps_pre)
        # pafs_pre = T(pafs_pre)

        loss = Loss_Weighted()
        loss = loss.calc(heatmaps_pre, heatmaps_target, heatmap_masks) + loss.calc(pafs_pre, pafs_target, paf_masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    output = 'epoch: %d/%d loss: %f'%(epoch + 1, num_epochs, total_loss)
    print(output)
    log_recorder += output + '\n'

output_pre = '%s-ep%d-loss%.2f'%(base_model, num_epochs, total_loss)
torch.save(model.state_dict(), os.path.join(model_save_dir, output_pre + '.pth'))
with open(os.path.join(model_save_dir, output_pre + '.log'), 'w') as log:
    log.write(log_recorder)
