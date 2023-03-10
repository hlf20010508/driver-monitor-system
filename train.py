import os
import torch
import torchvision.transforms as transforms
from module.load_data import Dataset
from module.loss import Loss_Weighted
from model.dm_net import DMNet
import cv2

mode = os.environ.get('mode', 'train')
base_model = os.environ.get('base_model', 'vgg19')
annotation_path = os.environ['annotation_path']
img_root_path = os.environ['img_root_path']
model_save_path = os.environ.get('model_save_path', 'model.pth')
model_load_path = os.environ.get('model_load_path', 'model.pth')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

width = int(os.environ.get('width', 224))
height = int(os.environ.get('height', 224))
num_epochs = int(os.environ.get('num_epochs', 300))
batch_size = int(os.environ.get('batch_size', 25))
learning_rate = float(os.environ.get('learning_rate', 4e-5))
weight_decay = float(os.environ.get('weight_decay', 5e-4))
heatmap_num = int(os.environ.get('heatmap_num', 6))
paf_num = int(os.environ.get('paf_num', 8))

heatmap_dict = {
    'left-shoulder': 0,
    'left-elbow': 1,
    'left-wrist': 2,
    'right-shoulder': 3,
    'right-elbow': 4,
    'right-wrist': 5,
}

# 下标为起点，值为终点，-1表示不连通
limb_dict = [1, 2, -1, 4, 5, -1]
# 起点对应在body_paf中的下标
paf_dict = [0, 1, -1, 2, 3, -1]

dataset = Dataset(
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

def train():
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
    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()
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

            T = transforms.Resize((width, height))

            heatmaps_pre = T(heatmaps_pre)
            pafs_pre = T(pafs_pre)

            loss = Loss_Weighted()
            loss = loss.calc(heatmaps_pre, heatmaps_target, heatmap_masks) + loss.calc(pafs_pre, pafs_target, paf_masks)

            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        print('epoch: %d/%d loss: %f'%(epoch + 1, num_epochs, total_loss))
    torch.save(model.state_dict(), model_save_path)

def test():
    model = DMNet(
        heatmap_num=heatmap_num,
        paf_num=paf_num
    )
    model_dict = torch.load(model_load_path, map_location=device)
    model.load_state_dict(model_dict)
    model = model.to(device)
    model.eval()

    for batch in train_loader:
        images, labels = batch
        images = images.to(device)

        heatmaps, pafs = model(images)

        for m in heatmaps:
            for image in m:
                image = image.to('cpu')
                image = image.detach().numpy()
                cv2.imshow('monitor', image)
                cv2.waitKey()

        for m in pafs:
            for image in m:
                image = image.to('cpu')
                image = image.detach().numpy()
                cv2.imshow('monitor', image)
                cv2.waitKey()

if __name__ == '__main__':
    if mode == 'train':
        train()
    elif mode == 'test':
        test()
    else:
        print('mode should be train or test')
