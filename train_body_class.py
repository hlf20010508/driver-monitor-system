import os
import torch
from module.load_data import Train_Dataset_Class
from module.entity import BODY_CLASS_DICT
from torchvision.models import inception_v3, Inception_V3_Weights

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

img_root_path = os.environ['img_root_path']
model_save_dir = os.environ.get('model_save_dir', './')

width = int(os.environ.get('width', 299))
height = int(os.environ.get('height', 299))

num_epochs = int(os.environ.get('num_epochs', 100))
batch_size = int(os.environ.get('batch_size', 50))
learning_rate = float(os.environ.get('learning_rate', 1e-3))
weight_decay = float(os.environ.get('weight_decay', 5e-4))

class_size = len(BODY_CLASS_DICT)

dataset = Train_Dataset_Class(
    path=img_root_path,
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

model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, class_size)
)

model = model.to(device)

# 定义损失和优化器
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)

criterion = torch.nn.CrossEntropyLoss()

model.eval()

log_recorder = ''
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        images, labels = batch

        images = images.to(device)
        labels = labels.to(device)

        out = model(images)

        print(labels)
        print(out)
        print()

        loss = criterion(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    output = 'epoch: %d/%d loss: %f'%(epoch + 1, num_epochs, total_loss)
    print(output)
    log_recorder += output + '\n'

output_pre = 'body-ep%d-loss%.4f'%(num_epochs, total_loss)
torch.save(model.state_dict(), os.path.join(model_save_dir, output_pre + '.pth'))
with open(os.path.join(model_save_dir, output_pre + '.log'), 'w') as log:
    log.write(log_recorder)
