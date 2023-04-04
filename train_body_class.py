from model.stgcn import STGCN
from module.entity import BODY_CLASS_DICT, BODY_HEATMAP_DICT, BODY_LIMB_DICT
from module.load_data import STGCN_Dataset
import os
import torch
import numpy as np

num_nodes = len(BODY_HEATMAP_DICT)
class_num = len(BODY_CLASS_DICT)

annotation_path = os.environ['annotation_path']
model_save_dir = os.environ.get('model_save_dir', './')

num_epochs = int(os.environ.get('num_epochs', 300))
batch_size = int(os.environ.get('batch_size', 3))
learning_rate = float(os.environ.get('learning_rate', 5e-5))
weight_decay = float(os.environ.get('weight_decay', 5e-4))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

dataset = STGCN_Dataset(
    num_nodes=num_nodes,
    point_dict=BODY_HEATMAP_DICT,
    class_dict=BODY_CLASS_DICT,
    annotation_path=annotation_path,
)

train_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

model = STGCN(num_nodes=num_nodes, class_num=class_num)

criterion = torch.nn.CrossEntropyLoss()

# 定义损失和优化器
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)

model = model.to(device)
model.eval()

edge_index = torch.tensor(np.array(list(zip(*BODY_LIMB_DICT)))).to(device)

log_recorder = ''
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        items, labels = batch

        items = items.to(device)
        labels = labels.to(device)

        out = model(items, edge_index)

        loss = criterion(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    output = 'epoch: %d/%d loss: %f'%(epoch + 1, num_epochs, total_loss)
    print(output)
    log_recorder += output + '\n'

output_pre = 'body-stgcn-ep%d-loss%.2f'%(num_epochs, total_loss)
torch.save(model.state_dict(), os.path.join(model_save_dir, output_pre + '.pth'))
with open(os.path.join(model_save_dir, output_pre + '.log'), 'w') as log:
    log.write(log_recorder)
