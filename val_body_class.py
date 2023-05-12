from model.stgcn import STGCN
from module.entity import BODY_CLASS_DICT, BODY_HEATMAP_DICT, BODY_LIMB_DICT, TIME_LEN
from module.load_data import STGCN_Dataset
import os
import torch
import numpy as np

num_nodes = len(BODY_HEATMAP_DICT)
class_num = len(BODY_CLASS_DICT)

time_len = int(os.environ.get('time_len', TIME_LEN))

annotation_path = os.environ['annotation_path']
model_load_path = os.environ['model_load_path']

num_epochs = int(os.environ.get('num_epochs', 1))
batch_size = int(os.environ.get('batch_size', 1))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = STGCN_Dataset(
    num_nodes=num_nodes,
    time_len=time_len,
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

model = STGCN(
    num_nodes=num_nodes,
    time_len=time_len,
    class_num=class_num
)

model_dict = torch.load(model_load_path, map_location=device)
model.load_state_dict(model_dict)

model = model.to(device)
model.eval()

edge_index = torch.tensor(np.array(list(zip(*BODY_LIMB_DICT)))).to(device)

confusion_matrix = np.zeros((class_num, class_num), dtype=int)
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        items, labels = batch

        items = items.to(device)
        labels = int(labels)

        out = model(items, edge_index)
        out = torch.squeeze(out)

        max_index = int(torch.argmax(out))
        
        confusion_matrix[max_index][labels] += 1

ave_p = []
ave_r = []
ave_f1 = []
for index in range(class_num):
    tp = confusion_matrix[index][index]
    fp = 0
    fn = 0
    for i in range(class_num):
        if i != index:
            fp += confusion_matrix[index][i]
            fn += confusion_matrix[i][index]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    ave_p.append(precision)
    ave_r.append(recall)
    ave_f1.append(f1)

ave_p = sum(ave_p) / len(ave_p)
ave_r = sum(ave_r) / len(ave_r)
ave_f1 = sum(ave_f1) / len(ave_f1)

print('ave_p: %.2f%%'%(ave_p * 100))
print('ave_r: %.2f%%'%(ave_r * 100))
print('ave_f1: %.2f%%'%(ave_f1 * 100))
