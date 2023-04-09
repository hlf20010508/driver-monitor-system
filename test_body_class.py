import torch
import os
from module.entity import BODY_CLASS_DICT, BODY_HEATMAP_DICT, BODY_LIMB_DICT, TIME_LEN
from model.stgcn import STGCN
import numpy as np

class_model_load_path = os.environ.get('class_model_load_path', './model.pth')
video_path = os.environ['video_path']

time_len = int(os.environ.get('time_len', TIME_LEN))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_nodes = len(BODY_HEATMAP_DICT)
class_num = len(BODY_CLASS_DICT)

model = STGCN(
    num_nodes=num_nodes,
    time_len=time_len,
    class_num=class_num
)

model_dict = torch.load(class_model_load_path, map_location=device)
model.load_state_dict(model_dict)
model = model.to(device)

model.eval()

edge_index = torch.tensor(np.array(list(zip(*BODY_LIMB_DICT)))).to(device)

def detect_class(point_list):
    point_list = torch.FloatTensor(point_list).unsqueeze(0).to(device)

    out = model(point_list, edge_index)
    
    out = torch.squeeze(out)

    max_index = int(torch.argmax(out))

    label = list(BODY_CLASS_DICT.keys())[max_index]
    return label
