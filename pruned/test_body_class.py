import torch
import os
from torchvision.models import inception_v3
from module.entity import BODY_CLASS_DICT, COLORS
import cv2
from PIL import Image
import torchvision.transforms as tf

class_model_load_path = os.environ.get('class_model_load_path', './model.pth')
video_path = os.environ['video_path']

class_size = len(BODY_CLASS_DICT)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = inception_v3()
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, class_size)
)
model_dict = torch.load(class_model_load_path, map_location=device)
model.load_state_dict(model_dict)
model = model.to(device)

model.eval()

TRANSFORMS = tf.Compose([
    tf.Resize((299, 299)),
    tf.ToTensor(),
    tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def detect_class(ori_img):
    image = Image.fromarray(cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB))
    image = TRANSFORMS(image)
    image = torch.unsqueeze(image, 0)

    image = image.to(device)

    out = model(image)
    
    out = torch.squeeze(out)

    max_index = int(torch.argmax(out))

    label = list(BODY_CLASS_DICT.keys())[max_index]
    return label

def detect_class_show():
    cap = cv2.VideoCapture(video_path)
    count = 0
    while(cap.isOpened()):
        ret, ori_img = cap.read() 
        if count % 10 == 0:
            label = detect_class(ori_img)

        cv2.putText(ori_img, label, (0,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, COLORS[4])
        cv2.imshow('monitor', ori_img) 
        k = cv2.waitKey(1)
        count += 1
        #q键退出
        if (k & 0xff == ord('q')): 
            break

    cap.release() 
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_class_show()