import os
import json
import wget
from PIL import Image, ImageOps, ImageDraw

labels = [
    'left_eyebrow_out',
    'right_eyebrow_out',
    'left_eyebrow_in',
    'right_eyebrow_in',
    'left_eyebrow_center_top',
    'left_eyebrow_center_bottom',
    'right_eyebrow_center_top',
    'right_eyebrow_center_bottom',
    'left_eye_out',
    'right_eye_out',
    'left_eye_in',
    'right_eye_in',
    'left_eye_center_top',
    'left_eye_center_bottom',
    'right_eye_center_top',
    'right_eye_center_bottom',
    'left_eye_pupil',
    'right_eye_pupil',
    'left_nose_out',
    'right_nose_out',
    'nose_center_top',
    'nose_center_bottom',
    'left_mouth_out',
    'right_mouth_out',
    'mouth_center_top_lip_top',
    'mouth_center_top_lip_bottom',
    'mouth_center_bottom_lip_top',
    'mouth_center_bottom_lip_bottom',
    'left_ear_top',
    'right_ear_top',
    'left_ear_bottom',
    'right_ear_bottom',
    'left_ear_canal',
    'right_ear_canal',
    'chin'
]

download_root_path = '/Users/hlf/Downloads/lfpw/images_origin'
img_save_path = '/Users/hlf/Downloads/lfpw/images'
# img_point_save_path = '/Users/hlf/Downloads/lfpw/images_point'
annotation_path = '/Users/hlf/Downloads/lfpw/annotations.json'
# train_csv_path = '/Users/hlf/Downloads/kbvt_lfpw_v1_train.csv'
# test_csv_path = '/Users/hlf/Downloads/kbvt_lfpw_v1_test.csv'
# train_csv_file = open(train_csv_path, 'r')
# train_csv_lines = train_csv_file.readlines()
# test_csv_file = open(test_csv_path, 'r')
# test_csv_lines = test_csv_file.readlines()

# items = []
# count = 1

# os.mkdir(download_root_path)
os.mkdir(img_save_path)
# def run(lines):
#     global items
#     global count
#     for line in lines[1:]:
#         item = {}
#         line = line.strip().split()
#         # 仅使用平均点
#         if line[1] != 'worker_1':
#             continue

#         download_url = line[0]
#         ext = download_url.strip().split('/')[-1].split('.')[-1]
#         name = '%d.%s'%(count, ext)
#         file_path = os.path.join(download_root_path, name)
#         try:
#             wget.download(download_url, file_path)
#         except Exception as e:
#             print(e)
#             continue

#         try:
#             image = Image.open(file_path).convert('RGB')
#         except Exception as e:
#             print('\n%s'%e)
#             os.remove(file_path)
#             continue

#         name = '%d.jpg'%count

#         width, height = image.size
#         line = line[2:]
#         for label_index in range(len(line)):
#             line[label_index] = float(line[label_index])
#         if width / height > 640 / 480:
#             height_new = int(height / width * 640)
#             image = image.resize((640, height_new), Image.ANTIALIAS)
#             padding_top = (480 - height_new) // 2
#             padding_bottom = 480 - padding_top - height_new
#             border = (0, padding_top, 0, padding_bottom)
#             image = ImageOps.expand(image, border=border, fill=0)
#             for label_index in range(len(line) // 3):
#                 line[label_index * 3] = line[label_index * 3] / width * 640
#                 line[label_index * 3 + 1] = line[label_index * 3 + 1] / height * height_new + padding_top
#         elif width / height < 640 / 480:
#             width_new = int(width / height * 480)
#             image = image.resize((width_new, 480), Image.ANTIALIAS)
#             padding_left = (640 - width_new) // 2
#             padding_right = 640 - padding_left - width_new
#             border = (padding_left, 0, padding_right, 0)
#             image = ImageOps.expand(image, border=border, fill=0)
#             for label_index in range(len(line) // 3):
#                 line[label_index * 3] = line[label_index * 3] / width * width_new + padding_left
#                 line[label_index * 3 + 1] = line[label_index * 3 + 1] / height * 480
#         else:
#             image = image.resize((640, 480), Image.ANTIALIAS)
#             for label_index in range(len(line) // 3):
#                 line[label_index * 3] = line[label_index * 3] / width * 640
#                 line[label_index * 3 + 1] = line[label_index * 3 + 1] / height * 480
        
#         image_draw = ImageDraw.Draw(image)
#         for label_index in range(len(line) // 3):
#             image_draw.rounded_rectangle((line[label_index * 3] - 2, line[label_index * 3 + 1] - 2, line[label_index * 3] + 2, line[label_index * 3 + 1] + 2),fill=(255,0,0))

#         image.save(os.path.join(img_save_path, name))

#         width, height = (640 ,480)
        

#         item['file_upload'] = name
#         item['annotations'] = [{
#             'result': []
#         }]

#         for  label_index in range(len(labels)):
#             item['annotations'][0]['result'].append({
#                 'original_width': width,
#                 'original_height': height,
#                 'value': {
#                     'x': line[label_index * 3] / width * 100,
#                     'y': line[label_index * 3 + 1] / height * 100,
#                     'keypointlabels': [labels[label_index]]
#                 }
#             })
#         items.append(item)
#         count += 1

# run(train_csv_lines)
# run(test_csv_lines)
# with open(annotation_path, 'w') as annotation_file:
#     json.dump(items, annotation_file)

with open('delete_img.txt', 'r') as  delete_img_file:
    delete_img_list = [i.strip() for i in delete_img_file.readlines()]

img_list = [i for i in os.listdir(download_root_path) if not i.startswith('.')]

for img in img_list:
    if img.split('.')[0] not in delete_img_list:
        file_path = os.path.join(download_root_path, img)
        image = Image.open(file_path).convert('RGB')
        width, height = image.size

        if width / height > 640 / 480:
            height_new = int(height / width * 640)
            image = image.resize((640, height_new), Image.ANTIALIAS)
            padding_top = (480 - height_new) // 2
            padding_bottom = 480 - padding_top - height_new
            border = (0, padding_top, 0, padding_bottom)
            image = ImageOps.expand(image, border=border, fill=0)
        elif width / height < 640 / 480:
            width_new = int(width / height * 480)
            image = image.resize((width_new, 480), Image.ANTIALIAS)
            padding_left = (640 - width_new) // 2
            padding_right = 640 - padding_left - width_new
            border = (padding_left, 0, padding_right, 0)
            image = ImageOps.expand(image, border=border, fill=0)
        else:
            image = image.resize((640, 480), Image.ANTIALIAS)

        image.save(os.path.join(img_save_path, img.split('.')[0] + '.jpg'))

with open(annotation_path, 'r') as annotation_file:
    annotation = json.load(annotation_file)
item = []

for i in annotation:
    if i['file_upload'].split('.')[0] not in delete_img_list:
        item.append(i)

with open(annotation_path, 'w') as annotation_file:
    json.dump(item, annotation_file)
