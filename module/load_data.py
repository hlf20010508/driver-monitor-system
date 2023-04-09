import os
from torch.utils.data import Dataset as Dst
from module.entity import TRANSFORMS, BODY_CLASS_DICT, BODY_CLASS_NUM_BALANCE
import torchvision.transforms as tf
from PIL import Image
import json
import numpy as np
from scipy import ndimage

# 热点范围参数
th = 4.6052
delta = th * 2 ** 0.5
# 可影响半径
sigma = 1
# paf半宽度参数
threshold = 2
stride = 16

# input size 640x480
class Train_Dataset(Dst):
    def __init__(
            self,
            heatmap_num,
            paf_num,
            heatmap_dict,
            limb_dict,
            annotation_path,
            img_root_path,
        ):
        self.heatmap_num = heatmap_num
        self.paf_num = paf_num
        self.heatmap_dict = heatmap_dict
        self.limb_dict = limb_dict
        self.annotation_path = annotation_path
        self.img_root_path = img_root_path
        # 处理图片
        self.transforms = TRANSFORMS
        # 获取图片路径列表和标签列表
        self.img_path_list, self.label_list, self.img_index_list = self.get_item_list()
    
    # 导入图片
    def get_image_matrix(self, path):
        image = Image.open(path).convert('RGB')
        return image

    # 输出处理过的图片数据和图片标签
    def __getitem__(self, index):
        return self.transforms(self.get_image_matrix(self.img_path_list[self.img_index_list[index]])), self.label_list[self.img_index_list[index]]
    
    # 生成图片路径列表和标签列表
    def get_item_list(self):
        # 打开标注文件
        with open(self.annotation_path, 'r') as annotation_file:
            annotation = json.load(annotation_file)
        img_path_list = []
        img_index_list = [] # 用于平衡各种动作的数量
        label_list = []
        for item_index, item in enumerate(annotation):
            name = item['data']['image'].strip().split('/')[-1]
            img_path_list.append(os.path.join(self.img_root_path, name))
            heatmap_points = [() for i in range(self.heatmap_num)]
            pafs = [() for i in range(self.paf_num // 2)]
            for result in item['annotations'][0]['result']:
                if result['type'] == 'keypointlabels':
                    index = self.heatmap_dict[result['value']['keypointlabels'][0]]
                    img_width = result['original_width']
                    img_height = result['original_height']
                    x = result['value']['x']
                    y = result['value']['y']
                    x = x / 100 * img_width / stride
                    y = y / 100 * img_height / stride
                    heatmap_points[index] = (x, y)
                elif result['type'] == 'choices':
                    class_name = result['value']['choices'][0]
                    for i in range(BODY_CLASS_NUM_BALANCE[class_name]):
                        img_index_list.append(item_index)
            for limb_index in range(len(self.limb_dict)):
                start = heatmap_points[self.limb_dict[limb_index][0]]
                end = heatmap_points[self.limb_dict[limb_index][1]]
                if start and end:
                    x1, y1 = start
                    x2, y2 = end
                    pafs[limb_index] = (x1, y1, x2, y2)

            heatmaps_target, heatmap_masks = self.get_heatmaps_and_masks(heatmap_points, img_width, img_height)
            pafs_target, paf_masks = self.get_pafs_and_masks(pafs, img_width, img_height)
            label_list.append({
                'heatmaps_target': heatmaps_target,
                'heatmap_masks': heatmap_masks,
                'pafs_target': pafs_target,
                'paf_masks': paf_masks
            })
        return img_path_list, label_list, img_index_list

    def get_heatmaps_and_masks(self, point_list, img_width, img_height):
        heatmap_list = []
        for point in point_list:
            heatmap= self.heatmap_gen(point, img_width, img_height)
            heatmap_list.append(heatmap)
        mask = self.mask_gen(heatmap_list, img_width, img_height)
        return np.array(heatmap_list), np.array([mask])

    def get_pafs_and_masks(self, limb_list, img_width, img_height):
        paf_list = []
        for limb in limb_list:
            paf_x, paf_y = self.paf_gen(limb, img_width, img_height)
            paf_list.append(paf_x)
            paf_list.append(paf_y)
        mask = self.mask_gen(paf_list, img_width, img_height)
        return np.array(paf_list), np.array([mask])

    def mask_gen(self, map_list, img_width, img_height):
        k_size = 3
        mask = np.zeros([img_height // stride, img_width // stride], dtype=float)
        for map in map_list:
            dilate = ndimage.grey_dilation(map ,size=(k_size, k_size))
            mask[np.where(dilate > 0.2)] = 1
        return mask

    # 生成关节点热力图
    def heatmap_gen(self, point, img_width, img_height):
        # 生成全零矩阵对热力图进行初始化
        heatmap = np.zeros([img_height // stride, img_width // stride], dtype=float)
        if point:
            # 中心点
            center_x, center_y = point
            # 确定左边界和上边界
            x0 = int(max(0, center_x - delta * sigma))
            y0 = int(max(0, center_y - delta * sigma))
            # 确定右边界和下边界
            x1 = int(min(img_width // stride, center_x + delta * sigma))
            y1 = int(min(img_height // stride, center_y + delta * sigma))

            exp_factor = 1 / 2.0 / sigma / sigma
            # 根据确定的边界提取该热点部分
            arr_heatmap = heatmap[y0:y1, x0:x1]
            # 使用二维正态分布赋值
            y_vec = (np.arange(y0, y1) - center_y)**2
            x_vec = (np.arange(x0, x1) - center_x)**2
            xv, yv = np.meshgrid(x_vec, y_vec)
            arr_sum = exp_factor * (xv + yv)
            arr_exp = np.exp(-arr_sum)
            arr_exp[arr_sum > th] = 0
            # 将结果传入heatmap
            heatmap[y0:y1, x0:x1] = np.maximum(arr_heatmap, arr_exp)
        return heatmap

    def paf_gen(self, limb, img_width, img_height):
        paf_x = np.zeros([img_height // stride, img_width // stride], dtype=float)
        paf_y = np.zeros([img_height // stride, img_width // stride], dtype=float)
        if limb:
            # 起点坐标
            x_from, y_from, x_to, y_to = limb
            # 对应的水平向量
            vector_x = x_to - x_from
            # 对应的垂直向量
            vector_y = y_to - y_from
            # 计算模长
            module = (vector_x**2 + vector_y**2) ** 0.5
            if module != 0:
                # 将向量首位向外拓展，用于组成六变形
                min_x = max(0, int(min(x_from, x_to) - threshold))
                min_y = max(0, int(min(y_from, y_to) - threshold))
                max_x = min(img_width // stride, int(max(x_from, x_to) + threshold))
                max_y = min(img_height // stride, int(max(y_from, y_to) + threshold))
                # 计算单位向量
                norm_x = vector_x / module
                norm_y = vector_y / module
                # 以下算法为：利用两向量的叉乘为平行四边形的面积的定律来画出六变形
                for y in range(min_y, max_y):
                    for x in range(min_x, max_x):
                        # 得到由遍历的点到起点所组成的向量bec
                        bec_x = x - x_from
                        bec_y = y - y_from
                        # 计算bec向量与单位向量的叉乘，得到的是两个向量组成的平行四边形的面积
                        size = abs(bec_x * norm_y - bec_y * norm_x)
                        # 如果计算得到的面积比阈值大，则不赋值
                        if size > threshold:
                            continue
                        paf_x[y][x] = norm_x
                        paf_y[y][x] = norm_y
        return paf_x, paf_y


    def __len__(self):
        return len(self.img_index_list)

class Train_Dataset_Class(Dst):
    def __init__(self, path, width, height):
        self.path = path
        self.transforms = tf.Compose([
            tf.Resize((width, height)),
            tf.ToTensor(),
            tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        self.item_list, self.item_class_list = self.get_item_list()
        
    def get_image_matrix(self, path):
        image=Image.open(path).convert('RGB')
        return image
    
    def __getitem__(self, index):
        return self.item_list[index], self.item_class_list[index]
        
    def get_item_list(self):
        class_list = [f for f in os.listdir(self.path) if not f.startswith('.')]
        item_list = []
        item_class_list = []
        for dir in class_list:
            item_name_list = [f for f in os.listdir(os.path.join(self.path, dir)) if not f.startswith('.')]
            for item_name in item_name_list:
                item = self.transforms(self.get_image_matrix(os.path.join(self.path, dir, item_name)))
                item_list.append(item)
                item_class_list.append(BODY_CLASS_DICT[dir])
        return item_list, item_class_list

    def __len__(self):
        return len(self.item_list)

class STGCN_Dataset(Dst):
    def __init__(
            self,
            num_nodes,
            time_len,
            point_dict,
            class_dict,
            annotation_path,
        ):
        self.num_nodes = num_nodes
        self.time_len = time_len
        self.point_dict = point_dict
        self.class_dict = class_dict
        self.annotation_path = annotation_path

        self.item_list, self.label_list = self.get_item_list()
    
    def __getitem__(self, index):
        return self.item_list[index: index + self.time_len], self.label_list[index + self.time_len - 1]
    
    # 生成图片路径列表和标签列表
    def get_item_list(self):
        # 打开标注文件
        with open(self.annotation_path, 'r') as annotation_file:
            annotation = json.load(annotation_file)
        item_list = []
        label_list = []
        for item in annotation:
            points_list = [(-1, -1) for i in range(self.num_nodes)]
            for result in item['annotations'][0]['result']:
                if result['type'] == 'keypointlabels':
                    index = self.point_dict[result['value']['keypointlabels'][0]]
                    img_width = result['original_width']
                    img_height = result['original_height']
                    x = result['value']['x']
                    y = result['value']['y']
                    x = x / 100 * img_width / stride
                    y = y / 100 * img_height / stride
                    points_list[index] = (x, y)
                elif result['type'] == 'choices':
                    class_name = result['value']['choices'][0]
                    class_id = self.class_dict[class_name]
                    label_list.append(class_id)
            item_list.append(points_list)
        item_list = np.array(item_list)
        # mean = np.mean(item_list, axis=0) # 计算均值
        # std = np.std(item_list, axis=0)   # 计算标准差
        # item_list = (item_list - mean) / std # 标准化处理
        item_list = item_list.astype(np.float32)
        return item_list, np.array(label_list)

    def __len__(self):
        return len(self.item_list) - self.time_len + 1
    