import torchvision.transforms as tf

COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
    [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
    [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
]

TRANSFORMS = tf.Compose([
    tf.ToTensor(),
    tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

INFINITE = 10000

THREADHOLD = 150

BODY_HEATMAP_DICT = {
    'left-shoulder': 0,
    'left-elbow': 1,
    'left-wrist': 2,
    'right-shoulder': 3,
    'right-elbow': 4,
    'right-wrist': 5,
    'nose': 6,
    'right-ear': 7,
    'wheel': 8
}

BODY_LIMB_DICT = [
    [0, 1], [1, 2], # left arm
    [3, 4], [4, 5], # right arm
]
# BODY_HEATMAP_DICT = {
#     'left-shoulder': 0,
#     'left-elbow': 1,
#     'left-wrist': 2,
#     'right-shoulder': 3,
#     'right-elbow': 4,
#     'right-wrist': 5,
#     'head': 6,
#     'wheel': 7
# }

# BODY_LIMB_DICT = [
#     [0, 6], [1, 0], [2, 1], # left arm
#     [3, 6], [4, 3], [5, 4], # right arm
#     [6, 7] # head to wheel
# ]

# FACE_HEATMAP_DICT = {
#     'left_eyebrow_out': 0,
#     'right_eyebrow_out': 1,
#     'left_eyebrow_in': 2,
#     'right_eyebrow_in': 3,
#     'left_eyebrow_center_top': 4,
#     'left_eyebrow_center_bottom': 5,
#     'right_eyebrow_center_top': 6,
#     'right_eyebrow_center_bottom': 7,
#     'left_eye_out': 8,
#     'right_eye_out': 9,
#     'left_eye_in': 10,
#     'right_eye_in': 11,
#     'left_eye_center_top': 12,
#     'left_eye_center_bottom': 13,
#     'right_eye_center_top': 14,
#     'right_eye_center_bottom': 15,
#     'left_eye_pupil': 16,
#     'right_eye_pupil': 17,
#     'left_nose_out': 18,
#     'right_nose_out': 19,
#     'nose_center_top': 20,
#     'nose_center_bottom': 21,
#     'left_mouth_out': 22,
#     'right_mouth_out': 23,
#     'mouth_center_top_lip_top': 24,
#     'mouth_center_top_lip_bottom': 25,
#     'mouth_center_bottom_lip_top': 26,
#     'mouth_center_bottom_lip_bottom': 27,
#     'chin': 28
#     # 'left_ear_top': 28,
#     # 'right_ear_top': 29,
#     # 'left_ear_bottom': 30,
#     # 'right_ear_bottom': 31,
#     # 'left_ear_canal': 32,
#     # 'right_ear_canal': 33,
#     # 'chin': 34
# }

FACE_HEATMAP_DICT = {
    'left_eyebrow_out': 0,
    'left_eyebrow_center': 1,
    'left_eyebrow_in': 2,
    'right_eyebrow_out': 3,
    'right_eyebrow_center': 4,
    'right_eyebrow_in': 5,
    'left_eye_out': 6,
    'left_eye_center_top': 7,
    'left_eye_in': 8,
    'left_eye_center_bottom': 9,
    'right_eye_out': 10,
    'right_eye_center_top': 11,
    'right_eye_in': 12,
    'right_eye_center_bottom': 13,
    'nose_center': 14,
    'mouse_left': 15,
    'mouse_center_top': 16,
    'mouse_right': 17,
    'mouse_center_bottom': 18,
    'chin': 19
}

FACE_LIMB_DICT = [
    [0, 1], [1, 2], # left eyebrow
    [3, 4], [4, 5], # right eyebrow
    [6, 7], [7, 8], [8, 9], # left eye
    [10, 11], [11, 12], [12, 13], # right eye
    [14, 15], [15, 16], [16, 17], [17, 18] # mouse
]

FACE_LIMB_DICT_NEW = [
    [0, 1], [1, 2], # left eyebrow
    [3, 4], [4, 5], # right eyebrow
    [6, 7], [7, 8], [8, 9], [9, 6], # left eye
    [10, 11], [11, 12], [12, 13], [13, 10], # right eye
    [15, 16], [16, 17], [17, 18], [18, 15] # mouse
]

# FACE_LIMB_DICT = [
#     [0, 4], [4, 2], [2, 5], [5, 0], # left eyebrow
#     [1, 6], [6, 3], [3, 7], [7, 1], # right eyebrow
#     [8, 12], [12, 10], [10, 13], # left eye
#     [9, 14], [14, 11], [11, 16], # right eye
#     [18, 20], [20, 19], [19, 21], # nose
#     [22, 24], [24, 23], [22, 25], [25, 23], [22, 26], [26, 23], [22, 27], [27, 23], # mouse
#     [27, 28] # mouse to chin
# ]

BODY_CLASS_DICT = {
    'safe_driving': 0,
    'texting': 1,
    'talking_on_phone': 2,
    'drinking': 3,
    'reaching': 4,
    'hair_and_makeup': 5
}

# BODY_CLASS_DICT = {
#     'safe_driving': 0,
#     'texting': 1,
#     'talking_on_phone': 2,
#     'operating_radio': 3,
#     'drinking': 4,
#     'reaching_behind': 5,
#     'hair_and_makeup': 6
# }

BODY_CLASS_LIST = [
    'safe_driving',
    'texting',
    'talking_on_phone',
    'drinking',
    'reaching',
    'hair_and_makeup'
]
# BODY_CLASS_LIST = [
#     'safe_driving',
#     'texting',
#     'talking_on_phone',
#     'operating_radio',
#     'drinking',
#     'reaching_behind',
#     'hair_and_makeup'
# ]
