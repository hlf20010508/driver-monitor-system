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

BODY_HEATMAP_DICT = {
    'left-shoulder': 0,
    'left-elbow': 1,
    'left-wrist': 2,
    'right-shoulder': 3,
    'right-elbow': 4,
    'right-wrist': 5,
    'mouse': 6,
    'right-ear': 7,
    'wheel': 8
}

BODY_LIMB_DICT = [
    [0, 1], [1, 2], # left arm
    [3, 4], [4, 5], # right arm
    [0, 6], [3, 6], # shoulder to mouse
    [6, 7], # mouse to ear
    [2, 8], [5, 8], # wrist to wheel
]

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
    [6, 7], [7, 8], [8, 9], [9, 6], # left eye
    [10, 11], [11, 12], [12, 13], [13, 10], # right eye
    [15, 16], [16, 17], [17, 18], [18, 15] # mouse
]

BODY_CLASS_DICT = {
    'safe_driving': 0,
    'texting': 1,
    'talking_on_phone': 2,
    'drinking': 3,
    'reaching_behind': 4,
    'reaching_nearby': 5,
    'hair_and_makeup': 6,
    'tired': 7,
    'operating_radio': 8,
}

BODY_CENTER_POINT_INDEX = 6

TIME_LEN = 12
