import cv2
import os

Video_Dir = "/Users/hlf/Downloads/dms640x480/hair_phone_text/gA_1_s2_2019-03-08T09;21;03+01;00_rgb_face.mp4"
Image_Dir_Root = '/Users/hlf/Downloads/dms640x480/hair_phone_text/image_face'

def video2images(Video_Dir, Image_Dir_Root):
 
    cap = cv2.VideoCapture(Video_Dir)
    c = 1  # 帧数起点
    index = 1  # 图片命名起点，如1.jpg
 
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    os.mkdir(Image_Dir_Root)

    while True:
        # 逐帧捕获
        ret, frame = cap.read()
        # 如果正确读取帧，ret为True
        if not ret:
            print("Can't receive frame.")
            break
        # 设置每5帧取一次图片，若想逐帧抽取图片，可设置c % 1 == 0
        if c % 60 == 0:
            # 图片存放路径，即图片文件夹路径
            cv2.imwrite(os.path.join(Image_Dir_Root, '%d.jpg'%index), frame) 
            index += 1
        c += 1
        cv2.waitKey(1)
        # 按键停止
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
 
video2images(Video_Dir, Image_Dir_Root)
