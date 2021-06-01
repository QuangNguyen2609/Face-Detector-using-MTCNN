from PIL import Image
import cv2
from facenet_pytorch import MTCNN

mtcnn = MTCNN()

vid1 = '/Users/quangnguyen/Documents/Deep Learning ULTIMATE/face_detection_data/strange_person1.mp4'
vid2 = '/Users/quangnguyen/Documents/Deep Learning ULTIMATE/face_detection_data/strange_person2.mp4'
vid3 = '/Users/quangnguyen/Documents/Deep Learning ULTIMATE/face_detection_data/strange_person3.mp4'
sis  = '/Users/quangnguyen/Documents/Deep Learning ULTIMATE/face_detection_data/sis.mp4'
dad  = '/Users/quangnguyen/Documents/Deep Learning ULTIMATE/face_detection_data/dad.mp4'
frames = []
for vid in [sis]:
    cap = cv2.VideoCapture(vid)
    # count number of frame
    cap_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # loop through all the frames and append to frames list
    for _ in range(cap_len):
        success,frame = cap.read()
        # check if true or not
        if not success:
            continue
        #convert to RGB format
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frames.append(frame)

    paths = [f'/Users/quangnguyen/Documents/Deep Learning ULTIMATE/face_detection_data/Train_data/not_dang_quang/image_{i}.jpg' for i in range(len(frames))]

    for frame,path in zip(frames,paths) :
        mtcnn(frame,save_path=path)