from facenet_pytorch import MTCNN
import cv2
mtcnn = MTCNN()

vid = '/Users/quangnguyen/Documents/Deep Learning ULTIMATE/face_detection_data/dang_quang.mp4'
frames = []
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

paths = [f'/Users/quangnguyen/Documents/Deep Learning ULTIMATE/face_detection_data/Train_data/dang_quang/image_{i}.jpg' for i in range(len(frames))]

for frame,path in zip(frames,paths) :
    mtcnn(frame,save_path=path)