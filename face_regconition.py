from tensorflow.keras.layers import Dense,Dropout,BatchNormalization,Flatten
from tensorflow.keras import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN


def get_vggmodel() :
    vgg=VGG16(include_top=False, weights='imagenet')

    # free pre_trained layers,only train dense layer (FC)
    for layer in vgg.layers :
        layer.trainable=False

    # input and output for model
    input = Input(shape=(128,128,3),name='image_size')
    output_vgg = vgg(input)
    x = Flatten(name='flatten')(output_vgg)
    x = Dense(4096,'relu',name='fc1')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(4096,'relu',name='fc2')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(4,'softmax',name='softmax')(x)

    vgg_model = Model(inputs=input,outputs=x)
    vgg_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
    return vgg_model
vgg_model = get_vggmodel()
vgg_model.summary()
vgg_model.load_weights('/Users/quangnguyen/Documents/Deep Learning ULTIMATE/face_detection_data/weights_best_vgg16.hdf5')

class_label=['DANG_QUANG','UNKNOWN','VAN_DANG','VAN_TRANG']

class detection :

    def __init__(self,mtcnn,classifier) :

        self.mtcnn = mtcnn
        self.classifier = classifier

    def draw(self,frame,bbox,prob,output) :
        try :
            for box in bbox :
                # draw bounding box
                cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),(102,255,51),2)

                # annotate prob on frame
                cv2.putText(frame,str(class_label[output]),(box[0],box[1]),cv2.FONT_HERSHEY_COMPLEX,1,(102,255,51),2,cv2.LINE_AA)
                cv2.putText(frame,str(prob),(box[2],box[3]),cv2.FONT_HERSHEY_COMPLEX,1,(102,255,51),2,cv2.LINE_AA)
                #return frame only return 1 face
                return frame 
        except :
            pass
        
    def detect_roi(self,bbox) :

        ROIS=[]
        for box in bbox :
            ROI = [int(box[1]),int(box[3]),int(box[0]),int(box[2])]
            ROIS.append(ROI)
        return ROIS

    def run(self) :
        cap=cv2.VideoCapture(0)
        while True :
            ret,frame=cap.read()
            try :
                bbox,prob,landmarks = self.mtcnn.detect(frame,landmarks=True)
                ROIS = self.detect_roi(bbox)
                for roi in ROIS :
                    (ystart,yend,xstart,xend) = roi
                    face  = frame[ystart:yend,xstart:xend]
                    image = cv2.resize(face,dsize=(128, 128))
                    image = image.astype('float')*1./255
                    image = np.expand_dims(image, axis=0)
                    pred  = self.classifier.predict(image)
                    output  = np.argmax(pred[0],axis=-1)
                    if np.max(pred[0]) > 0.8 and output != 1 :
                        self.draw(frame,bbox,np.max(pred[0]),output)
                    elif output == 1 :
                        self.draw(frame,bbox,np.max(pred[0]),output)
                    
            except :
                pass
            cv2.imshow('face detector',frame)
            if cv2.waitKey(1) & 0xFF == ord('q') :
                break
        cap.release()
        cv2.destroyAllWindows()
        

mtcnn = MTCNN()
detector = detection(mtcnn,vgg_model)
detector.run()
          