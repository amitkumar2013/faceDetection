import cv2, time, glob

class Faces(object):

    face_cascade = cv2.CascadeClassifier("learnai/model/haar/haarcascade_face.xml")
    facedict = {}
    image_size = (350,350)
    
    def __init__(self):
        return
    
    #To crop face in an image
    def crop_face(self, clahe_image, face):
        for (x, y, w, h) in face:
            faceslice = clahe_image[y:y+h, x:x+w]
            faceslice = cv2.resize(faceslice, self.image_size)
        self.facedict["face%s" %(len(self.facedict)+1)] = faceslice
        return faceslice

    def build_set(self, emotions):
        for i in range(0, len(emotions)):
            self.save_face(emotions[i])
        print("Great,You are Done!" )
        cv2.destroyWindow("preview")
        cv2.destroyWindow("webcam")


    #To save a face in a particular folder
    def save_face(self, emotion):
        print("\n\nplease look " + emotion)
        #To create timer to give time to read what emotion to express
        for i in range(0,5):
            print(5-i)
            time.sleep(1)
        #To grab 50 images for each emotion of each person and populate facedict
        while len(self.facedict.keys()) < 11: 
            self.open_webcamframe()
        #To save contents of dictionary to files
        for x in self.facedict.keys(): 
            cv2.imwrite(
                "learnai/emotions/cam/data/%s/%s.jpg" %(
                    emotion,  
                    len(glob.glob("learnai/emotions/cam/data/%s/*" %emotion))
                ), self.facedict[x])
        self.facedict.clear() #clear dictionary so that the next emotion can be stored

    def open_webcamframe(self):
        while True:
            if vc.isOpened(): # try to get the first frame
                rval, frame = vc.read()
            else:
                rval = False
            cv2.imshow("preview", frame)
            key = cv2.waitKey(40)
            if key == 27: # exit on ESC
                break
            if key == 32: # Space bar to continue
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # CLAHE -  Contrast Limited Adaptive Histogram Equalization
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                clahe_image = clahe.apply(gray_image)
                
                #To run classifier on frame
                face = self.face_cascade.detectMultiScale(clahe_image, 
                                                     scaleFactor=1.1, 
                                                     minNeighbors=15, 
                                                     minSize=(10, 10), 
                                                     flags=cv2.CASCADE_SCALE_IMAGE)
                #To draw rectangle around detected faces
                for (x, y, w, h) in face: 
                    #draw it on "frame", (coordinates), (size), (RGB color), thickness 2    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) 
                #Use simple check if one face is detected, or multiple (measurement error unless multiple persons on image)
                if len(face) == 1: 
                    faceslice = self.crop_face(clahe_image, face)
                    cv2.imshow("webcam", frame)
                    #slice face from image
                    return faceslice
                else:
                    print("no/multiple faces detected, passing over frame")
                    cv2.destroyWindow("preview")
                    cv2.destroyWindow("webcam")

    def cnn_faces.save_bottlebeck_features(self):
        #Function to compute VGG-16 CNN for image feature extraction.
        train_target = []
        
        datagen = ImageDataGenerator(rescale=1. / 255)
        # build the VGG16 network
        model = applications.VGG16(include_top=False,weights='imagenet')
        generator_train = datagen.flow_from_directory(
                                    train_data_dir,
                                    target_size=self.image_size,
                                    batch_size=batch_size,
                                    class_mode=None,
                                    shuffle=False)
        
        for i in generator_train.filenames:
            train_target.append(i[:])

if __name__ == '__main__':
    
    emotions =["anger","disgust","fear","happy","neutral","sad","surprise"]
    first_train = False
    if first_train:
        # Create a window and then load images to it
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(0)
        cnn_faces = Faces() 
        cnn_faces.build_set(emotions)
    else:
        top_model_weights_path = 'bottleneck_fc_model.h5'
        train_data_dir = 'dataset'
        nb_train_samples = 1011
        epochs = 50
        batch_size = 1
        bottleneck_features_train = model.predict_generator(generator_train, nb_train_samples // batch_size)
        bottleneck_features_train = bottleneck_features_train.reshape(1011,51200)
        np.save(open('data_features.npy', 'wb'), bottleneck_features_train)
        np.save(open('data_labels.npy', 'wb'), np.array(train_target))
        cnn_faces.save_bottlebeck_features()
        
