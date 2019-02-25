'''
Created on Dec 24, 2018

@author: Amit.30.Kumar

'''
import cv2

class Face(object):
    face_cascade = eye_cascade = None
    (width, height) = (130, 100) 
    
    def __init__(self, haar_face='haarcascade_face.xml', haar_eye='haarcascade_eye.xml'):
        self.face_cascade = cv2.CascadeClassifier(haar_face) 
        self.eye_cascade = cv2.CascadeClassifier(haar_eye) 
    
    def recognize(self, image, model, face_data_dir, names):
        label = 'unknown' 
        # skip if already gray
        gray_image = image if len(image.shape)<3 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast_image = cv2.equalizeHist(gray_image)
        #-- Detect faces 
        #-- options scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        faces = self.face_cascade.detectMultiScale(contrast_image)
        for (x,y,w,h) in faces:
            face_center = (x + w//2, y + h//2)
            image = cv2.ellipse(image, face_center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
            faceROI = contrast_image[y:y+h,x:x+w]
            # Resizing IS NOT NEEDED for Local Binary Patterns Histograms rather for Eigen & Fisher
            face_resize = cv2.resize(faceROI, (self.width, self.height))
            # Predicting Faces
            if model is not None: # Catering for 1st time
                imgId, confidence = model.predict(face_resize)
                #print(imgId , confidence)
                if (confidence > 700):
                    cv2.putText(image, 
                                '% s - %.0f' %(names[imgId], confidence), 
                                (x-10, y-10), 
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                else: 
                    for x in range(30):
                        cv2.imwrite('% s/% s.png' % (face_data_dir+'/unknown', x), face_resize)
                 
            #-- In each face, detect eyes
            eyes = self.eye_cascade.detectMultiScale(faceROI)
            for (x2,y2,w2,h2) in eyes:
                eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                radius = int(round((w2 + h2)*0.25))
                image = cv2.circle(image, eye_center, radius, (255, 0, 0 ), 4)
        return label, image
