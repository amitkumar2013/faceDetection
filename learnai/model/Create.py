'''
Created on Dec 24, 2018

@author: Amit.30.Kumar
'''
import os, numpy, cv2

class Create(object):

    face_data_dir = face_model_dir = None
    
    def __init__(self, face_data_dir, face_model_dir):
        self.face_data_dir = face_data_dir
        self.face_model_dir = face_model_dir
        
    def build_model(self, algo):
        # Create a list of images and a list of corresponding labels 
        (images, name_labels, id) = ([], [], 0) 
        for (subdirs, dirs, files) in os.walk(self.face_data_dir): 
            for subdir in dirs: 
                subjectpath = os.path.join(self.face_data_dir, subdir) 
                for filename in os.listdir(subjectpath): 
                    path = subjectpath + '/' + filename                     
                    images.append(cv2.imread(path, 0)) 
                    name_labels.append(int(id)) 
                id += 1        
        (images, name_labels) = [numpy.array(lis) for lis in [images, name_labels]] 

        if(algo=='lbph'):
            model = cv2.face.LBPHFaceRecognizer_create()
        elif(algo=='eigen'): 
            # 10 principal component & 70 confidence_threshold goes like (10,70.0)
            model = cv2.face.EigenFaceRecognizer_create() 
        elif(algo=='fisher'): 
            # robust against large changes in illumination
            model = cv2.face.FisherFaceRecognizer_create() 
        else:
            return None # Try SIFT & SURF
        model.train(images, name_labels) # fit has more controls on iterations
        #gender_model = model.train(images, gender_labels) # fit has more controls on iterations
        model.save(self.face_model_dir + '/face_model_' + algo + '.pkl')

        return model