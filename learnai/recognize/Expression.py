'''
Created on Dec 29, 2018

@author: Amit.30.Kumar
'''
from shutil import copyfile
import cv2, glob, random, numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

faceDet = cv2.CascadeClassifier("learnai/model/haar/haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("learnai/model/haar/haarcascade_face.xml")
faceDet_three = cv2.CascadeClassifier("learnai/model/haar/haarcascade_face1.xml")
faceDet_four = cv2.CascadeClassifier("learnai/model/haar/haarcascade_face2.xml")
#Define emotions
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] 
fishface = cv2.face.FisherFaceRecognizer_create() #Initialize fisher face classifier

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("learnai/emotions/data/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            training_data.append(gray) 
            training_labels.append(emotions.index(emotion))
        for item in prediction: #repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))
    return training_data, training_labels, prediction_data, prediction_labels

def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    print("size of fisher face training set is:", len(training_labels), "images")
    fishface.train(training_data, np.asarray(training_labels))
    print("predicting classification set")
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        if pred == prediction_labels[cnt]:
            print('.', end='')
            correct += 1
            cnt += 1
        else:
            print('x', end='')
            incorrect += 1
            cnt += 1
    return ((100*correct)/(correct + incorrect))

def detect_faces(emotion):
    #Get list of all images with emotion
    files = glob.glob("learnai/emotions/sorted/%s/*" %emotion) 
    filenumber = 0
    for f in files:
        print('.', end='')
        frame = cv2.imread(f) #Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
        #Detect face using 4 different classifiers
        #Go over detected faces, stop at first detected face, return empty if no face.
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(face) == 1:
            facefeatures = face
        else:
            face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            if len(face_two) == 1:
                facefeatures = face_two
            else:
                face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
                if len(face_three) == 1:
                    facefeatures = face_three
                else:
                    face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
                    if len(face_four) == 1:
                        facefeatures = face_four
                    else:
                        facefeatures = ""
        #Cut and save face
        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            gray = gray[y:y+h, x:x+w] #Cut the frame to size
            try:
                out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
                cv2.imwrite("learnai/emotions/data/%s/%s.jpg" %(emotion, filenumber), out) #Write image
            except:
               pass #If error, pass file
        filenumber += 1 #Increment image number    

def extract_data():
    #Define emotion order
    emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
    #Returns a list of all folders with participant numbers 
    participants = glob.glob("learnai/emotions/emotion/*") 
    for x in participants:
        #store current participant number - last four alphabets
        part = "%s" %x[-4:] 
        #Store list of sessions for current participant
        for sessions in glob.glob("%s\\*" %x): 
            for files in glob.glob("%s\\*" %sessions):
                current_session = files[25:-30]
                file = open(files, 'r')
                #emotions are encoded as a float, readline as float, then convert to integer.
                emotion = int(float(file.readline()))
                #get path for last image in sequence, which contains the emotion
                sourcefile_emotion = glob.glob("learnai/emotions/image/%s/%s/*" %(part, current_session))[-1]
                destination_emotion = "learnai/emotions/sorted/%s/%s" %(emotions[emotion], sourcefile_emotion[27:])
                #do same for neutral image 
                sourcefile_neutral = glob.glob("learnai/emotions/image/%s/%s/*" %(part, current_session))[0]
                #Generate path to put neutral image 
                destination_neutral = "learnai/emotions/sorted/neutral/%s" %sourcefile_neutral[27:] 
                copyfile(sourcefile_neutral, destination_neutral) #Copy file
                copyfile(sourcefile_emotion, destination_emotion) #Copy file
                
if __name__ == '__main__':
    first = False
    if first:
        extract_data()
        for emotion in emotions:
            detect_faces(emotion) 
        #Call function only once - remove duplicates esp from neutral
    else:
        metascore = []
        for i in range(0,5):# run 10 times - each time a random 80% set
            correct = run_recognizer()
            print(" -- ", correct, " percent!")
            metascore.append(correct)
        print("\n\nEnd score:", np.mean(metascore), " percent!")
        
    learningCurve = False
    if learningCurve:
        # VALIDATION CURVE 
        # LEARNING CURVE - tells if there is any benefit from adding more training data
        # BIAS & VARIANCE - low - for variance use more training data.
        plt.figure()
        plt.title('Learning Curves')
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.grid()
        
        # The fun
        train_sizes, train_scores, test_scores = learning_curve(
                fishface, data, emotions, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        plt.plot(data, train_scores_mean, marker='x-', color="r", label="Training score")
        plt.plot(data, test_scores_mean, marker='x-', color="g", label="Cross-validation score")
        
        plt.legend(loc="best")
        plt.show()

    