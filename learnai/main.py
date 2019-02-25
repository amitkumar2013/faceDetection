from recognize.Face import Face
from model.Create import Create 
import cv2, os, argparse

# Read Faces OR Model from here to build/load for recognizing.
face_data_dir = 'learnai/faces'
face_model_dir = 'learnai/model'
(names, id) = ({}, 0) 

def get_model(algo):
    
    # THE MODEL for RECOGNITION
    if(algo=='lbph'):
        model = cv2.face.LBPHFaceRecognizer_create()
    elif(algo=='eigen'): 
        # use confidence in predict
        # Also create can have PCA and confidence 
        # viz. 10 principal component & 70 confidence_threshold goes like (10,70.0)
        model = cv2.face.EigenFaceRecognizer_create() 
    elif(algo=='fisher'): 
        # robust against large changes in illumination
        model = cv2.face.FisherFaceRecognizer_create() 
    else:
        return None
    
    # Either load model or read if existing
    model_file_path = face_model_dir + '/face_model_' + algo + '.pkl'
    if os.path.isfile(model_file_path):
        model.read(model_file_path)
    else:
        create = Create(face_data_dir, face_model_dir)
        model = create.build_model(algo)
    
    return model
        
def show_image(img):
    model = get_model('fisher')# lbph-1996, eigen-1991 or fisher-1997 (best) SIFT - 1999 SURF - 2006
    # HAAR for face DETECTION
    face = Face(face_model_dir + '/haar/haarcascade_face.xml', face_model_dir + '/haar/haarcascade_eye.xml')
    # RECOGNIZE - if not put it in face_data_dir as unknown
    label, image = face.recognize(img, model, face_data_dir, names)
    cv2.imshow('Capture - Face detection', image)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="This program extracts faces from image or webcam.")
    parser.add_argument("-i", "--img_path", help="image path")
    args = parser.parse_args()

    show_web_cam = True if args.img_path is None else False
        
    # Map of Names with keys as index
    for (subdirs, dirs, files) in os.walk(face_data_dir): 
        for subdir in dirs: 
            names[id] = subdir 
            id += 1

    # Capture image
    if show_web_cam:
        web_cam = cv2.VideoCapture(0)
        # Read frames
        while (web_cam.isOpened()):
            ret, frame = web_cam.read()
            if frame is None:
                print('--(!) No captured frame -- Break!')
                break
            else:
                show_image(frame)
                # Press Q on keyboard to  exit 
                if cv2.waitKey(25) & 0xFF == ord('q'): 
                    break
        # Close the window 
        web_cam.release() 
        # De-allocate any associated memory usage 
        cv2.destroyAllWindows()
    else:
        show_image(cv2.imread(args.img_path))
        cv2.waitKey(0)
        
    
