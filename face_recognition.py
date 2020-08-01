import keras
import numpy as np
import cv2
import time
import keyboard


#utility functions
def is_pressed(key=""):
    if(keyboard.is_pressed(key)):
        time.sleep(0.05)
        return 1
    else:
        time.sleep(0.05)
        return 0

def test_label(image,label,label_type="corners"): #label_type can be corners or points
    image=cv2.UMat(np.array(image,dtype=np.uint8))
    label=np.array(label,dtype=np.int32)
    if(label_type=="corners"):
        if(len(label)==2):
            xmin,ymin,xmax,ymax=label.ravel()
            label=np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]],dtype=np.int32)
        label=label.reshape([-1,2])
        image=cv2.polylines(image,[label],True,(0,0,255),thickness=2)
    elif(label_type=="points"):
        label=label.reshape(-1,2).tolist()
        for x,y in label : cv2.circle(image,(x,y),radius=3,color=(0,0,255),thickness=-1)
    else:
        return None

    return image

def display_image(image,win_name="image",show_time=0,destroy=True):
    image=cv2.cvtColor(cv2.UMat(image),cv2.COLOR_RGB2BGR)
    cv2.imshow(win_name,image)
    if(show_time>0) : cv2.waitKey(show_time)
    if(destroy) : cv2.destroyWindow(win_name)

def write_on_image(image,text,bottom_left,font=cv2.FONT_HERSHEY_SIMPLEX,font_scale=1,color=(0,0,255),thickness=2,line_type=cv2.LINE_AA):
    cv2.putText(image,text,bottom_left,font,font_scale,color,thickness,line_type)
    return image

def save_numpy(array,save_name,compressed=True):
    if(compressed):
        with open(save_name,'wb') as file : np.savez_compressed(file,array)
    else:
        with open(save_name,'wb') as file : np.save(file,array)

def load_numpy(save_name,compressed=True):
    with open(save_name,'rb') as file : array=np.load(file)
    if(compressed) : return array
    else : return array





#helper fucntions
def transform(images):
    images=images.astype(np.float32)
    temp=[]
    for image in images : temp.append(cv2.resize(image,(160,160)).tolist())
    images=np.array(temp,dtype=np.float32)

    images=(images-images.mean())/(images.std()+1.0e-8)
    return images

def load_facenet(save_name):
    facenet=keras.models.load_model(save_name)
    return facenet

def detect_face(image):
    detector=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image=cv2.cvtColor(cv2.UMat(image),cv2.COLOR_RGB2GRAY)
    result=detector.detectMultiScale(image,scaleFactor=1.5,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    if(len(result)==0) : xmin,ymin,width,height=0,0,0,0
    else : xmin,ymin,width,height=result[0]
    xmin,ymin,xmax,ymax=abs(xmin),abs(ymin),abs(xmin)+width,abs(ymin)+height
    return xmin,ymin,xmax,ymax

def get_encodings(facenet,faces):
    encodings=facenet.predict(faces)
    return encodings

def recognize(face,faces,ids,facenet,threshold):
    encoding=np.squeeze(get_encodings(facenet,face[None,:,:,:]))
    faces_num=faces.shape[0]
    faces=faces.reshape(faces_num,-1)
    encoding=encoding.reshape(1,-1)
    output=((faces-encoding)**2).sum(axis=1)**0.5

    if(np.min(output)<threshold):
        idx=np.argmin(output)
        id=ids[idx]
    else:
        id="unknown"
    return id


def record_faces_and_ids(facenet_input_shape=(160,160),webcam_display_shape=(500,500)):

    timer=3
    save_key="s"
    faces=[]
    ids=[]

    cam=cv2.VideoCapture(0)
    if not cam.isOpened() : raise IOError("Cannot open webcam")
    print("\nRecording faces")
    print("Press esc to exit.")
    print("Press ",save_key," to save a face.")
    for i in range(1,timer+1) : print("starting in ",timer+1-i,end="\r\r");time.sleep(1.0)
    print("\n")

    while True:
        _,frame=cam.read()
        frame=cv2.resize(frame,webcam_display_shape)
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).astype(np.uint8)
        

        xmin,ymin,xmax,ymax=detect_face(frame)
        label=np.array([[xmin,ymin],[xmax,ymax]],dtype=np.int32)

        if(is_pressed(save_key)):
            if((xmin==0 and ymin==0) and (xmax==0 and ymax==0)):
                print("No face found.")
                time.sleep(0.05)
            else:
                face=frame[ymin:ymax,xmin:xmax,:]
                face=np.squeeze(transform(face[None,:,:,:]))
                id=input("Enter the id:")
                faces.append(face)
                ids.append(id)
                face=write_on_image(face,id,(70,120))
                display_image(face,id,show_time=3000)
                
                print("saved")
                time.sleep(0.05)

        frame=test_label(frame,label,"corners")
        display_image(frame,"WebCam",1,False)
        if(cv2.waitKey(1)==27) : break

    cam.release()
    cv2.destroyAllWindows()
    faces=np.array(faces,dtype=np.uint8)
    
    return faces,ids

def recognition(faces,ids,facenet,facenet_input_shape,webcam_display_shape,threshold):
    timer=3

    cam=cv2.VideoCapture(0)
    if not cam.isOpened() : raise IOError("Cannot open webcam")
    print("\nRecognizing faces.")
    print("Press esc to exit.")
    for i in range(1,timer+1) : print("starting in ",timer+1-i,end="\r\r");time.sleep(1.0)
    print("\n")

    while True:
        _,frame=cam.read()
        frame=cv2.resize(frame,webcam_display_shape)
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).astype(np.uint8)
        

        xmin,ymin,xmax,ymax=detect_face(frame)

        if((xmin==0 and ymin==0) and (xmax==0 and ymax==0)):
            id="No face found"
            time.sleep(0.05)
        else:
            face=frame[ymin:ymax,xmin:xmax,:]
            face=np.squeeze(transform(face[None,:,:,:]))
            id=recognize(face,faces,ids,facenet,threshold)
            time.sleep(0.05)

        frame=write_on_image(frame,id,(0,160))

        label=np.array([[xmin,ymin],[xmax,ymax]],dtype=np.int32)
        frame=test_label(frame,label,"corners")
        display_image(frame,"WebCam",1,False)
        if(cv2.waitKey(1)==27) : break

    cam.release()
    cv2.destroyAllWindows()






def main():

    #the name of the facenet model along with the path to its directory
    facenet_save_name=''

    #the name of the numpy array with .npy extension which stores the facial information (facial features) of the faces in its database
    faces_save_name=''

    #the name of the text file with .txt extension which stores the ids of the faces in its database
    ids_save_name=''

    #do not change this
    facenet_input_shape=(160,160)

    #the size of the webcam window
    webcam_display_shape=(500,500)

    #the threshold dis-similarity (out of 100) for face matching with faces in the database.If a face is dis-similar by more than 15% (in this ex) with a face in the database then the faces are not similar.
    threshold=15

    #loading the facenet model
    facenet=load_facenet(facenet_save_name)

    #uncomment to record and store new faces in the directory specified in the ids_save_name and faces_save_name variables.
    '''faces,ids=record_faces_and_ids(facenet_input_shape,webcam_display_shape)
    faces,ids=faces.astype(np.float32)," ".join(ids)
    faces=get_encodings(facenet,faces)
    save_numpy(faces,faces_save_name,False)
    with open(ids_save_name,"w") as file : file.write(ids)'''


    #loading the database information
    faces=load_numpy(faces_save_name,False)
    with open(ids_save_name,"r") as file : ids=list(file.read().split(' '))

    #starting the recognition task
    recognition(faces,ids,facenet,facenet_input_shape,webcam_display_shape,threshold)

if(__name__=="__main__") : main()
