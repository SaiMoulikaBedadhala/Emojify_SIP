from tkinter import *
from PIL import ImageTk,Image
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

face_model = Sequential()
face_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
face_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
face_model.add(MaxPooling2D(pool_size=(2, 2)))
face_model.add(Dropout(0.25))
face_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
face_model.add(MaxPooling2D(pool_size=(2, 2)))
face_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
face_model.add(MaxPooling2D(pool_size=(2, 2)))
face_model.add(Dropout(0.25))
face_model.add(Flatten())
face_model.add(Dense(1024, activation='relu'))
face_model.add(Dropout(0.5))
face_model.add(Dense(7, activation='softmax'))

face_model.load_weights('recognition_model.h5')

facial_dict = {0: "   Angry   ", 
1: "Disgusted", 
2: "  Fearful  ",
3: "   Happy   ",
4: "  Neutral  ", 
5: "    Sad    ", 
6: "Surprised"}

emojis_dict = {0:"/Users/saimoulikabedadhala/Desktop/Internship /Emojis/angry.png",
1:"/Users/saimoulikabedadhala/Desktop/Internship /Emojis/disgusted.png",
2:"/Users/saimoulikabedadhala/Desktop/Internship /Emojis/fearful.png",
3:"/Users/saimoulikabedadhala/Desktop/Internship /Emojis/happy.png",
4:"/Users/saimoulikabedadhala/Desktop/Internship /Emojis/neutral.png",
5:"/Users/saimoulikabedadhala/Desktop/Internship /Emojis/sad.png",
6:"/Users/saimoulikabedadhala/Desktop/Internship /Emojis/surprised.png"}


global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
show_text=[0]

def showFrames():
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cant open the Camera")
    flag,frame=cap.read()
    faces=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    resize_img = frame.resize((500,500))
    last_frame1 = resize_img
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face=face_cascade.detectMultiScale(faces,scaleFactor=1.3,minNeighbors=5)

    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        roi_frame = faces[y:y + h, x:x + w]
        crop_img = np.expand_dims(np.expand_dims(cv2.resize(roi_frame, (48, 48)), -1), 0)
        prediction = face_model.predict(crop_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, facial_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0]=maxindex
    
    if flag:
        # cv2.imshow('Frame',resize_img)
        img=Image.fromarray(last_frame1)
        imgtk=ImageTk.PhotoImage(img)
        label1.imgtk=imgtk
        label1.configure(imgtk)
        label1.after(20,showFrames)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit
    #cap.release()


# def showFrames():
#     cap=cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Cant open the Camera")
#     flag,frame=cap.read()
#     image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#     img=Image.fromarray(image)
#     resize_img = img.resize((500,500))
#     imgtk=ImageTk.PhotoImage(resize_img)
#     label1.imgtk=imgtk
#     label1.configure(image=imgtk)
#     label1.after(20,showFrames)

def Get_Emoji():
    frame2=cv2.imread(emojis_dict[show_text[0]])
    pic2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    img2=Image.fromarray(frame2)
    imgtk2=ImageTk.PhotoImage(image=img2)
    label2.imgtk2=imgtk2
    label3.configure(text=facial_dict[show_text[0]],font=('arial',45,'bold'))
    label2.configure(image=imgtk2)
    label2.after(10, Get_Emoji)


if __name__=="__main__":
    root = Tk()
    root.title("Emojify")
    root.configure(background='black')
    a = Label(root, text='SUMMER INTERNSHIP PROJECT',bg='black')
    a.pack()
    root.geometry("1200x800+10+20")

    # heading = Label(root,bg='black')
    # heading.pack()

    heading2=Label(root,text="Emojify",pady=20, font=('arial',45,'bold'),bg='black',fg='#CDCDCD')#to label the output
    heading2.pack()

    #bd:It represents the border width.
    image = Image.open("bg.jpeg")
    resize_image = image.resize((500,500))
    img = ImageTk.PhotoImage(resize_image)


    label1=Label(root,font=("Times",30,"bold"), bg='red')
    label1.pack(side=LEFT, padx=50,pady=15)


    label2=Label(root,font=("Times",30,"bold"), bg='blue')
    label2.pack(side=RIGHT, padx=50,pady=15)

    label3=Label(root, font=("Times",30,"bold"), bg='green')
    label3.pack(pady=15)

    b=Button(root,text="QUIT",command=root.destroy).pack(side=BOTTOM)
    b1=Button(root,text="CAPTURE",command=showFrames).pack(side=BOTTOM)
   
    root.mainloop()


