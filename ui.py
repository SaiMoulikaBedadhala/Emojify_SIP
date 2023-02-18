from tkinter import *
from PIL import ImageTk, Image
import cv2

global last_frame1
        
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
    
    if flag:
        # cv2.imshow('Frame',resize_img)
        img=Image.fromarray(resize_img)
        imgtk=ImageTk.PhotoImage(img)
        label1.imgtk=imgtk
        label1.configure(imgtk)
        label1.after(20,showFrames)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit
    cap.release()


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


    label2=Label(root,image=img,font=("Times",30,"bold"), bg='blue')
    label2.pack(side=RIGHT, padx=50,pady=15)

    # label3=Label(root, text="Label 3", font=("Times",30,"bold"), bg='green')
    # label3.pack(pady=15)

    b=Button(root,text="QUIT",command=root.destroy).pack(side=BOTTOM)
    b1=Button(root,text="CAPTURE",command=showFrames).pack(side=BOTTOM)
   
    root.mainloop()


