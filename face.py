import cv2
from  tkinter import *
from PIL import ImageTk, Image

def showFrames():
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap=cv2.VideoCapture(0)
    while True:
        flag,frame=cap.read()
        faces=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face=face_cascade.detectMultiScale(faces,scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)
        for (x,y,w,h) in face:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        if flag:
            last_frame1 = frame.copy()
            pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB) #to store the image
            img=Image.fromarray(pic)
            resize_img=img.resize((500,500))
            imgtk=ImageTk.PhotoImage(resize_img)
            label1.imgtk=imgtk
            label1.configure(image=imgtk)
            label1.after(0,showFrames)
        elif flag is None:
            print ("Error!")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    root = Tk()
    root.title("Emojify")
    root.configure(background='black')
    root.geometry("1200x800+10+20")
    heading2=Label(root,text="Emojify",pady=20, font=('arial',45,'bold'),bg='black',fg='#CDCDCD')#to label the output
    heading2.pack()
    label1=Label(root,font=("Times",30,"bold"), bg='white')
    label1.pack(padx=50,pady=200)
    b=Button(root,text="QUIT",command=root.destroy).pack(side=BOTTOM)
    #b1=Button(root,text="CAPTURE",command=showFrames).pack(side=BOTTOM)
    showFrames()
    root.mainloop()
    #showFrames()