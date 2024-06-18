import cv2
import os

def main():
    cam=cv2.VideoCapture(0)
    cv2.namedWindow("Image")
    while True:
        ret,frame=cam.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cwd = os.getcwd()
        paths= os.path.join(cwd, 'images/')
        if not ret:
            print("ERROR")
        cv2.imshow("Image",gray)
        img_name="image.jpg"
        k=cv2.waitKey(1)
        if k%256==27:
            cv2.imwrite(os.path.join(paths,img_name),gray)
            break
    cam.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()