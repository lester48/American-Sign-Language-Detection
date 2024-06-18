import cv2
import mediapipe as mp
import time
import numpy as np
import os
import infer
from PIL import Image

class handDetector():
    def __init__(self, mode = False, maxHands = 2, modelComplexity=1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelComplex = modelComplexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self,img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        coordinates = {}
        if self.results.multi_hand_landmarks:

            image_height, image_width, channels = img.shape
            for hand_landmark in self.results.multi_hand_landmarks:
                x = [landmark.x for landmark in hand_landmark.landmark]
                y = [landmark.y for landmark in hand_landmark.landmark]
                
                center = np.array([np.mean(x)*image_width, np.mean(y)*image_height]).astype('int32') 
                cv2.rectangle(img, (center[0]-150,center[1]-200), (center[0]+150,center[1]+220), (255,0,0), 1)
                coordinates = {'X': center[0]-150, 'Y': center[1]-200, 'X+W': center[0]+150,'Y+H': center[1]+220}
        return img, coordinates

    def findPosition(self, img, handNo = 0):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
        return lmlist

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        label = ""
        success, img = cap.read()
        img, coordinates = detector.findHands(img)
        lmlist = detector.findPosition(img)
        pp=img
        if len(lmlist) != 0:
            print(lmlist[4])

            try:
                cropped_image = img[coordinates['Y']:coordinates['Y+H'], coordinates['X']:coordinates['X+W']]
                pp=cropped_image
                cropped_image = Image.fromarray(cropped_image)
                
                label = infer.predict(cropped_image)
                cv2.putText(img, str(label), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                print(label)
                
            except:
                print(coordinates)
        k=cv2.waitKey(1)
        #print("BB",k)
        if k%256==27:
            cv2.imwrite("Save.jpg",pp)
            print("AA")
            break
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime


        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()