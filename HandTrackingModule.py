import cv2
import mediapipe as mp
import time
class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.result = None
        self.fingerTips = [4, 8, 12, 16, 20]
    def findHands(self,img, draw=True):
        
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRgb)

        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    def findPositions(self,img, handNo=0, draw=True):
        self.lmList = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw and id == 0:  
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)    
        return self.lmList
    def fingersUp(self):
        fingers = []
        if self.lmList[self.fingerTips[0]][1] < self.lmList[self.fingerTips[0] - 1][1]:  
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if self.lmList[self.fingerTips[id]][2] < self.lmList[self.fingerTips[id] - 2][2]:  
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

def main():
    pTime = 0  
    cap = cv2.VideoCapture(0)
    detector= handDetector()

    while True:
        ret, img = cap.read()
        img=detector.findHands(img)
        lmList = detector.findPositions(img)
        # if len(lmList) != 0:
            # print(lmList[4])
        cTime = time.time() 
        fps = 1 / (cTime - pTime + 1e-6) 
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == 27:  
            break


if __name__ == "__main__":
    main()