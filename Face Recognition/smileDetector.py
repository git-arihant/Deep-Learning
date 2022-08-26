import cv2

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smileCascade = cv2.CascadeClassifier("haarcascade_smile.xml")

def detect(gray,frame):
    faces = faceCascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
        
        roiGray = gray[y:y+h,x:x+w]
        roiFrame = frame[y:y+h,x:x+w]

        eyes = eyeCascade.detectMultiScale(roiGray,1.1,22)
        smile = smileCascade.detectMultiScale(roiGray,1.7,10)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roiFrame, (ex,ey), (ex+ew,ey+eh), (0,255,0),2)
            
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roiFrame, (sx,sy), (sx+sw,sy+sh), (0,0,255),2)
    return frame

videoCapture = cv2.VideoCapture(0)

while True:
    _,frame = videoCapture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    canvas = detect(gray,frame)
    cv2.imshow('Video', canvas)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
videoCapture.release()
cv2.destroyAllWindows()
