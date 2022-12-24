


import cv2

#import haar feature cascades
# ref haar cascades https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_smile.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

#detector function
def detect(gray_frame, org_frame):
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    for (x, y, w, h) in faces:
        #draw a rectangle on the original frame
        cv2.rectangle(org_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #get region of interest
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = org_frame[y:y+h, x:x+w]

        #use region of interest to detect eyes, cuts down on computation
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3) #apply to grey image, scale 1.1 with min neighbours of 3
        #draw rectangle around eyes when detected.
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22) #apply to grey image, scale 1.7 with min neighbours of 22
        #draw rectangle around eyes when detected.
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
    return org_frame

#Face recognition with webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, org_frame = video_capture.read()
    gray_frame = cv2.cvtColor(org_frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray_frame, org_frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()