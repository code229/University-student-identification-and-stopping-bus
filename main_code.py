import tensorflow as tf
import tensorflow_hub as hub
import cv2
import math
import time
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from matplotlib import pyplot as plt
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']
model1 = load_model('my_trained_model3.h5')
def gradient(pt1, pt2):
    if (pt2[0] - pt1[0] > 0): return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    return -90
def printt(img, x, y, st):
    cv2.putText(img, st, (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

def saveImages(img, x, y, w, h):
    # cv2.destroyWindow('face')
    faces = img[y:y + h, x:x + w]
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:

            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(faces, cv2.COLOR_BGR2RGB))

            # Draw face detections of each face.
        if not results.detections:
            print("not found")
            return 0
    faces=cv2.resize(faces,(150,150),interpolation = cv2.INTER_NEAREST)

    cv2.imwrite('output.jpg',faces)
    human=image.load_img('output.jpg',target_size=(150,150))

    human = image.img_to_array(human)
    human = np.expand_dims(human, axis=0)
    human=human/255
    x1 = model1.predict(human)
    classes = np.argmax(x1, axis=1)
    np_prediction = np.amax(x1)
    val = int(np_prediction * 100)
    cv2.putText(frame, str(int(val)), (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 0), 1)
    if (classes == 0 and val >= 70):
        printt(img, x, y, "varsity")
    else:
        printt(img, x, y, "outsider")
    cv2.imshow("face", faces)


def getAngle(pointist):
    pt1, pt2, pt3 = pointist
    m1 = gradient(pt1, pt2)
    m2 = gradient(pt1, pt3)
    angR = math.atan((m2 - m1) / (1 + (m2 * m1)))
    angD = round(math.degrees(angR))
    return angD
# def check_valid(x, y, x1, y1,h,w):
#     if (x < x1 and y < y1 and (x <= w and y <= h and x1 <= w and y1 < h) and (
#             x > 0 and x1 > 0 and y > 0 and y1 > 0)): return True
#     return False
def check_4000(x,y,h,w):
    if(x!=-4000 and y!=4000 and x<=w and y<=h): return True
    return False
def check_valid(x, y, x1, y1,h,w):
    if (x < x1 and y < y1 and (x <= w and y <= h and x1 <= w and y1 < h) and (
            x > 0 and x1 > 0 and y > 0 and y1 > 0)): return True
    return False
def draw_keypoints(frame, keypoints, confidence_threshold):
    # print("okd")
    y, x, c = frame.shape
    cv2.imshow("frame",frame)
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    print(y,x)
    cnt = 0
    ct = 0
    lmd = []
    for i in range(17):
        lmd.append([-4000, -4000])
    #print(lmd)
    for kp in shaped:
        ky, kx, kp_conf = kp
        #print([ct, "   point ", kx, ky, kp_conf])

        if kp_conf > confidence_threshold:
            lmd[ct][0] = int(kx)
            lmd[ct][1] = int(ky)
            if(int(kx)>x or int(ky)>y): print("found")
            #cv2.putText(frame, str(ct), (int(kx), int(ky)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
            cnt += 1
        ct += 1
        #cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)
    #print(lmd)
    if (lmd[5][0] != -4000 and lmd[5][1] != -4000 and lmd[9][0] != -4000 and lmd[9][1] != -4000 and lmd[3][0] != -4000 and
            lmd[3][1] != -4000 and lmd[4][0] != -4000 and lmd[4][1] != -4000 and
            check_4000(lmd[5][0],lmd[5][1],y,x) and check_4000(lmd[9][0] ,lmd[9][1],y,x)):
        diff = int(lmd[3][0] - lmd[4][0])
        dx = int(lmd[4][0])
        dy = max(0, int(lmd[4][1]) - int(diff))
        dx1 = int(lmd[3][0] + (diff * 0.15))
        yt = int(lmd[3][1] + (diff * 0.30))
        dy2 = int(min(y, yt))

        pointlist=[]
        pointlist.append([lmd[5][0],lmd[5][1]])
        pointlist.append([lmd[9][0],lmd[5][1]])
        pointlist.append([lmd[9][0], lmd[9][1]])
        angle=getAngle(pointlist)
        if -90 <= angle < 40 and lmd[5][0]<=lmd[9][0] and check_valid(dx, dy, dx1, dy2,y,x):
            #cv2.rectangle(frame, (dx, dy), (dx1, dy2), (0, 0, 255), 1)
            img=frame.copy()
            saveImages(frame, dx, dy, abs(dx1 - dx), abs(dy2 - dy))
            cv2.rectangle(frame, (dx, dy), (dx1, dy2), (0, 0, 255), 1)
        #cv2.putText(frame, str(int(getAngle(pointlist))), (lmd[5][0],lmd[5][1]), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)





def loop_through_people(frame, keypoints_with_scores, confidence_threshold):
    for person in keypoints_with_scores:
        draw_keypoints(frame, person, confidence_threshold)





cap = cv2.VideoCapture(0)
pTime = 0
while cap.isOpened():
    ret, frame = cap.read()

    # Resize image

    img = frame.copy()
    h,w,c=frame.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384, 640)
    input_img = tf.cast(img, dtype=tf.int32)
    #print(input_img.shape)
    # Detection section
    results = movenet(input_img)
    #print("1")
    keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))
    #print("2")
    #
    # # Render keypoints
    #print()
    #frame=cv2.resize(frame,(h1,256))
    loop_through_people(frame, keypoints_with_scores,  0.1)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("frame",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()