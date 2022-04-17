import cv2
import mediapipe as mp
import os

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For static images:
cnt=0
def check_valid(x, y, x1, y1,h,w):
    if (x < x1 and y < y1 and (x <= w and y <= h and x1 <= w and y1 < h) and (
            x > 0 and x1 > 0 and y > 0 and y1 > 0)): return True
    return False
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
for filename in os.listdir('./'):
    if filename.endswith(".jpg"):
        image = cv2.imread(filename)
        hh,ww,c=image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
        )

        if(len(faces)==0): continue
        for (x, y, w, h) in faces:
            if(check_valid(x,y,x + w, y + h,hh,ww)):
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                rt=image[y:y+h,x:x+w]
                # path='D://projects//varsity_student_identification//extract//output_images//check//'
                # rt=cv2.resize(rt,(150,150),interpolation = cv2.INTER_AREA)
                # cv2.imwrite(str(path)+'out'+str(cnt)+'.jpg',rt)
                cv2.imshow("itmg", rt)
                cnt+=1
        cv2.imshow("img",image)
        # FACIAL_KEYPOINTS=mp_face_detection.FaceKeyPoint
        # with mp_face_detection.FaceDetection(
        #     model_selection=1, min_detection_confidence=0.2) as face_detection:
        #     image = cv2.imread(filename)
        #     if image is None:
        #         print('Wrong path:')
        #     #image=cv2.resize(image,(1000,700))
        #     # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        #     results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #
        #     # Draw face detections of each face.
        #
        #     annotated_image = image.copy()
        #     for detection in results.detections:
        #
        #         keypoints = {}
        #
        #         for kp in FACIAL_KEYPOINTS:  # iterate over each landmarks and get from results
        #             keypoint = mp.solutions.face_detection.get_key_point(detection, kp)
        #             # convert to pixel coordinates and add to dictionary
        #             keypoints[kp.name] = {"x": int(keypoint.x * width), "y": int(keypoint.y * height)}
        #
        #         # bbox data
        #         bbox = detection.location_data.relative_bounding_box
        #
        #         x1=int(bbox.xmin * width)
        #         y1=int(bbox.ymin * height)
        #         x2=int(bbox.width * width + bbox.xmin * width)
        #         y2=int(bbox.height * height + bbox.ymin * height)
        #
        #         cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),3)
        #         # add score, bbox and keypoints to detection_results
        #
        #         print('Nose tip:')
        #         print(mp_face_detection.FaceKeyPoint)
        #         print(mp_face_detection.get_key_point(
        #           detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
        #         mp_drawing.draw_detection(annotated_image, detection)
        #         cv2.imshow("image", annotated_image)
        cv2.waitKey(1000)
        #         cv2.imwrite('/input_images/annotated_image'  +str(cnt)+ '.jpg', annotated_image)
        #     cnt+=1


    else:continue
