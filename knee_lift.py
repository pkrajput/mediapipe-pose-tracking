# webcam input
from scipy.spatial.distance import cdist
import cv2
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image
from numpy.linalg import norm

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('your_video.mp4', fourcc, 10.0, size, True)


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
                
        # Recolor image to RGB
        img = cv2.imread('reference.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        image.flags.writeable = False
        img.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
        results_ref = pose.process(img)
    
        # Recolor back to BGR
        image.flags.writeable = True
        img.flags.writeable = True
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        
        landmarks = results.pose_landmarks.landmark
        landmarks_ref = results_ref.pose_landmarks.landmark


        # Get coordinates
        hip_l_r = np.asarray([landmarks_ref[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks_ref[mp_pose.PoseLandmark.LEFT_HIP.value].y])
        knee_l_r = np.asarray([landmarks_ref[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks_ref[mp_pose.PoseLandmark.LEFT_KNEE.value].y])
        ankle_l_r = np.asarray([landmarks_ref[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks_ref[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])
        heel_l_r = np.asarray([landmarks_ref[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks_ref[mp_pose.PoseLandmark.LEFT_HEEL.value].y])

        # Get coordinates
        hip_r_r = np.asarray([landmarks_ref[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks_ref[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
        knee_r_r = np.asarray([landmarks_ref[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks_ref[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])
        ankle_r_r = np.asarray([landmarks_ref[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks_ref[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])
        heel_r_r = np.asarray([landmarks_ref[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks_ref[mp_pose.PoseLandmark.RIGHT_HEEL.value].y])

        # Get coordinates
        hip_l = np.asarray([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
        knee_l = np.asarray([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y])
        ankle_l = np.asarray([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])
        heel_l = np.asarray([landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y])

        # Get coordinates
        hip_r = np.asarray([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
        knee_r = np.asarray([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])
        ankle_r = np.asarray([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])
        heel_r = np.asarray([landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y])
           
            
            
     
        
        lefts = np.stack(hip_l, knee_l, ankle_l, heel_l)
        rights = np.stack(hip_r, knee_r, ankle_r, heel_r)
        dist_matrix = cdist(lefts, rights)
        dist = np.linalg.norm(dist_martix)
        
        
        lefts_r = np.stack(hip_l_r, knee_l_r, ankle_l_r, heel_l_r)
        rights_r = np.stack(hip_r_r, knee_r_r, ankle_r_r, heel_r_r)
        dist_matrix_r = cdist(lefts_r, rights_r)        
        dist_r = np.linalg.norm(dist_martix_r)
        
    
        
        
        
#         def get_distance(a, b):
#             A = np.asarray(a)
#             B = np.asarray(b)
#             return np.linalg.norm(a-b)
            
            


        

        

        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # Rep data
        
        cv2.putText(image, "Distance from perfect = {:.2f}.format(dist-dist_r)", 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        

        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        out.write(image)
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    #out.release()
    cv2.destroyAllWindows()
    
#destroyAllWindows()


