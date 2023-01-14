# webcam input

from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image
import sys
import tqdm
import math



def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle



# Curl counter variables
stage = "Let's go"

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('your_video.mp4', fourcc, 10.0, size, True)


# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# out = cv2.VideoWriter('output_video_.mp4', fourcc, 24, size)

frame_count = 0 

avg_hips_x = []
avg_hips_y = []
avg_shoulders_x = []
avg_shoulders_y = []
shoulder_right = []
hip_avg = []
shoulder_avg =[]

drawLine = False


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        # print(frame_count)
        stage = "let's go"
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            
            # Get left coordinates
            shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]


            # Get right coordinates
            shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]


            #average points for shoulder and hips -> Prateek's
            shoulder_avg = [(shoulder_left[0]+shoulder_right[0])/2, (shoulder_left[1] + shoulder_right[1])/2]
            hip_avg = [(hip_left[0]+hip_right[0])/2, (hip_left[1] + hip_right[1])/2]
            shoulder_avg = np.multiply(shoulder_avg, [width,height]).astype(int)
            hip_avg = np.multiply(hip_avg, [width,height]).astype(int)
#             ref_line = []
#             ref_line.append(tuple(shoulder_avg))
#             ref_line.append(tuple(hip_avg))


            #Get the middle 

            if frame_count in [*range(110, 131, 1)]:

                cv2.putText(image, str(f'Collecting points!!!'), (20, 400), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                mid_hips_point_x = abs(hip_right[0] + hip_left[0]) / 2    # x, y coordinates for mid hips
                mid_hips_point_y = abs(hip_right[1] + hip_left[1]) / 2
                mid_shoulders_point_x = abs(shoulder_right[0] + shoulder_left[0]) / 2     # x, y coordinates for mid shoulders
                mid_shoulders_point_y = abs(shoulder_right[1] + shoulder_left[1]) / 2
                avg_hips_x.append(mid_hips_point_x)
                avg_hips_y.append(mid_hips_point_y)
                avg_shoulders_x.append(mid_shoulders_point_x)
                avg_shoulders_y.append(mid_shoulders_point_y)

            
            if len(avg_hips_x) >= 20 or len(avg_hips_y) >= 20:
                shoulders_point_x = sum(avg_hips_x) / len(avg_hips_x)
                shoulders_point_y = sum(avg_hips_y) / len(avg_hips_y)
                hips_point_x = sum(avg_shoulders_x) / len(avg_shoulders_x)
                hips_point_y = sum(avg_shoulders_y) / len(avg_shoulders_y)
                drawLine = True


            # Calculate angle
            angle = calculate_angle(shoulder_left, elbow_left, wrist_left)
            
            angle_knee = calculate_angle(hip_left, knee_left, ankle_left) #Knee joint angle
            
            angle_hip = calculate_angle(shoulder_left, hip_left, knee_left)
#             hip_angle = 180-angle_hip
#             knee_angle = 180-angle_knee    

            
            # Visualize angle
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(elbow_left, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                       
                
            cv2.putText(image, str(angle_knee), 
                           tuple(np.multiply(knee_left, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (79, 121, 66), 2, cv2.LINE_AA
                                )
            
            cv2.putText(image, str(angle_hip), 
                           tuple(np.multiply(hip_left, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )


            
            if angle_knee < 130 and angle_hip < 130:
                stage = "That's good!"
            if angle_knee < 150 and angle_knee >= 130 and angle_hip < 150 and angle_hip >= 130:
                stage = "Lets go down a bit more"
            # entering and proceeding in squat
#             if angle_knee > 40 and angle_knee < 90 and angle_hip <60:
#                 stage = "You are in partial squat"
#             if angle_knee >= 90 and angle_knee < 105 and stage =='You are in partial squat' and angle_hip < 45:
#                 stage="go further down"
#             if angle_knee >=105 and angle_knee <= 120 and stage == 'You are in partial squat' and angle_hip < 30:
#                 stage = "Thats a squat"
#                 counter +=1
#                 print(stage)
        except:
            pass
        
        # Render squat counter
        # Setup status box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

        
        # Stage data
        cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )   

        start_point = tuple(shoulder_avg)
        end_point = tuple(hip_avg)
        color = (0, 255, 0)
        thickness = 9
        cv2.line(image, start_point, end_point, color, thickness)


        if drawLine:
            shoulder = [shoulders_point_x, shoulders_point_y]
            hip = [hips_point_x, hips_point_y]
            start_point = tuple(np.multiply(shoulder, [width, height]).astype(int))
            end_point = tuple(np.multiply(hip, [width, height]).astype(int))
            start_point_x, start_point_y = start_point
            start_point = (start_point_x, start_point_y + 120)
            color = (255, 255, 255)
            thickness = 9
            cv2.line(image, start_point, end_point, color, thickness)                                     
        
        out.write(image)
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    #out.release()
    cv2.destroyAllWindows()
    
#destroyAllWindows()

