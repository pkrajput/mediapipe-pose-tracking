from utils.bodyEmbedder import FullBodyPoseEmbedder
from utils.poseClassifier import PoseClassifier
from utils.poseFilter import EMADictSmoothing
from utils.repCounter import RepetitionCounter

import os
import numpy as np
import tqdm
import cv2
import numpy

from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose


# Specify your video name and target pose class to count the repetitions.
video_path = 'squat.mp4'
class_name='squats_mid'
out_video_path = 'squat-out.mp4'

# Open the video.


video_cap = cv2.VideoCapture(0)

# Get some video parameters to generate output video with classificaiton.

video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initilize tracker, classifier and counter.
# Do that before every video as all of them have state.

# Folder with pose class CSVs. That should be the same folder you using while
# building classifier to output CSVs.
pose_samples_folder = 'data/fitness_poses_csvs_out/'

# Initialize tracker.
pose_tracker = mp_pose.Pose()

# Initialize embedder.
pose_embedder = FullBodyPoseEmbedder()

# Initialize classifier.
# Ceck that you are using the same parameters as during bootstrapping.
pose_classifier = PoseClassifier(
    pose_samples_folder=pose_samples_folder,
    pose_embedder=pose_embedder,
    top_n_by_max_distance=30,
    top_n_by_mean_distance=10)

# # Uncomment to validate target poses used by classifier and find outliers.
# outliers = pose_classifier.find_pose_sample_outliers()
# print('Number of pose sample outliers (consider removing them): ', len(outliers))

# Initialize EMA smoothing.
pose_classification_filter = EMADictSmoothing(
    window_size=10,
    alpha=0.2)

# Initialize counter.
repetition_counter = RepetitionCounter(
    class_name=class_name,
    enter_threshold=6,
    exit_threshold=4)


# Run classification on a video.



# Open output video.
out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (video_width, video_height))

frame_idx = 0
output_frame = None

while True:
  # Get next frame of the video.
  success, input_frame = video_cap.read()
  if not success:
    break

  # Run pose tracker.
  input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
  result = pose_tracker.process(image=input_frame)
  pose_landmarks = result.pose_landmarks

  # Draw pose prediction.
  output_frame = input_frame.copy()
  if pose_landmarks is not None:
    mp_drawing.draw_landmarks(
        image=output_frame,
        landmark_list=pose_landmarks,
        connections=mp_pose.POSE_CONNECTIONS)
  
  if pose_landmarks is not None:
#       # Get landmarks.
    frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
    pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                for lmk in pose_landmarks.landmark], dtype=np.float32)
#       assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

    # Classify the pose on the current frame.
    pose_classification = pose_classifier(pose_landmarks)

    # Smooth classification using EMA.
    pose_classification_filtered = pose_classification_filter(pose_classification)

    # Count repetitions.
    repetitions_count = repetition_counter(pose_classification_filtered)
  else:
    # No pose => no classification on current frame.
    pose_classification = None

    # Still add empty classification to the filter to maintaing correct
    # smoothing for future frames.
    pose_classification_filtered = pose_classification_filter(dict())
    pose_classification_filtered = None

    # Don't update the counter presuming that person is 'frozen'. Just
    # take the latest repetitions count.
    repetitions_count = repetition_counter.n_repeats

  

#     # Draw classification plot and repetition counter.
    """segmented_frames = frameSegmentor(
         frame=output_frame,
         pose_classification=pose_classification,
         pose_classification_filtered=pose_classification_filtered,
         repetitions_count=repetitions_count)"""

    
         
    


#     # Save the output frame.
#     out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))

#     # Show intermediate frames of the video to track progress.
#     if frame_idx % 50 == 0:
#       show_image(output_frame)



  frame_idx += 1
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(output_frame, str(repetitions_count), (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
  cv2.imshow('frame', output_frame)
  if cv2.waitKey(5) & 0xFF == 27:
    break

  
    
  

# Close output video.
out_video.release()

# Release MediaPipe resources.
pose_tracker.close()
