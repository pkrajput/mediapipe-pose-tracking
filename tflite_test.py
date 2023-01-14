from asyncio import open_unix_connection
import tflite_runtime.interpreter as tflite
import cv2
import numpy as np

# # Load the TFLite model and allocate tensors.
poseDetector = tflite.Interpreter("/home/prateek/II/mediapipe/mp_env/lib/python3.8/site-packages/mediapipe/modules/pose_detection/pose_detection.tflite")
poseDetector.allocate_tensors()


poseLandmark = tflite.Interpreter("/home/prateek/II/mediapipe/mp_env/lib/python3.8/site-packages/mediapipe/modules/pose_landmark/pose_landmark_heavy.tflite")
poseLandmark.allocate_tensors()

# Get input and output tensors.
detectorInput = poseDetector.get_input_details()  # 224x224
detectorOutput = poseDetector.get_output_details()

poseInput = poseLandmark.get_input_details()  # 256x256
poseOutput = poseLandmark.get_output_details()
print(detectorInput)
# img = cv2.imread("person.png").astype(np.float32)


# h, w = img.shape[:-1]
# img256 = cv2.resize(img, (256, 256))
# img256_norm = img256.copy() / 255.
# # img224 = np.expand_dims(img224, 0)
# img256_norm = np.expand_dims(img256_norm, 0)

# poseLandmark.set_tensor(poseInput[0]['index'], img256_norm)

# poseLandmark.invoke()

# output_data = poseLandmark.get_tensor(poseOutput[0]['index'])
# output_data = np.ascontiguousarray(output_data).reshape(-1, 5)

# for c in output_data.astype(int):
#     c0 = int(c[0] * (w / 256))
#     c1 = int(c[1] * (h / 256))
#     print(c0, c1)
#     cv2.circle(img, (c0, c1), 2, (255, 155, 55), 4)

# cv2.imwrite('test_landmark.jpg', img)
