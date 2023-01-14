import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
from utils import preprocessing as prepost



# Load TFLite model and allocate tensors.
poseDetector = tflite.Interpreter(model_path="/home/prateek/II/mediapipe/mp_env/lib/python3.8/site-packages/mediapipe/modules/pose_detection/pose_detection.tflite")
poseDetector.allocate_tensors()

# Get input and output tensors.
detectorInput = poseDetector.get_input_details()  
detectorOutput = poseDetector.get_output_details()


# Test model on random input data.
input_data = cv2.imread("person.png").astype(np.float32)

h, w = input_data.shape[:-1]
img256 = cv2.resize(input_data, (128, 128))
img256_norm = img256.copy() / 255.
# img224 = np.expand_dims(img224, 0)
img256_norm = np.expand_dims(img256_norm, 0)

poseDetector.set_tensor(detectorInput[0]['index'], img256_norm)

input_data = np.expand_dims(input_data, axis=0)
print(f'The shape of the image is {input_data.shape}')

poseDetector.invoke()

# The function get_tensor() returns a copy of the tensor data.
# Use tensor() in order to get a pointer to the tensor.
output_data = poseDetector.get_tensor(detectorOutput[0]['index'])
# probabilities = poseDetector.get_tensor(detectorOutput['pro'])
print(output_data.shape)
