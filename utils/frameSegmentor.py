import io
import requests
from PIL import Image
from scipy.signal import argrelextrema

class frameSegmentor(object):

  def __init__(self,
               class_name):
    self._class_name = class_name
    

    self._pose_classification_history = []
    self._pose_classification_filtered_history = []

  def __call__(self,
               frame,
               pose_classification,
               pose_classification_filtered,
               repetitions_count):

    
    self._pose_classification_history.append(pose_classification)
    self._pose_classification_filtered_history.append(pose_classification_filtered)

    
    output_img = Image.fromarray(frame)

    confidence, frames = self._get_confidence()

    seg_frames = frames[argrelextrema(confidence, np.less)]

    
    
    return seg_frames

  def _get_confidence(self):

    for classification_history in [self._pose_classification_history,
                                   self._pose_classification_filtered_history]:
      y = []
      for classification in classification_history:
        if classification is None:
          y.append(None)
        elif self._class_name in classification:
          y.append(classification[self._class_name])
        else:
          y.append(0)
    frames = np.arange(1,len(y)+1)
    y = np.asarray(y)

    return y,frames