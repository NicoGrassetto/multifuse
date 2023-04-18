
import subprocess
pip_cmd = ['pip', 'install', 'mediapipe']
result = subprocess.run(pip_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

import random
import cv2
import mediapipe as mp
from google.colab.patches import cv2_imshow
import numpy as np
import cv2
import numpy as np
from skimage.io import imshow
from .snippet import *

class Segment:
    def __init__(self, segment_index, frames, path):
        self.path = path
        self.segment_index = segment_index
        self.frames = frames
        self.random_snippet = self._generate_random_snippet()

    def _generate_random_snippet(self):
        frame_index = self._generate_random_number(0, len(self.frames)-1)

        return Snippet(self.frames[frame_index], 
                       self._generate_optical_flow_fields(frame_index), 
                       self._generate_body_pose(frame_index), 
                       self.segment_index)

    def _generate_body_pose(self, frame_index):
        # Load the image
        image = cv2.cvtColor(np.array(self.frames[frame_index]), cv2.COLOR_RGB2BGR)#cv2.imread('/content/download.jpeg')

        # Initialize the MediaPipe Pose model
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2)

        # Detect pose in the image
        results = pose.process(image)

        # Visualize the results
        mp_drawing = mp.solutions.drawing_utils
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.waitKey(0)
        return annotated_image

    def _generate_optical_flow_fields(self, frame_index):
        # Open the video file
        cap = cv2.VideoCapture(self.path)
        # Initialize the previous frame and previous optical flow field
        prev_frame = None
        prev_flow = None

        # Initialize the list of optical flow fields
        flows = []

        # Loop over the video frames
        while True:
            # Read the next frame
            ret, frame = cap.read()

            # Check if the frame was successfully read
            if not ret:
                break

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Compute the optical flow between the current and previous frames
            if prev_frame is not None and prev_flow is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flows.append(flow)
                prev_flow = flow
            else:
                prev_flow = np.zeros_like(frame)
            
            # Set the current frame as the previous frame for the next iteration
            prev_frame = gray.copy()

        # Stack up the optical flow fields into a 3D tensor
        stacked_flow = np.stack(flows, axis=0)

        # Calculate the magnitude of the optical flow vectors
        mag, _ = cv2.cartToPolar(stacked_flow[..., 0], stacked_flow[..., 1])

        # Display the grayscale optical flow fields for the last frame
        #imshow(mag[-1])
        return mag[frame_index]
        
    def _generate_random_number(self, start, end):
        return random.randint(start, end)
