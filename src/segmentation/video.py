
from .segment import *
import cv2

class Video:
    def __init__(self, path, K):
        self.path = path
        self.segments_number = K
        self._segment_video()
    
    def _segment_video(self):
        # Open the video file
        cap = cv2.VideoCapture(self.path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the number of frames in each segment
        self.frames_per_segment = self.total_frames // self.segments_number

        # Initialize the list of segments
        self.segments = []

        # Set the frame position to the beginning of the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Read and save each segment of frames
        for i in range(self.segments_number):
            
            # Read the frames in the segment
            frames = []
            for j in range(self.frames_per_segment):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            # Create the segment dynamically
            segment = Segment(i, frames, self.path)
            self.segments.append(segment)

            # Set the frame position to the beginning of the next segment
            cap.set(cv2.CAP_PROP_POS_FRAMES, (i+1) * self.frames_per_segment)

        # Release the video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()
