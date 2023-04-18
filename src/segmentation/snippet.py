
import numpy as np
from skimage import io

class Snippet:
    """A class representing a snippet of video data. A snippet is a made of a frame sampled at random from a video segment alongside its flow frame and body pose frame.

    Attributes:
        rgb_frame (numpy.ndarray): An RGB frame of the video snippet.
        flow_frame (numpy.ndarray): An optical flow frame of the video snippet.
        body_pose_frame (numpy.ndarray): A body pose frame of the video snippet.
        segment_number (int): The segment number of the video snippet. E.g. if it is from the first segment the number will be 0.

    Methods:
        display_rgb_frame: Displays the RGB frame using the scikit-image library.
        display_body_pose_frame: Displays the body pose frame using the scikit-image library.
        display_optical_flow_frame: Displays the optical flow frame using the scikit-image library.
    """

    def __init__(self, rgb_frame: np.ndarray, flow_frame: np.ndarray, body_pose_frame: np.ndarray, segment_number: int) -> None:
        """Initializes a Snippet object with the given video data.

        Args:
            rgb_frame: An RGB frame of the video snippet.
            flow_frame: An optical flow frame of the video snippet.
            body_pose_frame: A body pose frame of the video snippet.
            segment_number: The segment number of the video snippet.
        """
        self.rgb_frame = rgb_frame
        self.flow_frame = flow_frame
        self.body_pose_frame = body_pose_frame
        self.segment_number = segment_number

    def display_rgb_frame(self) -> None:
        """Displays the RGB frame using the scikit-image library."""
        io.imshow(self.rgb_frame)
        io.show()

    def display_body_pose_frame(self) -> None:
        """Displays the body pose frame using the scikit-image library."""
        io.imshow(self.body_pose_frame)
        io.show()

    def display_optical_flow_frame(self) -> None:
        """Displays the optical flow frame using the scikit-image library."""
        io.imshow(self.flow_frame)
        io.show()
