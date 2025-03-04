from abc import ABC, abstractmethod

import cv2
import sys
import os


class CameraAdapter(ABC):
    @abstractmethod
    def get_frame(self):
        pass


class WebCameraModel(CameraAdapter):
    def __init__(self):
        self.camera_id = None
        self.camera = None
        
    def set_camera_id(self, camera_id):
        self.camera_id = camera_id
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            self.camera = None
            self.camera_id = None
        
    def get_frame(self) -> cv2.typing.MatLike | None:
        if self.camera is None:
            return None
        
        ret, frame = self.camera.read()
        return frame
    
    def list_available_ids(self):

        # Suppress stderr
        sys.stderr = open(os.devnull, 'w')
        sys.stdout = open(os.devnull, 'w')

        available_ids = []
        for i in range(10):
            camera = cv2.VideoCapture(i)
            if camera.isOpened():
                available_ids.append(i)
            camera.release()

        # Restore stderr
        sys.stderr = sys.__stderr__
        sys.stdout = sys.__stdout__

        return available_ids, self.camera_id
