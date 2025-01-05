from abc import ABC, abstractmethod

import cv2


class CameraAdapter(ABC):
    @abstractmethod
    def get_frame(self):
        pass


class WebCameraAdapter(CameraAdapter):
    def __init__(self, camera_id=0):
        self.camera = cv2.VideoCapture(camera_id)

    def get_frame(self):
        ret, frame = self.camera.read()
        return frame
