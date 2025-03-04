from abc import ABC, abstractmethod

import cv2


class CameraAdapter(ABC):
    @abstractmethod
    def get_frame(self):
        pass


class WebCameraModel(CameraAdapter):
    def __init__(self):
        self.camera_id = None
        
    def set_camera_id(self, camera_id):
        self.camera_id = camera_id
        self.camera = cv2.VideoCapture(self.camera_id)
        
    def get_frame(self) -> cv2.typing.MatLike | None:
        ret, frame = self.camera.read()
        return frame
    
    def list_available_ids(self):
        available_ids = []
        i = 0
        while True:
            camera = cv2.VideoCapture(i)
            if not camera.isOpened():
                break
            available_ids.append(i)
            camera.release()
            i += 1
        return available_ids
