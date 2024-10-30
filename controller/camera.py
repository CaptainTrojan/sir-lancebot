from abc import ABC, abstractmethod
import depthai as dai


class CameraAdapter(ABC):
    @abstractmethod
    def get_frame(self):
        pass


class DepthAICameraAdapter(CameraAdapter):
    def __init__(self):
        # Create pipeline
        pipeline = dai.Pipeline()

        # Define source and output
        camRgb = pipeline.create(dai.node.ColorCamera)
        xoutVideo = pipeline.create(dai.node.XLinkOut)

        xoutVideo.setStreamName("video")

        # Properties
        camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)

        xoutVideo.input.setBlocking(False)
        xoutVideo.input.setQueueSize(1)

        # Linking
        camRgb.video.link(xoutVideo.input)
        self.device = dai.Device(pipeline)
        self.video_queue = self.device.getOutputQueue(name="video", maxSize=1, blocking=False)

    def get_frame(self):
        videoIn = self.video_queue.get()
        return videoIn.getCvFrame()


# Usage example:
# camera_adapter = DepthAICameraAdapter(pipeline)
