from src.view.view import Visualizer
from src.model.camera import WebCameraAdapter
from src.model.arena import ArenaModel
import cv2 
import numpy as np


class BotViewModel:
    def __init__(self, arena_model, camera_model):
        self.camera_adapter = camera_model
        self.arena_model = arena_model

