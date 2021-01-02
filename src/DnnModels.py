import os
from cv2 import cv2

# Singlestons

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class CaffeModel(metaclass=Singleton):
    """
    Singleton class for returning a Caffe DNN model
    PARAMETERS
    ----------
    detector : str
        Path to the folder which contains Caffe .prototxt and .caffemodel files
    """
    def __init__(self, CaffeLocation: str):
        caffeProtoPath = os.path.sep.join([CaffeLocation, "deploy.prototxt"])
        caffeModelPath = os.path.sep.join([CaffeLocation, "res10_300x300_ssd_iter_140000.caffemodel"])
        self.detector = cv2.dnn.readNetFromCaffe(caffeProtoPath, caffeModelPath)

class TorchModel(metaclass=Singleton):
    """
    Singleton class for returning a Torch DNN model
    PARAMETERS
    ----------
    TorchLocation : str
        Path to the folder which contains Torch model
    """
    def __init__(self, TorchLocation: str):
        print("Torch Model initializing!")
        self.embedder = cv2.dnn.readNetFromTorch(str(TorchLocation))
