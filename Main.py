from src.DnnModels import *
from src.extract_embeddings import *
from src.recognize_video import *
from src.TakeExtraPictures import *
from src.train_model import *
from src.Utils import *
from src import *

class MainFunc:
  def __init__(self):
    self._DATASET = "Dataset/"
    self._EMBEDDING_OUTPUT = "Output/embeddings.pickle"
    self._RECOGNIZER = "Output/recognizer.pickle"
    self._LABELS = "Output/le.pickle"
    self._DETECTOR = "FaceDetectionModel/"
    self._EMBEDDINGMODEL = 'src/nn4.small2.v1.t7'

    self._TRAINER = None
    self._EMBEDDOR = None
    self._STREAM = None
    self._PICTURES = None

    self.switcher = {
        'e': self.Embeddor,
        't': self.Trainer,
        'r': self.Stream,
        'p': self.Pictures,
        'q': self.Quit
    }

  def main(self):
    while(True):
      self.PrintOptions()
      control = input()
      func = self.ChooseFunctionality(control)
      func()
      print("Finished!")
      print("---------")

  def PrintOptions(self):
    print("Please select the functionality which you wish to execute by inputting in the console")  
    print("Enter Q to quit")
    print("Enter E to extract and serialize facial embeddings using 'Dataset/' folder as input")
    print("Enter T to train the recognizer with the serialized embeddings")
    print("Enter R to recognize faces in real time from your webcams stream")
    print("Enter P to take some pictures with your webcam in order to extend the input dataset")

  def ChooseFunctionality(self, argument):
    # Get the function from switcher dictionary
    func = self.switcher.get(argument, lambda: "Invalid Input")
    return func

  def Embeddor(self):
    if self._EMBEDDOR is None:
      _EMBEDDOR = EmbeddingExtractor(self._DATASET, self._EMBEDDING_OUTPUT, self._EMBEDDINGMODEL, self._DETECTOR)
    _EMBEDDOR.ProcessFolders()

  def Trainer(self):
    if self._TRAINER is None:
      self._TRAINER = MachineLearningModel(self._RECOGNIZER, self._EMBEDDING_OUTPUT, self._LABELS)
    self._TRAINER.Train()

  def Stream(self):
    if self._STREAM is None:
      self._STREAM = RecognizeVideo(self._DETECTOR, self._EMBEDDINGMODEL, self._RECOGNIZER, self._LABELS)
    self._STREAM.StartVideoStream()

  def Pictures(self):
    if self._PICTURES is None:
      self._PICTURES = PictureCapture(self._DATASET, 5)
    self._PICTURES.TakePictures()

  def Quit(self):
    quit()

if __name__ == "__main__":
  main = MainFunc()
  main.main()