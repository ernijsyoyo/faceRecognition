# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

class MachineLearningModel():
  """
      Machine learning model (SVM) for associating particular image embeddings with labels
      Parameters
      ----------
      Embeddings : str
          Path to the serialized embeddings of an image
      Recognizer : str
          Path for outputting the trained ML model for image recognition
      Labels : str
          Path for outputting serialized labels
      Returns
      -------
      Numpy Array uint8
          Image representation
  """

  def __init__(self, Embeddings: str, Recognizer: str, Labels: str):
    self._RECOGNIZER = Recognizer
    self._LABELS = Labels

    # load the face embeddings
    print("[INFO] loading face embeddings...")
    self.data = pickle.loads(open(Embeddings, "rb").read())
        
  def Train(self):
    """
      Loads the serialized embeddings and train the SVC model
      Outputs
      ----------
      Trained Model
          Serialized object of the encoded trained model at path self.RECOGNIZER
      Labels
          Serialized object of the encoded labels at path self.RECOGNIZER
    """
    
    # encode the labels
    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(self.data["names"])

    # train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    print("[INFO] training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(self.data["embeddings"], labels)

    # write the actual face recognition model to disk
    f = open(self._RECOGNIZER, "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open(self._LABELS, "wb")
    f.write(pickle.dumps(le))
    f.close()

    print("[INFO] Finished")