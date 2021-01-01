from extract_embeddings import *
from recognize_video import *
from train_model import *
_DATASET = "Dataset/"
_EMBEDDING_OUTPUT = "Output/embeddings.pickle"
_DETECTOR = "FaceDetectionModel/"
_EMBEDDINGMODEL = "nn4.small2.v1.t7"

def main():
  embeddor = EmbeddingExtractor(_DATASET, 
                  _EMBEDDING_OUTPUT,
                  _DETECTOR,
                  _EMBEDDINGMODEL)
  embeddor.ProcessFolders(_DATASET)

if __name__ == "__main__":
  main()