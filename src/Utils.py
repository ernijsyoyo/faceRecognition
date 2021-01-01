from cv2 import cv2

def GetConfidence(detections, element):
  return detections[0, 0, element, 2]

def ShowFrameAndWait(frame):
  cv2.imshow("Frame", frame)
  return cv2.waitKey(1) & 0xFF

def ExtractEmbedding(torch, image, startX: int, endX: int, startY: int, endY: int):
  """Extract 128-D vector of facial characteristics from the image's specified region
  PARAMETERS
  ----------
  image : NumPyArray uint8 
      Return value of loading an image using OpenCV imread()
  startX
      Start coordinate of X axis
  startY
      Start coordinate of Y axis
  endX
      End coordinate of X axis
  endY
      End coordinate of Y axis
  RETURNS
  -------
  ndarray(float32)
      Array of facial embeddings of the particular face
  """

  # Extract the face and grab the region-of-interest dimensions
  face = image[startY:endY, startX:endX]
  (fH, fW) = face.shape[:2]

  # ensure the face width and height are sufficiently large
  if fW < 20 or fH < 20:
      return None

  # Construct a blob for the face ROI, then pass the blob through our face
  # embedding model to obtain the 128-d quantification
  faceBlob =  cv2.dnn.blobFromImage(face, 1.0 / 255,
      (96, 96), (0, 0, 0), swapRB=True, crop=False)
  embedder.setInput(faceBlob)
  return embedder.forward()