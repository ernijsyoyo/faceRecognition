from cv2 import cv2
import time
import re
from imutils import paths
import os

class PictureCapture():
  def __init__(self, Dataset, PicCount):
    self.Dataset = Dataset
    self.imagePaths = list(paths.list_images(self.Dataset))
    self.picCount = PicCount

  def SetPicCount(self, picCount):
    if picCount != 5:
      print("[OUTPUT]: Setting pic count to ", picCount)
    self.picCount = picCount
    
  def TakePictures(self):
    print("[INPUT]: Press ENTER if you wish to proceed by taking default number of photos - 5")
    print("[INPUT]: Or type in the number of how many extra photos you wish to take")
    decission = input()
    if decission.isdigit():
      self.SetPicCount(int(decission))

    # Determine total number of unique sub-folders
    subjects = {}
    for path in self.imagePaths:
      name = path.split(os.path.sep)[-2]
      if name not in subjects:
        subjects[name.lower()] = name

    # Detect the number from filenames at which extra cam pics will be named
    frames = []
    for (i, imagePath) in enumerate(self.imagePaths):
      # extract the person name from the image path
      fileName = os.path.basename(imagePath)
      frames.append(int(re.search(r'\d+', fileName).group(0)))
      
    cam =  cv2.VideoCapture(0)
    frameCounter = 0
    if len(frames) > 1:
      frameCounter = max(frames)


    if not cam.isOpened():
      raise IOError("Cannot open webcam")
    # Keep track of the last time a frame was processed
    last_recorded_time = time.time()

    print("[INPUT]: Please enter for which person you wish to add additional pictures")
    for value in subjects.values():
      print(value)
    
    subject = input().lower()
    if(subject not in subjects):
      os.mkdir(self.Dataset + subject)
      subjects[subject.lower()] = subject
    subjectToWriteTO = subjects.get(subject)


    iterations = 0
    while cam.isOpened():
      curr_time = time.time() # grab the current time

      # keep these three statements outside of the if statement, so we 
      #   can still display the camera/video feed in real time
      suc, img=cam.read()
      if suc==True:
        #operation on image, it's not important
        cv2.imshow('Frame', img)
        cv2.waitKey(1)
      else:
        break
      
      if curr_time - last_recorded_time >= 1: # it has been at least 2 seconds
        iterations = iterations + 1
        filePath = "Dataset/" + subjectToWriteTO + "/" + str(subjectToWriteTO) + str(frameCounter + iterations) + ".jpeg"
        last_recorded_time = curr_time
        cv2.imwrite(filePath, img)
        print("[OUTPUT]: Picture saved")
      if iterations == self.picCount:
        break  
    cv2.destroyAllWindows()
    cam.release()
    self.SetPicCount(5)
