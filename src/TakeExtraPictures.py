import cv2
import time
import re
from imutils import paths
import os

print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(_DATASET))
subjects = []
for path in imagePaths:
  name = path.split(os.path.sep)[-2]
  # result = re.search('/(.*)/', path)
  # subject = result.group(1)
  if name not in subjects:
    subjects.append(name)

# Detect the number from filenames at which extra cam pics will be named
frames = []
for (i, imagePath) in enumerate(imagePaths):
  # extract the person name from the image path
  fileName = os.path.basename(imagePath)
  frames.append(int(re.search(r'\d+', fileName).group(0)))
  

cam =  cv2.VideoCapture(0)
frameCounter = 999

if not cam.isOpened():
  raise IOError("Cannot open webcam")
last_recorded_time = time.time() # this keeps track of the last time a frame was processed

print("Please enter for which person you wish to add additional pictures")
for name in subjects:
  print(name)
subjectToWriteTO = input()

iterations = 0
while cam.isOpened():
  curr_time = time.time() # grab the current time

  # keep these three statements outside of the if statement, so we 
  #   can still display the camera/video feed in real time
  suc, img=cam.read()
  if suc==True:
    #operation on image, it's not important
     cv2.imshow('Frame', img)
  else:
    break
    cv2.waitKey(1)

  if curr_time - last_recorded_time >= 1: # it has been at least 2 seconds
    iterations = iterations + 1
    filePath = "Dataset/" + subjectToWriteTO + "/" + str(subjectToWriteTO) + str(frameCounter + iterations) + ".jpeg"
    last_recorded_time = curr_time
     cv2.imwrite(filePath, img)
    print("Picture saved")
  if iterations == _NUMBEROFPICS:
    break
    

 cv2.destroyAllWindows()
