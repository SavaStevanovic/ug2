import os
from shutil import copyfile
import argparse
import numpy as np
import pandas as pd
from skimage.io import imread, imsave
from skimage.transform import resize

args = argparse.ArgumentParser()
args.add_argument("sequences_folder")
args.add_argument("output_folder")
args.add_argument("--output_size", default=224)
args = args.parse_args()

def getCropBounds(objYmin, objYmax, objXmin, objXmax, cropSize, imgHeight, imgWidth):
  objHeight = objYmax - objYmin
  objWidth = objXmax - objXmin
  cropCenterX = objXmin + (objWidth // 2)
  cropCenterY = objYmin + (objHeight // 2)

  cropRadius = np.max([objHeight, objWidth, cropSize]) // 2
  cropRadius = np.min([cropRadius, cropCenterX, imgWidth-cropCenterX, cropCenterY, imgHeight-cropCenterY])

  cropStartY = cropCenterY - cropRadius
  cropEndY = cropCenterY + cropRadius
  cropStartX = cropCenterX - cropRadius
  cropEndX = cropCenterX + cropRadius
  return cropStartY, cropEndY, cropStartX, cropEndX

for v in os.listdir(args.sequences_folder):
  seq = os.path.join(args.sequences_folder, v)
  seqOut = os.path.join(args.output_folder, v)
  os.mkdir(seqOut)
  for s in os.listdir(seq):
      ss = os.path.join(seq, s)
      if os.path.isdir(ss) and os.path.exists(ss + ".txt"):
          annotationsPath = ss + ".txt"
          annotations = pd.read_csv(annotationsPath, sep=' ')

          copyfile(annotationsPath, os.path.join(seqOut, s + ".txt"))
          sOut = os.path.join(seqOut, s)
          os.mkdir(sOut)

          for i, f in enumerate(os.listdir(os.path.join(seq, s))):
              ann = annotations.loc[i]
              frame = imread(os.path.join(seq, s, f))
              yMin, yMax, xMin, xMax = getCropBounds(ann.ymin, ann.ymax, ann.xmin, ann.xmax, args.output_size, frame.shape[0], frame.shape[1])
              frame = frame[yMin:yMax, xMin:xMax, :]
              #print((yMin, yMax), (xMin, xMax))
              print(frame.shape)
              frame = resize(frame, (args.output_size, args.output_size, 3))
              imsave(os.path.join(sOut, f), frame)
