import os
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd
from skimage.io import imread
from sklearn.metrics import label_ranking_average_precision_score
import labels
import matplotlib.pyplot as plt

class VideoEnh(gym.Env):
    metadata = {'render.modes': ['human']}

  def __init__(self, sequencesFolder):
      self.sequencesFolder = sequencesFolder
      self.__loadSequencesAndAnnotations()
      self.targetModel = models.VGG16

  def step(self, action):
      img = imread(self.currentSequence[self.currentStep])
      label = self.currentAnnotations.iloc[self.currentStep]["label"]
      self.currentStep += 1
      # apply action
      processedImg = img
      return processedImg

  def reset(self):
      r = np.random.rand(len(self.sequences))
      seqFrames = os.listdir(self.sequences[r])
      self.currentSequence = [os.path.join(self.sequences[r], seqFrames[i] for i in range(len(seqFrames))]
      self.currentAnnotations = pd.read_csv(self.annotations[r], sep=' ')
      self.currentStep = 0
      return

  def render(self, mode='human'):
      return

  def close(self):
      return

  def __computeReward(self, processedImg, label):
      oneHotGroundTruth = labels.ug2LabelToOneHot(label)
      scores = self.targetModel.evalutate(processedImg)
      return label_ranking_average_precision_score(oneHotGroundTruth, scores)

  def __loadSequencesAndAnnotations(self):
      self.sequences = []
      self.annotations = []

      for v in os.listdir(self.sequencesFolder):
          seq = os.path.join(self.sequencesFolder, v)
          for s in seq:
              if os.path.isdir(s):
                  self.sequences.append(os.path.join(seq, s))
                  self.annotations.append(os.path.join(seq, s + ".txt"))
