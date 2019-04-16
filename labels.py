import numpy as np
import pandas as pd

imagenetClasses = pd.read_csv("imagenet_classes.csv")
ug2Classes = pd.read_csv("UG2 Dataset/UG2ImageNet.txt")

def ug2LabelToOneHot(label):
    oneHot = np.zeros(1000, dtype=np.int8)
    
    for l in ug2Classes.loc[ug2Classes["UG2Class"] == label]["INetDesc"]:
        i = imagenetClasses.loc[imagenetClasses["class"] == l]["id"].astype(int)
        oneHot[i] = 1

    return oneHot
