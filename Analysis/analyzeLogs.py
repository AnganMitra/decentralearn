import pandas as pd
import os
import matplotlib.pyplot as plt

dataLog={}
for file in os.listdir(""):
    lossArray = []
    gapArray = []
    localGapArray =[]
    a=open(file, "r").read().replace("0>", "").split("\n")
    for item in a:
        b=item.strip().replace(":", "").split()
        loss, gap, local_gap = b[2],b[4], b[-1]
        lossArray.append(loss)
        gapArray.append(gap)
        localGapArray.append(local_gap)

    dataLog[file]={
        "lossArray" :lossArray,
        "gapArray" : gapArray,
        "localGapArray" : localGapArray
    }

plotLoss("floorComp", "graphComp", "featureComp")