import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse as agp

dataLog={}

def populateDataLog(input_dir):
    for file in os.listdir(input_dir):
        lossArray = []
        gapArray = []
        localGapArray =[]
        a=open(input_dir+file, "r").read().replace("0>", "").split("\n")
        for item in a:
            try:
                b=item.strip().replace(":", "").split()
                loss, gap, local_gap = b[2],b[4], b[-1]
                lossArray.append(loss)
                gapArray.append(gap)
                localGapArray.append(local_gap)
            except:
                pass
        dataLog[file]={
            "lossArray" :lossArray,
            "gapArray" : gapArray,
            "localGapArray" : localGapArray
        }

def plotComparisons(keys, comparator, output_dir):
    plt.clf()
    for key in keys:
        plt.plot([i for i in range(len(dataLog[key]['lossArray']))],dataLog[key]['lossArray'], label="loss"  )
    plt.legend()
    plt.savefig(output_dir+f"{comparator}-Loss.png", dpi=200)

    plt.clf()
    for key in keys:
        plt.plot([i for i in range(len(dataLog[key]['gapArray']))],dataLog[key]['gapArray'], label="Gap"  )
    plt.legend()
    plt.savefig(output_dir+f"{comparator}-gapArray.png", dpi=200)
    
    plt.clf()
    for key in keys:
        plt.plot([i for i in range(len(dataLog[key]['localGapArray']))],dataLog[key]['localGapArray'], label="local Gap Array"  )
    plt.legend()
    plt.savefig(output_dir+f"{comparator}-localGapArray.png", dpi=200)    

if __name__=="__main__":
    featureOption = ["temperature"]
    compOptions = ["floorComp", "graphComp", "featureComp"]
    graphOptions=["complete","cycle"]
    floorOptions = [3,5,7]

    parser = agp.ArgumentParser()

    parser.add_argument("-i", "--input_dir", help="path of the input data folder")
    parser.add_argument("-o", "--output_dir", help="path of the output directory.")
    parser.add_argument("-c", "--comparator", help="Compare between graphs, features, floors")

    args = vars(parser.parse_args())
    input_dir=None
    output_dir = None
    floor = None
    graph_type=None
    feature=None
    try:
        input_dir = args['input_dir']
        print ("input_dir : ", input_dir)
    except:
        pass
    try:
        output_dir = args['output_dir']
        print ("output_dir : ", output_dir)
    except:
        pass
    try:
        comparator = args['comparator'].strip()
        print ("comparator : ", comparator)
    except:
        pass

    populateDataLog(input_dir)
    keys = []
    comparator =None
    if "floor" == comparator: 
        comparator = floor 
        keys = [i for i in dataLog.keys() if f"dfmw-{comparator}-" in i]
    elif "graph" == comparator: 
        keys = [i for i in dataLog.keys() if f"-{comparator}-" in i]
    elif "feature" == comparator: 
        keys = [i for i in dataLog.keys() if f"-{comparator}" in i]
    
    plotComparisons(keys, comparator, output_dir= "./")
