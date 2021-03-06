import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse as agp

featureOptions = ["temp", "ACPower", "lux", "lightPower"]
compOptions = ["floorComp", "graphComp", "featureComp"]
graphOptions=["complete","cycle", "line"]
floorOptions = [3,4,5,7]

def load_directory( directory):
    print (input_dir, directory)
    floorwisePerf = {}
    for file in os.listdir(directory):
        floorNumber = file.split("-")[1]
        lossArray = []
        gapArray = []
        localGapArray =[]
        a=open(directory+file, "r").read().split("\n")
        for item in a:
            if "loss" not in item: continue
            try:
                b=item.strip().replace(":", "").split()
                loss, gap, local_gap = b[2],b[4], b[-1]
                lossArray.append(loss)
                gapArray.append(gap)
                localGapArray.append(local_gap)
            except:
                pass
        floorwisePerf[floorNumber]={
            "lossArray" :lossArray,
            "gapArray" : gapArray,
            "localGapArray" : localGapArray
        }
    return floorwisePerf

def populateDataLog(input_dir):
    dataLog={}
    for graph in graphOptions:
        for feature in featureOptions:
            dataLog[f"{graph}-{feature}"] = load_directory( f"{input_dir}{graph}-{feature}/")    

    
def plotComparisons(dataLog,keys, comparator, output_dir):
    plt.clf()
    for measure in ["lossArray", "gapArray", "localGapArray" ]:
        for key in keys:
            plt.plot([i for i in range(len(dataLog[key][measure]))],dataLog[key][measure], label=key  )
        plt.legend()
        # plt.show()
        plt.savefig(output_dir+f"{comparator}-{measure}.png", dpi=200, bbox_anchor="tight")
 

if __name__=="__main__":


    parser = agp.ArgumentParser()

    parser.add_argument("-i", "--input_dir", help="path of the input data folder")
    parser.add_argument("-o", "--output_dir", help="path of the output directory.")
    parser.add_argument("-c", "--comparator", help="Compare between graphs, features, floors")
    parser.add_argument("-a", "--argument", help="mention the options to compare")
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

    try:
        argument = args['argument'].strip().split(",")
        print ("argument : ", argument)
    except:
        pass

    dataLog=populateDataLog(input_dir)

    keys = []
    if "floor" == comparator: 
        for arg in argument:
            keys += [i for i in dataLog.keys() if f"dfmw-{arg}-" in i]
    elif "graph" == comparator: 
        for arg in argument:
            keys = [i for i in dataLog.keys() if f"-{arg}-" in i]
    elif "feature" == comparator: 
        for arg in argument:
            keys = [i for i in dataLog.keys() if i.endswith(f"-{arg}.csv") ]
    
    print (keys)
    plotComparisons(dataLog,keys, comparator, output_dir)
