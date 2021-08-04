from lib import *
from params import *
from dataShaper import *
from trainer import *
from optimizerDMFW import *
from modelPredictor import *
from graphs import *

if __name__ == '__main__':

    parser = argp.ArgumentParser()

    parser.add_argument("-i", "--input_data", help="path of the input data folder")
    parser.add_argument("-o", "--output_dir", help="path of the output directory.")
    parser.add_argument("-sdt", "--start_date", help="start date: [YYYY-MM-DD]")
    parser.add_argument("-edt", "--end_date", help="end date: [YYYY-MM-DD]")
    parser.add_argument("-cdt", "--cut_date", help="cut date: [YYYY-MM-DD]")
    parser.add_argument("-fl", "--floor", help="floor number [int between 1 to 7 inclusive]")
    parser.add_argument("-zn", "--nb_zone", help="number of zones")
    parser.add_argument("-grt", "--graph_type", help="Types of graph: [grid, cycle, line, complete, isolated]")
    parser.add_argument("-feat", "--feature", help="Feature to learn : [temperature, humidity, power]")
    parser.add_argument("-resm", "--resample_method", help="Resample method: [min, max, sum]")


    args = vars(parser.parse_args())
    input_dir = None
    output_dir = None
    start_date = None
    end_date=None
    cut_date = None
    floor=None
    nb_zone=None
    graph_type=None
    model = None
    
    try:
        input_dir = args['input_dir']
        print ("IFC File : ", input_dir)
    except:
        pass
    try:
        output_dir = args['output_dir']
        print ("Output Directory : ",output_dir)
    except:
        pass
    try:
        start_date = args['start_date']
        print ("start date : ", start_date)
    except:
        pass
    try:
        end_date = args['end_date']
        print ("end_date : ",end_date)
    except:
        pass
    try:
        cut_date = args['cut_date']
        print ("cut_date : ", cut_date)
    except:
        pass
    try:
        floor = args['floor']
        print ("floor : ",floor)
    except:
        pass
    try:
        nb_zone = int(args['nb_zone'])
        print ("Number of zones : ", nb_zone)
    except:
        pass
    try:
        graph_type = args['graph_type']
        print ("graph_type : ",graph_type)
        graph, graph_name = None, None
        if graph_type == "complete":
            graph, graph_name = completegraph(nb_zone)
        if graph_type == "cycle": 
            graph, graph_name = cycle_graph(nb_zone)
        if graph_type == "grid": 
            grid_graph, grid = gridgraph(int(np.sqrt(nb_zone)),int(np.sqrt(nb_zone)))
        if graph_type == "line": 
            grid_graph_line, line = gridgraph(nb_zone,1)
    except:
        pass

    try:
        feature = args['feature']
        print ("feature: ", feature)
    except:
        pass    

    try:
        feature = args['feature']
        print ("feature: ", feature)
    except:
        pass
    try:
        resample_method = args['resample_method']
        print ("resample_method: ",resample_method)
    except:
        pass
    
    

    floor_dict = createDictFloor(input_dir, f"Floor{floor}")
    for data in floor_dict.keys():
        zone = floor_dict[data]
        print("{} Start: {} End: {} Count:{}".format(data,zone.index.min(),zone.index.max(), zone.shape[0]))
        print("")
    resample,scalers, index_nan = createDTFeat(start_date, end_date, floor_dict, feature,resample_method=resample_method)
    cleanedData, remain_date = cleanNan(resample, index_nan)
    for data in cleanedData.keys():
        zone = cleanedData[data]
        print("{} Start: {} End: {} Count:{}".format(data,zone.index.min(),zone.index.max(), zone.shape[0]))
        print("{} Dates: {}".format(data,len(np.unique(zone["date"]))))
    
    getInfoTimeShape(cleanedData)

    train_date, test_date = splitDate(remain_date, cut_date)
    databyDate = createDataByDate(cleanedData, feature, remain_date)
    getInfoDataByDate(databyDate, train_date)

    trainloader = []
    testloder = []
    for zone in range(1,nb_zone+1):
        zoneID = f"Floor{floor}Z{zone}"
        loaderZtrain = LoaderByZone(databyDate, zoneID, train_date, lookback, lookahead, batch_size, shuffle=True)
        loaderZtest = LoaderByZone(databyDate, zoneID, test_date, lookback, lookahead, batch_size)
        trainloader.append(loaderZtrain)
        testloder.append(loaderZtest)
    zone_no=0
    for trainloader_item, testloder_item in zip(trainloader, testloder):
        zone_no+=1
        trainXMFW = Trainer(graph,trainloader_item,model, (8,lookahead,lookback,5), loss_fn,num_iters_base)
        values_dmfw = trainXMFW.train(DMFW, L_DMFW, eta_coef_DMFW, eta_exp_DMFW, reg_coef_DMFW,1,
                                path_figure_date= output_dir)

        plt.clf()
        plt.figure(figsize=(10,5))
        plt.suptitle("{}".format(graph_name))
        plt.plot(values_dmfw[:,0][:-1],values_dmfw[:,2][:-1], label='DMFW', marker='^', markersize=4,
                markevery=[i for i in range(len(values_dmfw[:,0][1:])) if i%10==0])
        plt.axhline(y=0, color='grey', linestyle='--')
        plt.legend(loc='upper right')
        #plt.ylim((1e-4, 1e0))
        plt.yscale("log")
        plt.xlabel("#Iterations",fontsize=15)
        plt.ylabel("Gap",fontsize=15)

        plt.savefig(output_dir+f"Gap-F{floor}Z{zone_no}.png", dpi=200)
        print(f"{np.mean(values_dmfw[:,1][:-1])}")

        onlineloss = np.cumsum(values_dmfw[:,1][:-1])
        arangement = np.arange(1,len(onlineloss)+1)
        onlineloss = onlineloss/arangement
        print(f"{onlineloss}")


        plt.clf()
        plt.figure(figsize=(10,5))
        plt.suptitle("{}".format(graph_name))
        plt.plot(values_dmfw[:,0][:-1],values_dmfw[:,1][:-1], label='Step Loss', marker='^', markersize=4,
                markevery=[i for i in range(len(values_dmfw[:,0][1:])) if i%10==0])
        plt.plot(values_dmfw[:,0][:-1],onlineloss, label='Online Loss', marker='^', markersize=4,
                markevery=[i for i in range(len(values_dmfw[:,0][1:])) if i%10==0])
        plt.axhline(y=0, color='grey', linestyle='--')
        plt.legend(loc='upper right')
        #plt.ylim((1e-3, 1e2))
        plt.yscale("log")
        plt.xlabel("#Iterations",fontsize=15)
        plt.ylabel("Loss",fontsize=15)
        plt.savefig(output_dir+f"OnlineLoss-F{floor}Z{zone_no}.png", dpi=200)

        model_trained = trainXMFW.models[0]
        true, pred = ModelPrediction(model_trained,cut_date, testloder_item, lookahead)

        plt.plot(true,label='Ground Truth' ) 
        plt.plot(pred, label='Predicted')
        plt.legend(loc='upper right')
        plt.xlabel("#Iterations",fontsize=15)
        plt.ylabel("Sensor Value",fontsize=15)
        plt.savefig(output_dir+f"Prediction-F{floor}Z{zone_no}.png", dpi=200)