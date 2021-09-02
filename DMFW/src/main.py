from posix import listdir
from lib import *
from params import *
from dataShaper import *
from trainer import *
from optimizerDMFW import *
from modelPredictor import *
from graphs import *
from models import cnn, linear,lstm,seq2seq
from QBucketUploader import pushToQBucket
if __name__ == '__main__':

    parser = argp.ArgumentParser()

    parser.add_argument("-i", "--input_dir", help="path of the input data folder")
    parser.add_argument("-o", "--output_dir", help="path of the output directory.")
    parser.add_argument("-sdt", "--start_date", help="start date: [YYYY-MM-DD]")
    parser.add_argument("-edt", "--end_date", help="end date: [YYYY-MM-DD]")
    parser.add_argument("-cdt", "--cut_date", help="cut date: [YYYY-MM-DD]")
    parser.add_argument("-fl", "--floor", help="floor number [int between 1 to 7 inclusive]")
    parser.add_argument("-zn", "--nb_zone", help="number of zones")
    parser.add_argument("-grt", "--graph_type", help="Types of graph: [grid, cycle, line, complete, isolated]")
    parser.add_argument("-feat", "--feature", help="Feature to learn : [temperature, humidity, power]")
    parser.add_argument("-resm", "--resample_method", help="Resample method: [mean, max, sum]")
    parser.add_argument("-model", "--model", help="Choose between linear,lstm, cnn")
    parser.add_argument("-plotFig", "--plotFig", help="True to plot figures")
    parser.add_argument("-modePred", "--modePred", help="True to predict")


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
    plotFig = False
    modePred = False

    
    try:
        input_dir = args['input_dir']
        print ("input_dir : ", input_dir)
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
        floors = args['floor']
        print ("floor : ",floors)
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
    
    try:
        model = args['model']
        if model == "cnn":
            model = cnn.CNN1D
            model_arguments = (8, lookback, lookahead, 5)
        elif model == "linear":
            model = linear.Linear
        elif model == "lstm":
            model = lstm.LSTM
            model_arguments = (1, 60, lookback, lookahead)
        
    except:
        model = seq2seq
    finally:
        print ("model: ",model)
    

    try:
        plotFig = args['plotFig']
        if plotFig == "True": plotFig =True
        print ("plot Figures : ", plotFig)
    except:
        pass
    
    try:
        modePred = args['modePred']
        if modePred == "True": modePred =True
        print ("Prediction mode  : ", modePred)
    except:
        pass


    
    # import pdb; pdb.set_trace()
    floor_dict = createDictFloor(floors, input_dir)
    resample, remain_date = createDTFeat(start_date, end_date, floor_dict, feature,resample_method_X=resample_method)
    scaled_data, scalers = data_scaling(resample, feature)
    
    get_info_zone(scaled_data)

    train_date, test_date = splitDate(remain_date, cut_date)
    databyDate = createDataByDate(scaled_data, feature, remain_date)
    
    trainloader, testloader = get_loader(floors, databyDate, train_date, test_date, lookback, lookahead, batch_size)

    #nb_iterations = len(train_date)*len(trainloader[0]["2019-03-08"])

    '''
    trainloader = []
    testloder = []
    for zone in range(1,nb_zone+1):
        if zone != 3:
            zoneID = f"Floor{floor}Z{zone}"
            loaderZtrain = LoaderByZone(databyDate, zoneID, train_date, lookback, lookahead, batch_size, shuffle=True)
            loaderZtest = LoaderByZone(databyDate, zoneID, test_date, lookback, lookahead, batch_size)
            trainloader.append(loaderZtrain)
            testloder.append(loaderZtest)
    zone_no=len(trainloader)'''
    

    try:
        graph_type = args['graph_type']
        print ("graph_type : ",graph_type)
        graph, graph_name = None, None
        if graph_type == "complete":
            graph, graph_name = completegraph(len(trainloader))
        if graph_type == "cycle": 
            graph, graph_name = cycle_graph(len(trainloader))
        if graph_type == "grid": 
            grid_graph, grid = gridgraph(int(np.sqrt(len(trainloader))),int(np.sqrt(len(trainloader))))
        if graph_type == "line": 
            grid_graph_line, line = gridgraph(len(trainloader,1))
    except:
        pass


    try:
    # if True:
        trainXMFW = Trainer(graph,trainloader,model,*model_arguments, loss_fn,num_iters_base)
        values_dmfw = trainXMFW.train(DMFW, L_DMFW, eta_coef_DMFW, eta_exp_DMFW, reg_coef_DMFW,1,
                                path_figure_date=None)
        # import pdb; pdb.set_trace()
        #for item in [i for i in os.listdir(output_dir) if "Model_" in i]: pushToQBucket(output_dir+item+"/", f"Exp-{floors}-{feature}-{graph_name}/"+item+"/")
        #pushToQBucket(output_dir, f"Exp-{floor}-{feature}-{graph_name}/", skipDirectoryInclude=True)

        if not os.path.exist(output_dir):
            os.makesdirs(output_dir)

        save_log_csv(values_dmfw, output_dir)

        for i in range(len(trainXMFW.best_models)):
            save_state_dict(trainXMFW.best_models[i],i, output_dir)

    except:
    # else:
        print ("error in training... quitting...")
        import pdb; pdb.set_trace()

    if plotFig:
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
    
    if modePred:
        try:
            os.mkdir(output_dir+"Prediction/")
        except:
            pass
        # model_trained = trainXMFW.models[0]
        # import pdb; pdb.set_trace()
        for zone_no, model_trained in enumerate(trainXMFW.best_models):
            for date in testloader[zone_no].keys():
                true, pred = ModelPrediction(model_trained,date, testloader[zone_no], lookahead)
                plt.clf()
                # import pdb; pdb.set_trace()
                plt.plot([i for i in range(len(true))],true,label='Ground Truth' ) 
                plt.plot([i for i in range(len(pred))],pred, label='Predicted')
                plt.legend(loc='upper right')
                plt.xlabel("#Iterations",fontsize=15)
                plt.ylabel("Gap",fontsize=15)
                plt.savefig(output_dir+f"Prediction/F{floor}Z{zone_no}-{date}.png", dpi=200)
                pd.DataFrame.from_dict({"True":true, "Predicted": pred}).to_csv(output_dir+f"Prediction/F{floor}Z{zone_no}-{date}.csv")
