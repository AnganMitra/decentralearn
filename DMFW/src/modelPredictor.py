from lib import *


def ModelPrediction(model_to_test, date ,loader, lookahead):
    try:
        prediction = []
        true = []
        assert(len(loader)>0)
        assert(date in loader.keys())
        # import pdb; pdb.set_trace()

        for val, valpred in loader[date]:
            
            model_to_test.eval()
            pred = model_to_test(val)
            #print(pred.shape)
            prediction.append(pred.detach().numpy())
            true.append(valpred.detach().numpy())
        pred_array = np.asarray(prediction)
        true_array = np.asarray(true)
        #print(pred_array.shape)
        pred_shape = pred_array.shape
        #print(pred_shape)
        flattenTrue = true_array.reshape(pred_shape[0]*pred_shape[1], lookahead)[::lookahead].flatten()
        flattenPred = pred_array.reshape(pred_shape[0]*pred_shape[1], lookahead)[::lookahead].flatten()
        return flattenTrue, flattenPred
    except:
        print (f"Data not found...Skipping Prediction for date {date}..")
        # import pdb;pdb.set_trace()
        return [],[]
