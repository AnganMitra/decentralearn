from lib import *
from params import *
from modelPredictor import *

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
class Trainer:
    def __init__(self, graph, loaders, model, model_param, loss,
                 num_iterations):
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.model = model
        self.param = model_param
        self.A = torch.tensor(nx.adjacency_matrix(graph).toarray())
        self.dataloader = loaders
        self.num_iterations = num_iterations
        self.loss = loss
        self.obj_values = np.ndarray((self.num_iterations + 1, 4),
                                     dtype='float')

        self.optimizers = [0.] * self.num_nodes
        self.models = [0.] * self.num_nodes
        self.losses = [0.] * self.num_nodes
        #self.gaps = [0.]*self.num_nodes

    def reset(self):
        self.optimizers = [0.] * self.num_nodes
        self.models = [0.] * self.num_nodes
        self.losses = [0.] * self.num_nodes
        self.gaps = [0.] * self.num_nodes

        self.obj_values = np.ndarray((self.num_iterations + 1, 4),
                                     dtype='float')
        
    def __nodeInit(self, data, label):
        nodewrap = TensorDataset(data, label)
        nodes = DataLoader(nodewrap, batch_size=data.size(0), shuffle=False)
        return nodes

    def weight_reset(self, layer):
        if isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.Linear) or isinstance(layer,nn.Conv1d):
            layer.reset_parameters()
            
    def initModelWeight(self, model):
        for name,param in model.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param,0.)
            elif 'weight' in name:
                if not 'batch' in name:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.uniform_(param)

    def saveCheckPts(self, t, path):
        check_pts = {}
        for i in range(self.num_nodes):
            ckp_i = {
                "t": t,
                "weight": [param for param in self.models[i].parameters()],
                "optimizer_weight": self.optimizers[i].w_dict,
                "oracles": self.optimizers[i].G,
                "loss": self.losses[i]
            }  #,
            #"avg_loss": self.avg_loss[i]}
            check_pts[i] = ckp_i
        torch.save(check_pts, path + "checkpts_models" + "_" + str(t) + ".tar")
        
    def plotPrediction(self, true, pred,date,path_to_save):
        fig = plt.figure(figsize=(5,3))
        plt.suptitle("{}".format(date))
        plt.plot(true)
        plt.plot(pred)
        #plt.show()
        fig.savefig(os.path.join(path_to_save,date))
        plt.close()

    def train(self, optimizer, L, eta_coef, eta_exp, reg_coef, radius, path_figure_date):
        seed_everything()
        self.reset()
        
        z1= list(self.dataloader.keys())[0]

        for i in range(self.num_nodes):
            self.models[i] = self.model(*self.param)
            self.optimizers[i] = optimizer(self.models[i].parameters(),
                                           eta_coef=eta_coef,
                                           eta_exp=eta_exp,
                                           L=L,
                                           matrix_line=self.A[i],
                                           reg_coef=reg_coef,
                                           radius=radius)

        self.final_gap = [0.] * self.num_nodes
        
        t = 0
        
        for date in z1.keys():
            
            for i,loader in enumerate(self.dataloader):
                truez, predz = ModelPrediction(self.models[i], date, loader,lookahead)
                path = path_figure_date+"/Model_"+str(i)+"/"
                if not os.path.exists(path):
                    os.makedirs(path)
                self.plotPrediction(truez, predz,date,path_to_save=path)
            
            # for (couple1, couple2, couple4,couple5) in zip(z1[date],z2[date], z4[date], z5[date]):
            #     datazones = [self.__nodeInit(*couple1), 
            #                  self.__nodeInit(*couple2),
            #                  self.__nodeInit(*couple4),
            #                  self.__nodeInit(*couple5)]
            datazones =[] 
            for zone in self.dataloader:
                datazones.append(self.__nodeInit(*zone))

                for i in range(self.num_nodes):
                    self.initModelWeight(self.models[i])
                    self.models[i].train()

                    def closure():
                        self.optimizers[i].zero_grad(set_to_none=True)
                        x, y = iter(datazones[i]).next()
                        output = self.models[i](x)
                        loss = self.loss(output,y)
                        loss.backward()

                    self.optimizers[i].initValue(closure)
                
                for l in range(L):
                    #print("--------------------------")
                    for i in range(self.num_nodes):
                        self.optimizers[i].neighborsAverage(self.optimizers)
                    for i in range(self.num_nodes):
                        
                        def closure():
                            self.optimizers[i].zero_grad(set_to_none=True)
                            x, y = iter(datazones[i]).next()
                            output = self.models[i](x)
                            loss = self.loss(output, y)
                            loss.backward()
                            
                        self.optimizers[i].step(l, closure)

                self.gaps_off = [0.] * self.num_nodes
                for i in range(self.num_nodes):
                    with torch.no_grad():
                        self.models[i].eval()
                        x, y = iter(datazones[i]).next()
                        outputs = self.models[i](x)
                        curr_loss = self.loss(outputs, y)
                    self.final_gap[i] += self.optimizers[i].init_gap
                    self.final_gap[i] /= (t + 1)
                    self.gaps_off[i] = self.optimizers[i].init_gap
                    self.losses[i] = curr_loss.detach().numpy()

                loss = np.mean(self.losses)
                gap = np.max(self.final_gap)  #.detach().numpy(
                local_gap = np.max(self.gaps_off)
                if t % 1 == 0:
                    print("t_{} : loss : {:.5f} gap : {} local_gap {}".format(
                        t, loss, gap, local_gap))
                    
                self.obj_values[t, :] = [t, loss, gap, local_gap]
                    
                t+=1
            

        return self.obj_values