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

    def reset(self):
        self.optimizers = [0.] * self.num_nodes
        self.models = [0.] * self.num_nodes
        self.losses = [0.] * self.num_nodes
        self.gaps = [0.] * self.num_nodes
        self.best_models = [0.]*self.num_nodes

        self.obj_values = np.ndarray((self.num_iterations + 1, 4),
                                     dtype='float')
        
    def __nodeInit(self, data, label):
        nodewrap = TensorDataset(data, label)
        nodes = DataLoader(nodewrap, batch_size=data.size(0), shuffle=False)
        return nodes

    def initModelWeight(self, model):
        for m in model.modules():
            if isinstance(m, nn.LSTM):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.xavier_normal_(param.data)
                    else:
                        nn.init.constant_(param.data,0)
            elif isinstance(m,nn.BatchNorm1d):
                nn.init.uniform_(m.weight.data)
                nn.init.constant_(m.bias.data,0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.uniform_(m.bias.data)
                #nn.init.constant_(m.bias.data,0)

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

    def train(self, optimizer, L, eta_coef, eta_exp, reg_coef, radius, path_figure_date, print_freq):
        seed_everything()
        self.reset()

        days= list(self.dataloader[0].keys())

        for i in range(self.num_nodes):
            self.models[i] = self.model(*self.param)
            self.optimizers[i] = optimizer(self.models[i].parameters(),
                                           eta_coef=eta_coef,
                                           eta_exp=eta_exp,
                                           L=L,
                                           matrix_line=self.A[i],
                                           reg_coef=reg_coef,
                                           radius=radius)
            self.best_models[i] = copy.deepcopy(self.models[i])

        self.final_gap = [0.] * self.num_nodes
        
        t = 0
        
        for date in days:
            
            try:
                for i,loader in enumerate(self.num_nodes):
                    truez, predz = ModelPrediction(self.best_models[i], date, loader,lookahead)
                    path = path_figure_date+"/Model_"+str(i)+"/"
                    if not os.path.exists(path):
                        os.makedirs(path)
                    self.plotPrediction(truez, predz,date,path_to_save=path)
            except :
                if print_freq == 1:
                    print(f"----{date}----")
                pass
            
            loaderz = []

            for i in range(self.num_nodes):
                loaderz.append(self.dataloader[i][date])

            for couples in zip(*loaderz):
                datazones = [self.__nodeInit(*couples[i]) for i in range(len(couples))] 
            

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
                # import pdb; pdb.set_trace()

                opt_index = np.random.randint(low=0,high=L, size=1)


                for l in range(L):
                    #print("--------------------------")

                    if l==opt_index:
                        for i in range(self.num_nodes):
                            self.best_models[i] = copy.deepcopy(self.models[i])


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
                        outputs = self.best_models[i](x)
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
                try:
                    self.obj_values[t, :] = [t, loss, gap, local_gap]
                except:
                    print ("problem in Training Process... Skipping for debug...")
                    # import pdb; pdb.set_trace()
                    
                t+=1
            

        return self.obj_values