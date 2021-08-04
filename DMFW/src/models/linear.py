from lib import *
class Linear(nn.Module):
    def __init__(self, nb_units, input_dim, output_dim):
        super(Linear, self).__init__()
        
        self.nb_units = nb_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nb_units2 = 16
    
        self.fc1 = nn.Linear(input_dim, nb_units)
        #self.fc2 = nn.Linear(nb_units, self.nb_units2)
        self.fc3 = nn.Linear(nb_units, output_dim)
        
        self.batch1 = nn.BatchNorm1d(nb_units)
        #self.batch2 = nn.BatchNorm1d(self.nb_units2)
        self.drop = nn.Dropout(0.5)
        
    def forward(self, x):
        inputs = x.squeeze(-1)
        out = self.fc1(inputs)
        out = self.batch1(out)
        out = nn.ReLU()(out)
        
        out = self.fc3(out)
        return out