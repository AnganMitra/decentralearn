from lib import *

class CNN1D(nn.Module):
    def __init__(self,output_chan, output_dim,input_dim,kernel_size):
        super(CNN1D, self).__init__()
        
        self.output_chan = output_chan
        self.output_dim = output_dim
        self.input_dim = input_dim
        
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=output_chan, kernel_size=kernel_size, stride=1, 
                                padding=int(np.floor(kernel_size/2)))
        
        
        self.maxpool = nn.MaxPool1d(3)
        self.batchnorm = nn.BatchNorm1d(output_chan)
        self.fc1 = nn.Linear(output_chan*int((input_dim/3)), output_dim, bias=True)
        
    def forward(self, x):
        inputs = x.unsqueeze(1).squeeze(-1)
        #print(inputs.shape)
        out = self.conv1d(inputs)
        out = self.batchnorm(out)
        #print(out.shape)
        out = self.maxpool(out)
        #print(out.shape)
        out = nn.LeakyReLU()(out)
        
        out = out.view(-1, out.shape[1]*out.shape[2])
        
        out = self.fc1(out)
        return out