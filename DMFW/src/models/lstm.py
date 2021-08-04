from lib import *
class LSTM(nn.Module):
    def __init__(self, input_size, output_size, time_step_in, time_step_out):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.time_step_in = time_step_in
        self.time_step_out = time_step_out
        self.num_layers = 1
        
        self.encoder = nn.LSTM(self.input_size, self.output_size,
                               num_layers=self.num_layers, batch_first=True, bias=True)
        
        self.batch = nn.BatchNorm1d(self.output_size)
        self.linear2 = nn.Linear(self.num_layers*self.output_size, self.time_step_out)
        
    def forward(self,x):
        out_en, (h_en,_) = self.encoder(x)
        h_en = h_en.view(-1, self.num_layers*self.output_size)
        h_en = self.batch(h_en)
        out = nn.LeakyReLU()(h_en)
        #out = torch.sigmoid(self.linear(out))
        out = self.linear2(out)
        return out