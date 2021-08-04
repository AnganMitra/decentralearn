
from lib import *


lookback = 13
lookahead = 1
batch_size = 32


loss_fn = nn.SmoothL1
num_iters_base = 504
eta_coef_DMFW = 1
eta_exp_DMFW = 0.75
rho_coef_DMFW = 4e-0
rho_exp_DMFW = 1/2
reg_coef_DMFW = 10
L_DMFW = 504

model = "CNN1D"

shuffle_Data= True # SHUFFLE DAILY DATA

