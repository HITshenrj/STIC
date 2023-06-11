import torch
import torch.nn as nn
from utils.ER import generate_ode, structural_causal_process
from utils.args import get_args
from utils.evaluation import evaluate
import numpy as np
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

args = get_args()


def get_gtrue_from_links(args, links_with_out_func):
    gtrue = np.zeros((args.data_dim, args.data_dim, args.max_lag+1))
    for k, v in links_with_out_func.items():
        if len(v) == 0:
            continue
        for links in v:
            i = links[0][0]
            lag = links[0][1]
            gtrue[i][int(k)][-lag] = 1
    gtrue = np.where(gtrue != 0, 1, 0)
    return gtrue

if args.simulate is True:
    links, links_with_out_func = generate_ode(
    args.data_dim, args.graph_degree, args.max_lag, args.lag_edge_prob)
    data, nonstat = structural_causal_process(
        links, args.data_sample_size)
else:
    data = np.load("{}/{}/data.npy".format(args.load_path, args.load_dir))
    args.data_dim = data.shape[1]
    args.data_sample_size = data.shape[0]
    with open("{}/{}/{}".format(args.load_path,args.load_dir, "links.json"), 'r') as f:
        links_with_out_func = json.load(f)

gtrue = get_gtrue_from_links(args, links_with_out_func)

# Time Invariance Block with transformation (conv1d)

class CNN(nn.Module):
    def __init__(self, args) -> None:
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=args.data_dim,
                              out_channels=1,
                              kernel_size=(args.max_lag+1,))
        self.conv1 = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features=args.batch_size - args.max_lag,
                      out_features=args.data_dim*args.data_dim*(args.max_lag+1)),
            nn.PReLU())

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        graph = self.conv1(x)
        return graph


data = torch.from_numpy(data).to(args.device)
cnn = CNN(args).to(args.device)

optim = torch.optim.SGD(cnn.parameters(), lr=args.lr)
loss_fn = nn.MSELoss()

torch_zero = torch.tensor(0).to(device=args.device, dtype=torch.float)

for epoch in range(args.epochs):
    loss_sum = 0
    for i in range(args.max_lag, args.data_sample_size-args.batch_size, args.max_lag):
        x = data[i:i+args.batch_size, :]
        x_T = x.T.unsqueeze(0)
        conv_x = cnn(x_T)
        conv_x = conv_x.reshape(
            (args.data_dim, args.data_dim, (args.max_lag+1)))

        # Avoid self cycle
        for k in range(args.data_dim):
            conv_x[k, k, 0] = 0
        
        # Hadamard product and Mechanism Invariance Block
        y_hat = torch.zeros_like(x)
        for j in range(args.data_dim):
            for lag in range(args.max_lag+1):
                conv_x_j_lag = conv_x[:, j:j+1, lag]
                y_hat[:, j] += torch.mm(data[i-lag:i-lag +
                                        args.batch_size, :], conv_x_j_lag).squeeze(1)

        loss = loss_fn(y_hat, x)
        loss_sum += loss
        optim.zero_grad()
        loss.backward()
        optim.step()

        
    if epoch % args.print_interval == 0:
        # Threshold
        graph_pre = np.where(
            np.abs(conv_x.detach().cpu().numpy()) > args.threshold, 1, 0)
        print("====================epoch {}:=========================".format(epoch))
        print("sum loss:{}".format(loss_sum))
        print("{}".format(evaluate(gtrue, graph_pre)))
