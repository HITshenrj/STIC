import torch
import torch.nn as nn
from utils.ER import generate_ode, structural_causal_process
from utils.args import get_args
from utils.evaluation import evaluate
import numpy as np
import json
import os
import copy

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
    data = copy.deepcopy(data[:args.data_sample_size, :])
    args.data_dim = data.shape[1]
    args.data_sample_size = data.shape[0]
    with open("{}/{}/{}".format(args.load_path, args.load_dir, "links.json"), 'r') as f:
        links_with_out_func = json.load(f)

gtrue = get_gtrue_from_links(args, links_with_out_func)

# Mechanism Invariance Block

class DataTrans1(nn.Module):
    def __init__(self, d, lag) -> None:
        super(DataTrans1, self).__init__()
        self.lag = lag
        self.weight_matrix = nn.Parameter(torch.randn(d, lag))
        self.bias_matrix = nn.Parameter(torch.randn(d, lag))
        self.activate = nn.Tanh()

    def forward(self, x):
        x_return = list()
        for i in range(x.shape[1]-self.lag+1):
            tmp_x = self.activate(
                x[:, i:i+self.lag] * self.weight_matrix + self.bias_matrix)
            x_return.append(tmp_x.unsqueeze(0))
        x_return = torch.cat(x_return, dim=0)
        return x_return


class DataTrans(nn.Module):
    def __init__(self, d, lag):
        super(DataTrans, self).__init__()
        self.lag = lag
        self.weight_matrix = nn.Parameter(torch.randn(1, d, lag))
        self.bias_matrix = nn.Parameter(torch.randn(1, d, lag))
        self.activate = nn.Tanh()

    def forward(self, x):
        s = x.shape[0]
        wm = torch.cat([self.weight_matrix for _ in range(s)], dim=0)
        bm = torch.cat([self.bias_matrix for _ in range(s)], dim=0)

        x = self.activate(x*wm+bm)
        return x

# Time Invariance Block with transformation (conv1d)

class CNN(nn.Module):
    def __init__(self, args) -> None:
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=args.data_dim,
                              out_channels=args.data_dim//2,
                              kernel_size=(args.max_lag+1,))
        self.conv1 = nn.Conv1d(in_channels=args.data_dim//2,
                               out_channels=1,
                               kernel_size=(1,))
        self.conv2 = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features=args.batch_size - args.max_lag,
                      out_features=args.data_dim*args.data_dim*(args.max_lag+1)),
            nn.PReLU())

        self.lag = args.max_lag+1
        self.data_trans = DataTrans1(args.data_dim, self.lag)
        self.data_trans2 = DataTrans(args.data_dim, self.lag)
        self.activate = nn.Tanh()

    def forward(self, x: torch.Tensor, x_lag: torch.Tensor):
        x_cov = self.conv(x)
        x_conv1 = self.conv1(x_cov)
        graph = self.conv2(x_conv1)
        x_trans = self.data_trans(x_lag)
        x_trans = self.data_trans2(x_trans)
        return graph, x_trans


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
        x_lag = data[i-args.max_lag:i+args.batch_size, :].T
        conv_x, x_trans = cnn(x_T, x_lag)
        conv_x = conv_x.reshape(
            (args.data_dim, args.data_dim, (args.max_lag+1)))

        # Avoid self cycle
        for k in range(args.data_dim):
            conv_x[k, k, 0] = 0

        # Hadamard product
        y_hat = torch.zeros_like(x)
        for j in range(args.data_dim):
            for lag in range(args.max_lag+1):
                conv_x_j_lag = conv_x[:, j:j+1, lag]
                y_hat[:, j] += torch.mm(x_trans[:, :, args.max_lag-lag],
                                        conv_x_j_lag).squeeze(1)

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
