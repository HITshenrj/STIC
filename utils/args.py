import argparse
import torch


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--simulate', type=bool, default=False,
                            help='whether generate data.')
    arg_parser.add_argument('--load_path', type=str,
                            default="data/cos_gaussian_data/5_5_2", help='load dataset path.')
    arg_parser.add_argument('--load_dir', type=str,
                            default="0", help='load dataset dir.')
    arg_parser.add_argument('--data_dim', type=int, default=20,
                            help='the number of variables in synthetic generated data.')
    arg_parser.add_argument('--data_sample_size', type=int, default=100,
                            help='the number of samples of data.')
    arg_parser.add_argument('--graph_degree', type=int, default=15,
                            help='the number of degree in generated DAG graph.')
    arg_parser.add_argument('--max_lag', type=int, default=4,
                            help='the number of lag in generated DAG graph.')
    arg_parser.add_argument('--lag_edge_prob', type=float, default=0.5,
                            help='the probility of lag edge exits in generated DAG graph.')
    arg_parser.add_argument('--batch_size', type=int, default=64,
                            help='the number of batch size of data.')
    arg_parser.add_argument('--epochs', type=int, default=10000000,
                            help='the number of epochs for training.')
    arg_parser.add_argument('--lr', type=float, default=1e-5,
                            help='the learning rate for training.')
    arg_parser.add_argument('--print_interval', type=int, default=100,
                            help='the number of epochs for printing.')
    arg_parser.add_argument('--threshold', type=float, default=0.3,
                            help='the number of epochs for printing.')
    
    args = arg_parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(args)

    return args
