import argparse

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Amazon', help='dataset name: Flickr/ACM/BlogCatalog/cora/citeseer/pubmed')
    parser.add_argument('--hidden_dim', type=int, default=64, help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--tests', type=int, default=1, help='Training epoch')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=1, help='balance parameter')
    parser.add_argument('--beta', type=float, default=0.3, help='balance parameter')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument('--patience', type=int, default=500, help='Patience')
    parser.add_argument('--weight_decay', type=float, default=0e-4)#0.0001
    parser.add_argument('--runs', type=int, default=5, help='Number of runs')
    args = parser.parse_args()
    return args