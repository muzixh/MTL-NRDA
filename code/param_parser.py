"""Parameter parsing."""

import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run Higher Order GCN")
    parser.add_argument('--model', default="HOGCN")
    parser.add_argument('--cuda', action='store_false', default=True,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--hidden1', type=int, default=64,
                        help='Number of hidden units for encoding layer 1.')
    parser.add_argument('--hidden2', type=int, default=32,
                        help='Number of hidden units for encoding layer 2.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--network_type', nargs='?', default="DDI", choices=["LRI"],
                        help='choose from LRI')
    parser.add_argument('--input_type', nargs='?', default="node2vec", choices=["one_hot", "node2vec"],
                        help='choose from one_hot, node2vec')
    parser.add_argument("--early-stopping", type=int, default=10,
                        help="Number of early stopping rounds. Default is 10.")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
                        help="Learning rate. Default is 5e-4.")
    parser.add_argument("--layers-1", nargs="+", type=int,
                        help="Layer dimensions separated by space (top). E.g. 200 20.")
    parser.add_argument("--layers-2", nargs="+", type=int,
                        help="Layer dimensions separated by space (bottom). E.g. 200 200.")
    parser.add_argument("--fold_id", type=int, default=1,
                        help="Identifier to preprocessed splits. Default is 1.")
    parser.add_argument("--order", type=int, default=3,
                        help="Order of neighborhood (if order = 3, P ={0, 1, 2, 3}).  Default is 3.")
    parser.add_argument("--dimension", type=int, default=32,
                        help="Dimension for each adjacency. Default is 32.")
    parser.add_argument('--encoder_layers', type=int, default=2, help="Number of encoder layers.")
    parser.add_argument('--num_heads', type=int, default=4, help="Number of attention heads.")
    parser.add_argument('--d_model', type=int, default=128, help="Dimension of the model.")
    parser.add_argument('--d_ff', type=int, default=256, help="Dimension of feedforward network.")
    parser.add_argument('--ratio', action='store_true', default=False,
                        help='Train with the missing edges')
    parser.add_argument("--train_percent", type=int, default=10,
                        help="percentage of training edges. Default is 10.")

    parser.set_defaults(layers_1=[parser.parse_known_args()[0].dimension]*(parser.parse_known_args()[0].order + 1))
    parser.set_defaults(layers_2=[parser.parse_known_args()[0].dimension]*(parser.parse_known_args()[0].order + 1))

    return parser.parse_args()
