import argparse

# get arguments #
#################
parser = argparse.ArgumentParser(description='patch_encoder for CIFAR-10')
parser.add_argument('-r', type=int, default=None)
parser.add_argument('-c', type=int, default=None)
parser.add_argument('--num_clusters', type=int, default=10)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--beta', type=int, default=1)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--grayscale', dest='grayscale', action='store_true')
parser.add_argument('--colour', dest='grayscale', action='store_false')
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--load', dest='train', action='store_false')
parser.add_argument('--show_graphs', dest='show_graphs',
                    action='store_true')
parser.add_argument('--deterministic',
                    dest='deterministic', action='store_true')

parser.set_defaults(train=True, show_graphs=False,
                    grayscale=True, deterministic=False)

args = parser.parse_args()

# set (hyper)parameters #
#########################

# data parameters
if args.dataset == 'cifar10':
    if args.grayscale:
        input_shape = (32, 32, 1)
    else:
        input_shape = (32, 32, 3)
elif args.dataset == 'ising' or args.dataset == 'mnist' or args.dataset == 'test':
    input_shape = (32, 32, 1)
elif args.dataset == 'dimer':
    input_shape = (32, 32, 4)

# patch parameters
sz = 8
r = args.r
c = args.c

# (patch) encoder/decoder parameters
hidden_dim = 128
num_filters = 32
num_conv = 4
intermediate_dim = 128
latent_dim = 32

# training parameters
batch_size = 128
beta = args.beta
epochs = args.epochs

# show receptive field parameters
num_clusters = args.num_clusters
