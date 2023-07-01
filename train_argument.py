import argparse

def parser():
   #This creates the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=["Base_CNN", "CNN_simple"], default="Base_CNN", #This adds an argument to the parser
        help='Which model to use') #These arguments can be access with args.name, where args = parser.parse_args()
    parser.add_argument('--dataset', choices=['mnist'], default ='mnist')
    parser.add_argument('--data_root', default='data', 
        help='the directory to save the dataset')
    parser.add_argument('--log_root', default='log', 
        help='the directory to save the logs or other imformations (e.g. images)')
    parser.add_argument('--model_root', default='checkpoint', help='the directory to save the models')
    parser.add_argument('--load_checkpoint', default='./model/default/model.pth')
    parser.add_argument('--affix', default='natural_train', help='the affix for the save folder')
    

    ## Training realted 
    parser.add_argument('--num_clients', '-N', type=int, default=50, help='number of clients')
    parser.add_argument('--n_bit', type = int, default = 16, help = 'quantization level for local training')
    parser.add_argument('--m_bit', type = int, default = 16, help = 'quantization level for transmission')
    parser.add_argument('--schedulingsize', type=int, default = 5, help = 'how many clients will be sampled')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size')
    parser.add_argument('--comm_rounds', '-m_e', type=int, default=200, 
        help='the maximum communication rounds')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum(defalt: 0.9)")
    parser.add_argument('--gpu', '-g', default='0', help='which gpu to use')
    parser.add_argument('--seed', default=1, help='The random seed')
    parser.add_argument('--alpha', type=float, default=0.1, help="Dirichelet concentration parameter")
    parser.add_argument('--weight_decay', type=float, default=0., help="SGD weight decay(defalt: 0.)")
    parser.add_argument('--local_epoch', type=int, default = 1, help = "number of local iterations (default = 5)")

    return parser.parse_args()

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))