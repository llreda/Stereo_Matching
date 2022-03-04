from __future__ import print_function
import yaml
import utils.Config as Cfg
import logging, logging.config
from utils.runs import *


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a PSMNet')
    parser.add_argument('--maxdisp', type=int, default=96,
                        help='maximum disparity')
    parser.add_argument('--model', default='gwc_gc',
                        help='select model')
    parser.add_argument('--datapath', default='/home/mz/llreda/Stereo_Dataset',
                        help='datapath')
    parser.add_argument('--dataset_csv_root', default=None,
                        help="datasets's .csv root")
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train')
    parser.add_argument('--loadmodel', default=None,
                        help='load model')
    parser.add_argument('--savemodel', default='./gwc',
                        help='save model')
    parser.add_argument('--log', default='./logging.yaml',
                        help='load logging config file')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--device', default='cuda',
                        help='device')
    parser.add_argument('--seed', type=int, default=666, metavar='S',
                        help='random seed (default: 1)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    assert os.path.exists(args.log), f"Logger config file {args.log} does not exist."
    with open(args.log, "r") as f:
        log_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    logging.config.dictConfig(log_cfg)
    filelog = logging.getLogger('filelog')
    if args.savemodel is None:
        args.savemodel = './'
    if not os.path.exists(args.savemodel):
        os.mkdir(args.savemodel)
    hds = filelog.handlers
    for hd in hds:
        if isinstance(hd, logging.FileHandler):
            hd.close()
            loc = os.path.split(hd.baseFilename)[1]
            hd.baseFilename = os.path.join(args.savemodel, loc)

    config = Cfg.Train_Config(args)
    for k, v in config.__dict__.items():
        filelog.info(f'################## {k} = {v}')

    run(config)

