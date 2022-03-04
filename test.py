import os
import torch
import yaml
import utils.Config as Cfg
import logging, logging.config
from utils.runs import *


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a PSMNet')
    parser.add_argument('--model', default='constancy',
                        help='select model')
    parser.add_argument('--maxdisp', type=int, default=96,
                        help='maximum disparity')
    parser.add_argument('--datapath', default='/home/mz/llreda/Stereo_Dataset',
                        help='datapath')
    parser.add_argument('--dataset_csv_root', default=None,
                        help="test datasets's .csv root")
    parser.add_argument('--loadmodel', default=None,
                        help='load model')
    parser.add_argument('--savedir', default=None,
                        help='save results')
    parser.add_argument('--log', default='./logging.yaml',
                        help='load logging config file')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.loadmodel is None:
        print('No model to load !!!')
        exit()
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    assert os.path.exists(args.log), f"Logger config file {args.log} does not exist."
    with open(args.log, "r") as f:
        log_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    logging.config.dictConfig(log_cfg)
    filelog = logging.getLogger('filelog')

    if args.savedir is None:
        args.savedir = os.path.join(os.getcwd(), f'test_{os.path.split(args.loadmodel)[1]}')
        if not os.path.exists(args.savedir):
            os.mkdir(args.savedir)
    hds = filelog.handlers
    for hd in hds:
        if isinstance(hd, logging.FileHandler):
            hd.close()
            loc = os.path.split(hd.baseFilename)[1]
            hd.baseFilename = os.path.join(args.savedir, loc)

    config = Cfg.Validate_Config(args)
    for k, v in config.__dict__.items():
        filelog.info(f'################## {k} = {v}')

    TEST(config)