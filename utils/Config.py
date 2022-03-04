from utils import transforms


class Train_Config(object):
    def __init__(self, args, batch_size=2, lr=0.001, downsampling=2, train_sets=[7,],
                 validation_sets=[8,], least_percent_of_valid=0.1):
        super(Train_Config, self).__init__()
        self.model = args.model
        self.datapath = args.datapath
        self.dataset_csv_root = args.dataset_csv_root
        self.maxdisp = args.maxdisp
        self.epochs = args.epochs
        self.loadmodel = args.loadmodel
        self.savemodel = args.savemodel
        self.cuda = args.cuda
        self.seed = args.seed
        self.batch_size = batch_size
        self.lr = lr
        self.downsampling = downsampling
        self.train_sets = train_sets    # 如果修改了训练集和验证集划分方式，则重新获取路径.csv文件并修改配置
        self.validation_sets = validation_sets
        self.gpu_ids = [0]
        self.least_percent_of_valid = least_percent_of_valid
        self.train_transform = transforms.Compose([transforms.RandomColor(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


class Validate_Config(object):
    def __init__(self, args, batch_size=4, downsampling=2, test_sets=[9,], least_percent_of_valid=0.1):
        super(Validate_Config , self).__init__()
        self.model = args.model
        self.datapath = args.datapath
        self.dataset_csv_root = args.dataset_csv_root
        self.loadmodel = args.loadmodel
        self.savedir = args.savedir
        self.maxdisp = args.maxdisp
        self.cuda = args.cuda
        self.batch_size = batch_size
        self.downsampling = downsampling
        self.test_sets = test_sets    # 如果修改了训练集和验证集划分方式，则重新获取路径.csv文件并修改配置
        self.least_percent_of_valid = least_percent_of_valid
        self.gpu_ids = [0]