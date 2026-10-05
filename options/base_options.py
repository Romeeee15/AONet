import argparse

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing.
    #It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.opt_msg = ''

    def initialize(self, parser):
        # parser.add_argument('--dataroot', default='path to data', help='path to data')
        parser.add_argument('--dataroot', default='Path to data', help='path to data')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids')
        parser.add_argument('--method', type=str, default='AONet', help='AONet')
        parser.add_argument('--train_ids',type=list,default=[0, 120],help='train id number')
        parser.add_argument('--val_ids',type=list,default=[120, 160],help='val id number')
        parser.add_argument('--test_ids',type=list,default=[160, 200],help='test id number')
        parser.add_argument('--modality_filename', type=list, default=['OCTA_3mm'], help='dataroot/dataset filename')
        parser.add_argument('--data_size', type=list, default=[256, 304, 304], help='input data size separated with comma')


        parser.add_argument('--label_filename', type=list, default=['GT_CAVF'], help='dataroot/dataset filename')
        parser.add_argument('--slicelen', default=320)
        parser.add_argument('--data_clip', type=list, default=[90, 346], help='input data size clipped')
        parser.add_argument('--in_channels', type=int, default=256, help='input channels')
        parser.add_argument('--n_classes', type=int, default=5, help='class number')
        parser.add_argument('--saveroot', default='./log', help='path to save models and results')
        parser.add_argument('--dataset', default='OCTA', help='DATASET')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once)."""
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        # print(message)
        # print('')
        # self.opt_msg = message
        return message

    def parse(self):
        """Parse our options"""
        opt = self.gather_options()
        opt.opt_msg = self.print_options(opt)
        self.opt = opt
        return self.opt



