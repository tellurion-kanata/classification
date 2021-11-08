import argparse
import os

class Options() :
    def __init__ (self):
        super(Options, self).__init__()
        parser = argparse.ArgumentParser()

        # overall options
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataroot', required=True, help='Training dataset')
        parser.add_argument('--model', default='resnet50', help='Type of training networks [resnet50 (default) | resnet34 | vgg16]')
        parser.add_argument('--batch_size', '-bs', type=int, default=32, help='Number of batch size')
        parser.add_argument('--num_classes', type=int, default=6000, help='Number of class')
        parser.add_argument('--load_size', type=int, default=320, help='Size of image')
        parser.add_argument('--crop_size', type=int, default=288, help='Crop size')
        parser.add_argument('--load_model', action='store_true', help='Load existed model')
        parser.add_argument('--load_epoch', type=str, default='latest', help='Epoch to load')
        parser.add_argument('--image_pattern', '-p', type=str, default='*.jpg', help='pattern of training images')
        parser.add_argument('--save_path', '-s', type=str, default='./checkpoints', help='Training record path')
        parser.add_argument('--pretrained_model', '-pre', type=str, default=None, help='Pre-trained model filename for classifier')

        #training options
        parser.add_argument('--loss', default='binary', help='Loss function type [binary | focal]')
        parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='Learning rate')
        parser.add_argument('--lr_policy', type=str, default='lambda', help='Policy of learning rate decay')
        parser.add_argument('--optimizer', type=str, default='adam', help='Training optimizer [adam (default) | sgd]')
        parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
        parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for Adam')
        parser.add_argument('--start_step', type=int, default=0, help='Start step')
        parser.add_argument('--epoch_count', type=int, default=1, help='Start epoch')
        parser.add_argument('--no_shuffle', action='store_true', help='Not to shuffle every epoch')
        parser.add_argument('--no_resize', action='store_true', help='Not to resize image')
        parser.add_argument('--no_crop', action='store_true', help='Not to crop image')
        parser.add_argument('--no_flip', action='store_true', help='Not to flip image')
        parser.add_argument('--no_rotate', action='store_true', help='Not to rotate image')
        parser.add_argument('--save_model_freq', type=int, default=5, help='Saving network states per epochs')
        parser.add_argument('--save_model_freq_step', type=int, default=5000, help='Saving latest network states per steps')
        parser.add_argument('--print_state_freq', type=int, default=1000, help='Print training states per iterations')
        parser.add_argument('--device', type=int, default=0, help='GPU used in training')
        parser.add_argument('--num_threads', type=int, default=0, help='Number of threads when reading data')
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--reweight', '-rw', action='store_true', help='Assign weights based on label distribution')

        #evaluation options
        parser.add_argument('--eval', action='store_true', help='Evaluation mode')
        parser.add_argument('--gt_label', action='store_true', help='Test data includes ground truth labels')
        parser.add_argument('--score_thres', type=float, default=0.5, help='Threshold of score')
        parser.add_argument('--top_k', type=int, default=20, help='Top k predicted classes')

        self.parser = parser

    def mkdirs(self, opt):
        def mkdir(path):
            if not os.path.exists(path):
                os.mkdir(path)

        opt.ckpt_path = os.path.join(opt.save_path, opt.name)
        opt.sample_path = os.path.join(opt.ckpt_path, 'sample')
        opt.test_path = os.path.join(opt.ckpt_path, 'test')

        mkdir(opt.save_path)
        mkdir(opt.ckpt_path)
        mkdir(opt.sample_path)
        mkdir(opt.test_path)


    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
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
        print(message)

        # save to the disk
        phase = 'train' if not opt.eval else 'evaluation'
        mode = 'at' if not opt.eval else 'wt'
        file_name = os.path.join(opt.ckpt_path, '{}_opt.txt'.format(phase))
        with open(file_name, mode) as opt_file:
            opt_file.write(message)
            opt_file.write('\n')


    def get_options(self):
        opt = self.parser.parse_args()
        self.mkdirs(opt)
        self.print_options(opt)

        return opt
