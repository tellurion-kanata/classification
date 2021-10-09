import argparse

def get_options():
    parser = argparse.ArgumentParser()
    # overall options
    parser.add_argument('--name', required=True)
    parser.add_argument('--dataroot', required=True, help='Training dataset')
    parser.add_argument('--model', default='resnet50', help='Type of training networks [resnet50 (default) | resnet34 | vgg16]')
    parser.add_argument('--batch_size', '-bs', type=int, default=32, help='Number of batch size')
    parser.add_argument('--num_classes', type=int, default=8000, help='Number of class')
    parser.add_argument('--load_size', type=int, default=512, help='Size of image')
    parser.add_argument('--load_model', action='store_true', help='Load existed model')
    parser.add_argument('--load_epoch', type=str, default='latest', help='Epoch to load')
    parser.add_argument('--image_pattern', '-p', type=str, default='*.jpg', help='pattern of training images')
    parser.add_argument('--save_path', '-s', type=str, default='./checkpoints', help='Training record path')
    parser.add_argument('--pretrained_model', '-pre', type=str, default=None, help='Pre-trained model filename for classifier')

    #training options
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
    parser.add_argument('--no_flip', action='store_true', help='Not to flip image')
    parser.add_argument('--no_rotate', action='store_true', help='Not to rotate image')
    parser.add_argument('--save_model_freq', type=int, default=5, help='Saving network states per epochs')
    parser.add_argument('--save_model_freq_step', type=int, default=5000, help='Saving latest network states per steps')
    parser.add_argument('--print_state_freq', type=int, default=1000, help='Print training states per iterations')
    parser.add_argument('--device', type=int, default=0, help='GPU used in training')
    parser.add_argument('--num_threads', type=int, default=0, help='Number of threads when reading data')
    parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')

    #evaluation options
    parser.add_argument('--eval', action='store_true', help='Evaluation mode')
    parser.add_argument('--gt_label', action='store_true', help='Test data includes ground truth labels')
    parser.add_argument('--score_thres', type=float, default=0.5, help='Threshold of score')
    parser.add_argument('--top_k', type=int, default=20, help='Top k predicted classes')

    return parser.parse_args()
