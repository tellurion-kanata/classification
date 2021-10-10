import os

from options import Options
from classification import classifier



if __name__ == '__main__':
    opt = Options().get_options()

    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    model = classifier(opt)

    if not opt.eval:
        model.train()
    else:
        model.test()
