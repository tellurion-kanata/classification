import os
import sys
import cv2
import argparse
import threading
import warnings
import PIL.Image as Image

from PIL import ImageFile
from glob import glob


def get_path():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='dataroot path')
    parser.add_argument('--print_tag_size', action='store_true', help='print tags file size')
    parser.add_argument('--delete', action='store_true', help='delete images which don\'t have corresponding tag file')
    parser.add_argument('--check_paired', action='store_true')
    parser.add_argument('--check_truncated', action='store_true')
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--size', type=tuple, default=(512, 512))
    return parser.parse_args()


def is_paired(img_file, delete=False):
    if img_file.find('png'):
        os.rename(img_file, img_file.replace('png', 'jpg'))
    tag_file = img_file.replace('image', 'tags').replace('jpg', 'json')

    if not os.path.exists(tag_file):
        print(tag_file)
        if delete:
            os.remove(img_file)
            return True
    return False


def is_truncated(file, delete=False):
    try:
        Image.open(file)
    except:
        print(file)
        if delete:
            os.remove(file)


def resize(file, size):
    width, height = size[:]
    try:
        img = cv2.imread(file)
        img = cv2.resize(img, (width, height))
        cv2.imwrite(file, img)
    except:
        raise IOError('Image is not opened correctly. path:{}'.format(file))


def processing(thread_id, opt, img_files, delete=False):
    data_size = len(img_files)
    if opt.check_paired and opt.check_truncated:
        for i in range(data_size):
            if i % 5000 == 0:
                print('id:{}, step: [{}/{}]'.format(thread_id, i, data_size))
            is_delete = is_paired(img_files[i], delete)
            if not is_delete:
                is_truncated(img_files[i], delete)

    elif opt.check_paired:
        for i in range(data_size):
            is_paired(img_files[i], delete)
            if i % 5000 == 0:
                print('id:{}, step: [{}/{}]'.format(thread_id, i, data_size))

    elif opt.check_truncated:
        for i in range(data_size):
            is_truncated(img_files[i], delete)
            if i % 5000 == 0:
                print('id:{}, step: [{}/{}]'.format(thread_id, i, data_size))

    elif opt.resize:
        for i in range(data_size):
            resize(img_files[i], opt.size)
            if i % 5000 == 0:
                print('id:{}, step: [{}/{}]'.format(thread_id, i, data_size))


def create_threads(opt):
    dataroot = opt.dataroot
    delete = opt.delete
    warnings.filterwarnings("error", category=UserWarning)
    
    img_path = os.path.join(dataroot, 'original')
    tag_path = os.path.join(dataroot, 'tags')
    img_files = glob(os.path.join(img_path, '*/*'))
    tag_files = glob(os.path.join(tag_path, '*/*.json'))
    data_size = len(img_files)
    tag_size = len(tag_files)

    print('total image size: {}, total tag file size: {}'.format(data_size, tag_size))
    thread_num = 8
    thread_size = data_size // thread_num
    threads = []

    for t in range(thread_num):
        if t == thread_num - 1:
            thread = threading.Thread(target=processing, args=(t, opt, img_files[t*thread_size :], delete))
        else:
            thread = threading.Thread(target=processing, args=(t, opt, img_files[t*thread_size : (t+1)*thread_size], delete))
        threads.append(thread)
    for t in threads:
        t.start()
    thread.join()


if __name__ == '__main__':
    opt = get_path()
    create_threads(opt)

