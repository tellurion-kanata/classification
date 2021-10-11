import json
import os
import cv2
import argparse
import threading
import warnings
import PIL.Image as Image

from glob import glob


def get_path():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', '-d', required=True, help='dataroot path')
    parser.add_argument('--print_tag_size', action='store_true', help='print tags file size')
    parser.add_argument('--delete', action='store_true', help='delete images which don\'t have corresponding tag file')
    parser.add_argument('--check_paired', action='store_true')
    parser.add_argument('--check_truncated', action='store_true')
    parser.add_argument('--tags_threshold', '-ts', default=10, help='minimum number of tags')
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--size', type=tuple, default=(512, 512))
    parser.add_argument('--num_threads', default=8, type=int, help='number of threads for processing data')
    return parser.parse_args()


# delete image which doesn't have a paired json file and enough tags
def is_paired(img_file, ts=8, delete=False):
    if img_file.find('png'):
        new_file = img_file.replace('png', 'jpg')
        os.rename(img_file, new_file)
        img_file = new_file
    tag_file = img_file.replace('color', 'tags').replace('jpg', 'json')

    if not os.path.exists(tag_file):
        print(tag_file)
        if delete:
            os.remove(img_file)
            return None, True

    with open(tag_file, 'r') as f:
        img_dict = json.load(f)
    if len(img_dict['tags']) < ts:
        if delete:
            os.remove(img_file)
            return None, True

    return img_file, False


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
    ts = opt.tags_threshold

    if opt.check_paired and opt.check_truncated:
        for i in range(data_size):
            if i % 5000 == 0:
                print('id:{}, step: [{}/{}]'.format(thread_id, i, data_size))
            filename, is_delete = is_paired(img_files[i], ts, delete)
            if not is_delete:
                is_truncated(filename, delete)

    elif opt.check_paired:
        for i in range(data_size):
            is_paired(img_files[i], ts, delete)
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
    
    img_path = os.path.join(dataroot, 'color')
    tag_path = os.path.join(dataroot, 'tags')
    img_files = glob(os.path.join(img_path, '*/*'))
    tag_files = glob(os.path.join(tag_path, '*/*.json'))
    data_size = len(img_files)
    tag_size = len(tag_files)

    print('total image size: {}, total tag file size: {}'.format(data_size, tag_size))
    num_threads = opt.num_threads

    if num_threads == 0:
        processing(0, opt, img_files, delete)
    else:
        thread_size = data_size // num_threads
        threads = []
        for t in range(num_threads):
            if t == num_threads - 1:
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

