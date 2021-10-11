import os
import sys
import shutil
import threading

from glob import glob

img_root = sys.argv[1]
tag_root = sys.argv[2]
dst_root = sys.argv[3]

def get_id(filename, dataroot):
    id = filename.replace(dataroot, '').replace('_crop.jpg', '')
    return int(id)


def processing(img_root, tag_root, dst_root):
    def get_tagfile(thread_id, img_files, img_root, tag_root, dst_root):
        data_size = len(img_files)
        for i in range(data_size):
            img_file = img_files[i]
            id = get_id(img_file, img_root)
            tail = id % 1000
            tag_path = os.path.join(tag_root, '%04d' % tail) + '/'
            tag_file = img_file.replace(img_root, tag_path).replace('_crop.jpg', '.json')
            try:
                shutil.copy(tag_file, tag_file.replace(tag_path, dst_root))
            except:
                os.remove(img_file)
                continue

            if i % 5000 == 0:
                print('thread id: {}, processing [{} / {}]'.format(thread_id, i+1, data_size))


    img_files = glob(os.path.join(img_root, '*.jpg'))
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    data_size = len(img_files)
    print('total image size: {}'.format(data_size))

    num_threads = 8
    thread_size = data_size // num_threads
    if num_threads == 0:
        get_tagfile(0, img_files, img_root, tag_root, dst_root)
    else:
        thread_size = data_size // num_threads
        threads = []
        for t in range(num_threads):
            if t == num_threads - 1:
                thread = threading.Thread(target=get_tagfile, args=(t, img_files[t*thread_size :], img_root, tag_root, dst_root))
            else:
                thread = threading.Thread(target=get_tagfile, args=(t, img_files[t*thread_size : (t+1)*thread_size], img_root, tag_root, dst_root))
            threads.append(thread)
        for t in threads:
            t.start()
        thread.join()

processing(img_root, tag_root, dst_root)