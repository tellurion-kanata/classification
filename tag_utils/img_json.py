import json
import os

def save_json(path, dict):
    if not os.path.exists(path):
        os.mkdir(path)
    filename = os.path.join(path, str(dict['id']) + '.json')
    with open(filename, 'w') as file:
        json.dump(dict, file, indent=1)

metadata_root = 'e:/data/danbooru/danbooru2020'
save_path = 'e:/data/danbooru/danbooru2020/mini_tags'
tag_path = 'mini_tag_dict.json'

with open(tag_path, 'r', encoding='utf-8') as file:
    tag_dict = json.load(file)
for i in range(16):
    dict_name = 2020000000000000 + i
    dict_path = os.path.join(metadata_root, str(dict_name))
    with open(dict_path, 'r', encoding='utf-8') as file:
        for line in file:
            img_dict = {}
            dict = json.loads(line)
            img_dict['id'] = int(dict['id'])
            img_dict['height'] = dict['image_height']
            img_dict['width'] = dict['image_width']
            tag_list = []
            for tag in dict['tags']:
                tag_idx = tag_dict.get(tag['name'], -1)
                if tag_idx > -1:
                    tag_list.append(tag_idx)
            img_dict['tags'] = tag_list
            tail = img_dict['id'] % 1000
            if tail > 6:
                continue
            path = os.path.join(save_path, '%04d' % tail)
            save_json(path, img_dict)

    print("%d finished" % dict_name)
