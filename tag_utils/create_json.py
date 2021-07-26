import json
import os

metadata_root = 'e:/data/danbooru/danbooru2020/'

tag_dict = {}
for i in range(16):
    filename = os.path.join(metadata_root, str(2020000000000000 + i))
    for line in open(filename, 'r', encoding='utf-8'):
        dict = json.loads(line)
        for tag in dict['tags']:
            tag_dict[tag['name']] = tag_dict.get(tag['name'], 0) + 1

tag_list = sorted(tag_dict.items(), key=lambda tag_dict:tag_dict[1], reverse=True)[:800]
tag_dict = {}
tag_index = {}
for i, tag in enumerate(tag_list):
    tag_dict[tag[0]] = i
    tag_index[i] = tag[0]
with open('mini_tag_index.json', 'w') as file:
    json.dump(tag_index, file, indent=1)
with open('mini_tag_dict.json', 'w') as file:
    json.dump(tag_dict, file, indent=1)
