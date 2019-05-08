import sys
import random
import os.path as osp
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

def img_test():
    ids = []
    root_path = "/data/stu18/MIC935"
    for line in open(osp.join(root_path, 'ImageSets', 'Main', 'train.txt')):
        ids.append((root_path, line.strip()))
    # print("test: ", ids[102])

    _annopath = osp.join('%s', 'Annotations', '%s.xml')
    _imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')

    for id in ids:
        # id = ('/data/stu18/MIC935', '000500')
        target = ET.parse(_annopath % id).getroot()
        # print(target)
        if target.find('object') == None:
            print("None")
            continue
        for obj in target.iter('object'):
            # print(obj)
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text)
                print(pt, cur_pt)

def create_train_val_file(filename):
    root_path = "/data/stu18/MIC935"
    ids = []
    trainval_ids = []
    for i in range(1, 935):
        ids.append("%06d" % i)

    _annopath = osp.join('%s', 'Annotations', '%s.xml')

    for id in ids:
        tmp_id = (root_path, str(id))
        target = ET.parse(_annopath % tmp_id).getroot()
        if target.find('object') is not None:
            trainval_ids.append(id)

    # print(trainval_ids)
    # print(len(trainval_ids))

    file = open(filename, "w+")
    for i, id in enumerate(trainval_ids):
        if i > 300:
            break
        else:
            file.write(str(id)+'\n')
    file.close()

def shuffle_file(filename):
    f = open(filename, 'r+')
    lines = f.readlines()
    random.shuffle(lines)
    f.seek(0)
    f.truncate()
    f.writelines(lines)
    f.close()

if __name__ == '__main__':
    filename = "/data/stu18/MIC935/ImageSets/Main/trainval_hotown.txt"
    create_train_val_file(filename)
    shuffle_file(filename)