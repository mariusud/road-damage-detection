import xml.etree.ElementTree as ET
import os
import glob
import sys
import shutil
import numpy as np

# for file in glob.glob(os.path.join("SSD/datasets/RDD2020_filtered/Annotations", "*.xml")):
#     # open xml
#     tree = ET.parse(file)
#     root = tree.getroot()
#     object_root = root.find("object")
#     label = object_root.find("name")
#     print(label.text)
#     # print(str(file).split('\\')[1].lstrip().split('.')[0])
#     shutil.copy(file,"SSD/datasets/RDD2020_filtered/%s/Annotations" % label.text)
#     shutil.copy(os.path.join("SSD/datasets/RDD2020_filtered/JPEGImages/%s.jpg" % (str(file).split('\\')[1].lstrip().split('.')[0])), "SSD/datasets/RDD2020_filtered/%s/JPEGImages" % label.text)

delta_x = []
delta_y = []
for file in glob.glob(os.path.join("SSD/datasets/RDD2020_filtered/D40/Annotations", "*.xml")):
    # open xml
    tree = ET.parse(file)
    root = tree.getroot()
    object_root = root.find("object")
    bndbox = object_root.find("bndbox")
    xmin = int(bndbox.find("xmin").text)
    ymin = int(bndbox.find("ymin").text)
    xmax = int(bndbox.find("xmax").text)
    ymax = int(bndbox.find("ymax").text)
    delta_x.append(xmax-xmin)
    delta_y.append(ymax-ymin)

mean_delta_x = np.mean(np.asarray(delta_x))
mean_delta_y = np.mean(np.asarray(delta_y))

print(mean_delta_x)
print(mean_delta_y)
