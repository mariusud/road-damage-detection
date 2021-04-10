import xml.etree.ElementTree as ET
import os
import glob
import sys
import shutil
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt

# for file in glob.glob(os.path.join("datasets/RDD2020_filtered/Annotations", "*.xml")):
#     tree = ET.parse(file)
#     root = tree.getroot()
#     for node in tree.findall('.//object/name'):
#         label = node.text
#         shutil.copy(file,"datasets/RDD2020_filtered/%s/Annotations" % label)
#         shutil.copy(os.path.join("datasets/RDD2020_filtered/JPEGImages/%s.jpg" % (str(file).split('\\')[1].lstrip().split('.')[0])), "datasets/RDD2020_filtered/%s/JPEGImages" % label)

delta_x = []
delta_y = []
area = []
for file in glob.glob(os.path.join("datasets/RDD2020_filtered/D40/Annotations", "*.xml")):
    tree = ET.parse(file)
    root = tree.getroot()
    for node in tree.findall('.//object'):
        if (node.find("name").text=="D40"):
            bndbox = node.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            # if(xmax-xmin<600 and ymax-ymin<600):
            delta_x.append(xmax-xmin)
            delta_y.append(ymax-ymin)
            area.append((xmax-xmin)*(ymax-ymin))
            # if(xmax-xmin>600 or ymax-ymin>600):
            #     print (file)

delta_x = np.asarray(delta_x)
delta_y = np.asarray(delta_y)
area = np.asarray(area)

# min_pos_x = np.argmin(delta_x)
# min_pos_y = np.argmin(delta_y)
# min_x = np.min(delta_x)
# min_y = np.min(delta_y)
# print('minx:',min_x,delta_y[min_pos_x])
# print('miny:',delta_x[min_pos_y],min_y)

# max_pos_x = np.argmax(delta_x)
# max_pos_y = np.argmax(delta_y)
# max_x = np.max(delta_x)
# max_y = np.max(delta_y)
# print('maxx:',max_x,delta_y[max_pos_x])
# print('maxy:',delta_x[max_pos_y],max_y)

# mean_delta_x = np.mean(delta_x)
# mean_delta_y = np.mean(delta_y)
# new_array = np.vstack((delta_x,delta_y))

# print(mean_delta_x)
# print(mean_delta_y)

# ratio_vector = np.zeros(np.shape(new_array)[1])
# for i in range(np.shape(new_array)[1]):
#     ratio_vector[i] = round(new_array[0][i]/new_array[1][i])
# _, counts = np.unique(ratio_vector,return_counts=True)
# plt.bar(np.arange(len(counts)), counts)
# print(counts)
# plt.show()
print(np.shape(area))
plt.hist(area)
plt.show()

