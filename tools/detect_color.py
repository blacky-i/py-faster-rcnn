import os
#import pgmagick as pg
from collections import namedtuple
from math import sqrt
import numpy as np
import Tkinter as tk
from PIL import Image, ImageTk
import json
import random
import struct
    
import csv

from colormath.color_objects import LabColor,sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff_matrix import delta_e_cie2000

from sklearn.cluster import KMeans, DBSCAN, MeanShift,estimate_bandwidth
from sklearn.utils import shuffle
from time import time
from skimage import io
from collections import Counter
import scipy.misc
def scilearn_cluster(filename,k,son_object,index):
    image_to_analyze = io.imread(filename)
    image_to_analyze = image_to_analyze[son_object['bboxes'][index]['x_left']:son_object['bboxes'][index]['x_right'],son_object['bboxes'][index]['y_left']:son_object['bboxes'][index]['y_right']]
    #china = load_sample_image('china.jpg')
    image_to_analyze = np.array(image_to_analyze, dtype=np.float64) / 255
    w, h, d = original_shape = tuple(image_to_analyze.shape)
    assert d == 3
    image_array = np.reshape(image_to_analyze, (w * h, d))


    print("Fitting model on a small sub-sample of the data")
    t0 = time()
    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=k, random_state=0).fit(image_array_sample)
    print("done in %0.3fs." % (time() - t0))

    # Get labels for all points
    print("Predicting color indices on the full image (k-means)")
    t0 = time()
    labels = kmeans.predict(image_array)
    print("done in %0.3fs." % (time() - t0))
    
    counter = Counter(labels)
    colors_array =[]
    colors_result=[]

    for label in counter.keys():
        colors_array.append((label,counter[label]))
    
    sorted_colors_array = sorted(colors_array, key=lambda color_count: color_count[1])
    for i in xrange(0,10):
        colors_result.append(255*kmeans.cluster_centers_[sorted_colors_array.pop()[0]])
    return colors_result
        #return clusters
def dbscan_cluster(filename,son_object,index):
    #image_to_analyze = io.imread(filename)
    _image = Image.open(filename)
    bbox_array=[]
    bbox_array.append(son_object['bboxes'][index]['x_left'])
    bbox_array.append(son_object['bboxes'][index]['y_left'])
    bbox_array.append(son_object['bboxes'][index]['x_right'])
    bbox_array.append(son_object['bboxes'][index]['y_right'])

    _image=_image.crop(bbox_array)


    #im = Image.open(filename)

    #image_to_analyze = image_to_analyze[son_object['bboxes'][index]['x_left']:son_object['bboxes'][index]['x_right'],son_object['bboxes'][index]['y_left']:son_object['bboxes'][index]['y_right']]
    image_to_analyze=np.array(_image.getdata(),
                    np.uint8).reshape(_image.size[1], _image.size[0], 3)
   # print np.array(im.getdata(),
   #                 np.uint8).reshape(im.size[1], im.size[0], 3)

    w_orig, h_orig = _image.size
    square_width=128
    if w_orig<square_width and h_orig<square_width:
        square_width=64
    x_init = w_orig/3-square_width/2
    y_init = h_orig/3-square_width/2
    bbox_array=[]
    bbox_array.append(x_init)
    bbox_array.append(y_init)
    bbox_array.append(x_init+square_width)
    bbox_array.append(y_init+square_width)
    _image=_image.crop(bbox_array)
    image_to_analyze=np.array(_image.getdata(),
                    np.uint8).reshape(_image.size[1], _image.size[0], 3)

   # image_to_analyze = image_to_analyze[x_init:x_init+square_width,y_init:y_init+square_width]
  

        #china = load_sample_image('china.jpg')
    image_to_analyze = np.array(image_to_analyze, dtype=np.float64) / 255
    w, h, d = original_shape = tuple(image_to_analyze.shape)
    assert d == 3
    image_array = np.reshape(image_to_analyze, (w * h, d))

    t0 = time()
    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    bandwidth = estimate_bandwidth(image_array, quantile=0.2, n_samples=500)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(image_array)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    labels_counters = Counter(labels)
    n_clusters_ = len(labels_unique)
    return 255*cluster_centers
#     dbscan = DBSCAN(eps=0.5,min_samples=50).fit(image_array)
#     print("done in %0.3fs." % (time() - t0))
    
#     core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
#     core_samples_mask[dbscan.core_sample_indices_] = True
#     labels = dbscan.labels_
#     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#     print('Estimated number of clusters: %d' % n_clusters_) 

#  #   dbscan.fit_predict(image_array)

#     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#     print('Estimated number of clusters: %d' % n_clusters_) 

#     # Number of clusters in labels, ignoring noise if present.
#     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#     print('Estimated number of clusters: %d' % n_clusters_) 

colors =["blue","black","gray","white","brown","red","green","multicolor","beige","purple","silver","orange","pink","yellow","gold","khaki","animal","floral","transparent","teal"]

def contains_in_colors(array,colorname):
    result=False
    for row in array:
        if row['colorname']==colorname:
            result=True
    return result

def remove_background(filename,indices_array,json_object):
    """ Remove the background of the image in 'filename' """
    filename = filename

    index=1
    # for idx,row in enumerate(indices_array):
    #     colors = colorz(filename,3,indices_array[index]['indx'],json_object)
    #     rgb_color_vector = [struct.unpack('BBB',colors[0][1:].decode('hex'))[0],struct.unpack('BBB',colors[0][1:].decode('hex'))[1],struct.unpack('BBB',colors[0][1:].decode('hex'))[2]]
    #     print get_color_name(rgb_color_vector),colors[0]
    # for idx,row in enumerate(indices_array):
    #     rgb_colors=scilearn_cluster(filename,64,j,indices_array[idx]['indx'])
    #     rgb_color_vector = [rgb_colors[0][0],rgb_colors[0][1],rgb_colors[0][2]]
    recognized_colors=[]

    for index in xrange(0,len(indices_array)):
        colors=[]
        rgb_color_vector=[]
        rgb_colors=dbscan_cluster(filename,json_object,indices_array[index]['indx'])
        for color_item in rgb_colors:
            rgb_color_vector.append([color_item[0],color_item[1],color_item[2]])

        for i in xrange(0,len(rgb_colors)):
            temp = (rgb_colors[i][0],rgb_colors[i][1],rgb_colors[i][2])
            colors.append('#'+struct.pack('BBB',*temp).encode('hex'))
        k=0
        rec_color=get_color_name(rgb_color_vector[0])[0]  
        while contains_in_colors(recognized_colors,rec_color)==True and k<len(rgb_color_vector):
            k=k+1
            rec_color = get_color_name(rgb_color_vector[k])[0]  
        color = {'colorname':rec_color,'color_hex':colors[0],'classname':indices_array[index]['classname']}
        recognized_colors.append(color)
    # use a Tkinter label as a panel/frame with a background image
    # note that Tkinter only reads gif and ppm images
    # use the Python Image Library (PIL) for other image formats
    # free from [url]http://www.pythonware.com/products/pil/index.htm[/url]
    # give Tkinter a namespace to avoid conflicts with PIL
    # (they both have a class named Image)
    return recognized_colors
    root = tk.Tk()
    root.title('background image')

    # pick an image file you have .bmp  .jpg  .gif.  .png
    # load the file and covert it to a Tkinter image object
    imageFile = filename
    img_uncropped=Image.open(imageFile)
    bbox_array=[]
    bbox_array.append(json_object['bboxes'][indices_array[index]['indx']]['x_left'])
    bbox_array.append(json_object['bboxes'][indices_array[index]['indx']]['y_left'])
    bbox_array.append(json_object['bboxes'][indices_array[index]['indx']]['x_right'])
    bbox_array.append(json_object['bboxes'][indices_array[index]['indx']]['y_right'])

    imageToRender=img_uncropped.crop(bbox_array)

    render = imageToRender.resize((800,600))
    image1 = ImageTk.PhotoImage(render)

    # get the image size
    w = 800#image1.width()
    h = 600#image1.height()

    # position coordinates of root 'upper left corner'
    x = 0
    y = 0

    # make the root window the size of the image
    root.geometry("%dx%d+%d+%d" % (w, h, x, y))

    # root has no image argument, so use a label as a panel
    panel1 = tk.Label(root, image=image1)
    panel1.pack(side='top', fill='both', expand='yes')

    button1 = tk.Button(panel1, text='button2', background=colors[0])
    button1.pack(side='bottom')
    label1 = tk.Label(panel1,text=colors[2])
    label1.pack(side='bottom')

    button2 = tk.Button(panel1, text='button2', background=colors[1])
    button2.pack(side='bottom')
    label2 = tk.Label(panel1,text=colors[1])
    label2.pack(side='bottom')

    button3 = tk.Button(panel1, text='button2', background=colors[2])
    button3.pack(side='bottom')
    label3 = tk.Label(panel1,text=colors[0])
    label3.pack(side='bottom')
    

    # save the panel's image from 'garbage collection'
    panel1.image = image1

    # start the event loop
    root.mainloop()
def get_color_name(rgb_vector):
    ##Find color - https://making.lyst.com/2014/02/22/color-detection/
    # load list of 1000 random colors from the XKCD color chart

    # with open('colors.csv', 'rb') as csvfile:
    #     reader = csv.reader(csvfile)
    #     rgb_matrix = np.array([map(float, row[0:3]) for row in reader],dtype=float)
 
    # with open('colors.csv', 'rb') as csvfile:
    #     reader = csv.reader(csvfile)
    #     color_labels= np.array([row[3:] for row in reader])
    #     # the reference color
    #     lab_matrix=np.array([convert_color(sRGBColor(row[0],row[1],row[2],is_upscaled=True),LabColor) for row in rgb_matrix])
    # print lab_matrix
    # with open('lab_colors.csv', 'wb') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',',
    #                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     for idx,row in enumerate(lab_matrix):
    #          writer.writerow([row.lab_l,row.lab_a,row.lab_b,color_labels[idx][0]]) 
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'lab_colors.csv'), 'rb') as csvfile:
        reader = csv.reader(csvfile)
        lab_matrix = np.array([map(float, row[0:3]) for row in reader],dtype=float)
 
    color_index=0
    lab_color=convert_color(sRGBColor(rgb_vector[0],rgb_vector[1],rgb_vector[2],is_upscaled=True),LabColor)
    #          writer.writerow([row.lab_l,row.lab_a,row.lab_b,color_labels[idx][0]]) 
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'lab_colors.csv'), 'rb') as csvfile:
        reader = csv.reader(csvfile)
        color_labels= np.array([row[3:] for row in reader])

    _color=[]
    _color.append(lab_color.lab_l)
    _color.append(lab_color.lab_a)
    _color.append(lab_color.lab_b)


    nearest_color= lab_matrix[np.argmin(delta_e_cie2000(_color,lab_matrix))]
    color_idx = 0
    for idx,item in enumerate(lab_matrix):
        if item[0] == nearest_color[0] and item[1] == nearest_color[1]  and  item[2] == nearest_color[2] :
            color_idx=idx
    # Find the color difference
    #print '%s OR %s is closest to %s %s' % (rgb_vector,_color, nearest_color,color_labels[color_idx])
    return color_labels[color_idx]

def detect_colors(jsonfile):
    path_in_json=jsonfile['file']
    
    useful_indices=[]

    for idx,classname in enumerate(jsonfile['classes']):
        if classname != "__background__":
            useful_indices_item={'indx':idx,'classname':classname}
            useful_indices.append(useful_indices_item)
    
    #image_dir_path="/home/bitummon/Projects/datasets/fashion/fashion2017_800/images"


    #image_filepath= os.path.join(image_dir_path, path_in_json.rsplit('/').pop())
    image_filepath=path_in_json
    return remove_background(image_filepath,useful_indices,jsonfile)


if __name__ == "__main__":
    path_to_jsonfile="/home/bitummon/tests/colors/00041.json"
    f=file(path_to_jsonfile,'r')
    j=json.load(f)
    f.close()
    detect_colors(j)
