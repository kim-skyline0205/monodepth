import os
import numpy as np
import cv2
import math
np.set_printoptions(threshold=np.nan)

def _sample(im, x_offset, _height, _width, _num_channels, _height_f, _width_f):
    x_t = np.tile(np.linspace(0.0,   _width_f - 1.0,  _width), (_height, 1))
    x_t_i = np.clip(np.round(x_t - x_offset), 0., _width_f - 1.0).astype(np.int32)
    gen = np.zeros(im.shape)
    mask = np.zeros(im.shape[0:-1])
    element = np.linspace(0.0,   _width_f - 1.0,  _width)
    for i in range(_height):
        mask[i,:] = np.in1d(element, x_t_i[i])
        gen[i, x_t_i[i,:], :] = im[i, :, :]

    mask = np.expand_dims(mask, axis=2)
    fill = np.zeros(im.shape)
    fill_mask = (gen == 0)
    fill = gen.copy()
    for i in range(1,101):
        # cpow = math.pow(2,i)
        # fill = fill * np.logical_not(fill_mask) + fill_mask * cv2.resize(cv2.resize(gen,None,fx=1./cpow, fy=1./cpow, interpolation = cv2.INTER_NEAREST), (_width,_height), fx=cpow, fy=cpow, interpolation = cv2.INTER_AREA)
        fill = fill * np.logical_not(fill_mask) + fill_mask * cv2.resize(cv2.resize(gen,None,fx=1./i, fy=1./i, interpolation = cv2.INTER_NEAREST), (_width,_height), fx=i, fy=i, interpolation = cv2.INTER_NEAREST)
        fill_mask = (fill == 0)
    gen = gen * mask + fill * np.logical_not(mask)
    
    return gen.astype(np.uint8)

baseline = 22.
dataset = "NYU"
img_list = open("/home/cvlab/data/{}/img.txt".format(dataset))
disp_list = open("/home/cvlab/data/{}/disp.txt".format(dataset))
save_dir = "/home/cvlab/data/{}/gen/".format(dataset)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
train_filename = "/home/cvlab/data/{}/train.txt".format(dataset)
ext = "jpg"
#train_file = open(train_filename, 'w')

file_list = []
file_list.append(img_list)
file_list.append(disp_list)

scaler = 0
if dataset == "NYU":
    scaler = baseline / 100
elif dataset == "KITTI":
    scaler = baseline / 54

i = 0
for img_path, disp_path in zip(*file_list):
    x_offset = cv2.imread(disp_path.rstrip(), cv2.IMREAD_ANYDEPTH)

    if dataset == "NYU":
        x_offset = x_offset / 255. * 10
        x_offset = scaler * 582.6 / x_offset
    elif dataset == "KITTI":
        x_offset = x_offset * scaler

    input_images = cv2.imread(img_path.rstrip())

    _height       = np.shape(input_images)[0]
    _width        = np.shape(input_images)[1]
    _num_channels = np.shape(input_images)[2]
    _height_f = float(_height)
    _width_f  = float(_width)

    output = _sample(input_images, x_offset, _height, _width, _num_channels, _height_f, _width_f) 

    save_path = save_dir + "{}.{}".format(str(i).zfill(8), ext)
    i += 1
  #  train_file.write(img_path.rstrip() + " " + save_path + "\n")
    cv2.imwrite(save_path, output)

train_file.close()
img_list.close()
disp_list.close()
