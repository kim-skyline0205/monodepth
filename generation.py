import argparse
import numpy as np
import cv2
import sys
from PIL import Image

parser = argparse.ArgumentParser(description='NPY File to Image')
parser.add_argument('--npy_taxt_path', type=str, help='directory of npy', required=True)
parser.add_argument('--save_path', type=str, help='directory of save', required=True)
parser.add_argument('--image_taxt_path', type=str, help='directory of save', required=True)

args = parser.parse_args()

def main():

    npy_path_taxtname = args.npy_taxt_path
    save_path = args.save_path
    image_path_taxtname = args.image_taxt_path
  
    image = open(image_path_taxtname, 'r')
    
    npy = open(npy_path_taxtname, 'r')
    
    for i in range(len(43111)):
        image_line = f.readline()
        npy_line = f.readline()

        disparities = np.load(npy_line.split('\n')[0])  
        (minval, maxval, minloc, maxloc) = cv2.minMaxLoc(disparities[0])

        im=Image.open('/media/keti/4tkim/modu/DataSet/KITTI/'+image_line.split(' ')[0])
        image_width,image_height=im.size
        
        size = (int(image_width), int(image_height)) 
        disp = np.array((disparities[0] -minval)/(maxval-minval)*255, dtype = np.uint8)
        disp = cv2.resize(disp, im.size, 0, 0, cv2.INTER_CUBIC)

        cv2.imwrite("{}.png".format(save_path+str(i)), disp)
   
    print("end")  
   
if __name__ == "__main__":
    main()
    
