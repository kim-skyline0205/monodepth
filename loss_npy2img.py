import argparse
import numpy as np
import cv2
import os

parser = argparse.ArgumentParser(description='NPY File to Image')
parser.add_argument('--npy_path', type=str, help='directory of npy', required=True)
parser.add_argument('--save_path', type=str, help='directory of save', required=True)
parser.add_argument('--width', type=str, help='image width', required=True, default=1226)
parser.add_argument('--height', type=str, help='image height', required=True, default=370)

args = parser.parse_args()

def main():
    cnt=0
    npy_path = args.npy_path
    save_path = args.save_path
    image_width =args.width
    image_height =args.height
    
    npy_list=os.listdir(npy_path)
    npy_list.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    for item in npy_list:
        if(item.find('.npy')):
         print(item)
         disparities = np.load(npy_path+item)  
         disparities_R = disparities[:,:,:,0]
         disparities_G = disparities[:,:,:,1]
         disparities_B = disparities[:,:,:,2]
         print(disparities.shape) 
         size = (int(image_width), int(image_height))
         for index in range(len(disparities)):
             (minval, maxval, minloc, maxloc) = cv2.minMaxLoc(disparities_R[index])
             disp_R = np.array((disparities_R[index] -minval)/(maxval-minval)*255, dtype = np.uint8)

             (minval, maxval, minloc, maxloc) = cv2.minMaxLoc(disparities_G[index])
             disp_G = np.array((disparities_G[index] -minval)/(maxval-minval)*255, dtype = np.uint8)
        
             (minval, maxval, minloc, maxloc) = cv2.minMaxLoc(disparities_B[index])
             disp_B = np.array((disparities_B[index] -minval)/(maxval-minval)*255, dtype = np.uint8)
        
             disp = (disp_R+disp_G+disp_B)/3
             disp = cv2.resize(disp, size, 0, 0, cv2.INTER_CUBIC)
       # colored_disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
             
             cv2.imwrite("{}.png".format(save_path+str(cnt)), disp)
             cnt=cnt+1
         print("end")    
if __name__ == "__main__":
    main()
