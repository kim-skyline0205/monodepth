import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='NPY File to Image')
parser.add_argument('--npy_path', type=str, help='directory of npy', required=True)
parser.add_argument('--save_path', type=str, help='directory of save', required=True)
parser.add_argument('--width', type=str, help='image width', required=True, default=1226)
parser.add_argument('--height', type=str, help='image height', required=True, default=370)

args = parser.parse_args()

def main():

    npy_path = args.npy_path
    save_path = args.save_path
    image_width =args.width
    image_height =args.height
    disparities = np.load(npy_path)  
    size = (int(image_width), int(image_height))

    for index in range(len(disparities)):
        (minval, maxval, minloc, maxloc) = cv2.minMaxLoc(disparities[index])
        disp = np.array((disparities[index] -minval)/(maxval-minval)*255, dtype = np.uint8)
        disp = cv2.resize(disp, size, 0, 0, cv2.INTER_CUBIC)
        colored_disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

        cv2.imwrite("{}.png".format(save_path+str(index)), disp)
    print("end")    
if __name__ == "__main__":
main()
