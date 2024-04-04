from PIL import Image
import numpy as np
import h5py
import argparse
import glob


def process_image_pair(Im1, Im2, threshold, time):
    im1_a = np.array(Im1, dtype=float)
    im2_a = np.array(Im2, dtype=float)
    diff = np.subtract(im2_a,im1_a)
    events = []
    for x in range(0, diff.shape[0]):
        for y in range(0, diff.shape[1]):
            if(diff[x,y] > threshold):
                diff[x,y] = 255
                events.append((x,y,1,time))
            elif(diff[x,y] < -threshold):
                diff[x,y] = 0
                events.append((x,y,-1,time))
            else:
                diff[x,y] = 127
    return events

def process_folder(dir, threshold):
    images = glob.glob(f"{dir}/*.png")
    print(f"Found {len(images)} images in {dir}")
    events = []
    for i in range(1,len(images)):
        if(i == 1):
            Im1 = Image.open(images[0]).convert('L')
            Im2 = Image.open(images[i]).convert('L')
        else:
            Im1 = Im2
            Im2 = Image.open(images[i]).convert('L')
        events.extend(process_image_pair(Im1,Im2, threshold, i-1))
    events.sort(key=lambda e: e[3])
    f = h5py.File(f'{dir}\events.h5', 'w')
    f.create_dataset('events', data=events)
    f.close()

    print(f"Generated {len(events)} events and stored in {dir}\events.h5")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", help="Directory containing rendered images. The output events will be stored in DIRNAME\events.h5", nargs="+")
    parser.add_argument("-t", "--threshold", help="Contrast threshold to use. Default 5.", type=int, default=5, nargs="?")
    args = parser.parse_args()
    for dir in args.dir:
        process_folder(dir, args.threshold)