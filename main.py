import cv2
import os
from merge import merge

all_pano_dirs = [name for name in os.listdir('HW2_Dataset') if os.path.isdir(os.path.join('HW2_Dataset', name)) ]

for pano in all_pano_dirs:
    continue_merging = '-'
    images_in_pano = os.listdir(os.path.join('HW2_Dataset', pano))
    img_left = cv2.imread(os.path.join('HW2_Dataset', pano, images_in_pano[0]))
    img_right = cv2.imread(os.path.join('HW2_Dataset', pano, images_in_pano[1]))
    err, merged_img = merge(img_left, img_right)
    if err == 1:
        print("Error: There is not enough matches to create homography matrix. More than 4 good matches are needed.\n It is started to create next panorama...\n\n")
        continue

    for i, img_name in enumerate(images_in_pano):
        if i == 0 or i == 1:
            continue
        elif continue_merging == 'n':
            break
        else:
            img_left = merged_img
            img_right = cv2.imread(os.path.join('HW2_Dataset', pano, images_in_pano[i]))
            err, merged_img = merge(img_left, img_right)
            if err == 1:
                print("\n***Error: There is not enough matches to create homography matrix. More than 4 good matches are needed.\n It is started to create next panorama...\n")
                break
        continue_merging = input("If you don't want to proceed to create panorama due to distortions,\nEnter 'n' -standing for 'next'- and It will be started to create next panorama.\nIf not, press any key")

