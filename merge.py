import cv2
import numpy as np
from matplotlib import pyplot as plt
from homography import findHomographyMatrix
from merging import warpAndMerge

def merge(img_left, img_right):
    #CONVERT IMAGES TO GRAY SCALE
    img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    #FEATURE EXTRACTION
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img_left_gray, None)
    kp2, des2 = orb.detectAndCompute(img_right_gray, None)

    #PLOT KEY POINTS FOR LEFT AND RIGHT IMAGE
    feature_extraction_left_out = cv2.drawKeypoints(img_left, kp1, outImage=None, color=(0,255,0), flags=0)
    plt.imshow(cv2.cvtColor(feature_extraction_left_out, cv2.COLOR_BGR2RGB))
    plt.show()
    feature_extraction_right_out = cv2.drawKeypoints(img_right, kp2, outImage=None,color=(0,255,0), flags=0)
    plt.imshow(cv2.cvtColor(feature_extraction_right_out, cv2.COLOR_BGR2RGB))
    plt.show()

    #FEATURE MATCHING
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    #PLOT MATCHES
    feature_matching_out = cv2.drawMatches(img_left,kp1,img_right,kp2,matches[:50],None, flags=2)
    plt.imshow(cv2.cvtColor(feature_matching_out, cv2.COLOR_BGR2RGB))
    plt.show()

    #GET ONLY GOOD ONES
    good_matches = matches[:50]

    #GET COORDINATES OF GOOD MATCHED POINTS ON LEFT AND RIGHT IMAGE
    good_matched_points_in_left = np.zeros((len(good_matches), 2))
    good_matched_points_in_right = np.zeros((len(good_matches), 2))
    for i, match in enumerate(good_matches):
        good_matched_points_in_left[i, :] = kp1[match.queryIdx].pt
        good_matched_points_in_right[i, :] = kp2[match.trainIdx].pt


    #FIND HOMOGRAPHY MATRIX
    if(len(good_matches) > 4):
        homography_matrix = findHomographyMatrix(good_matched_points_in_left, good_matched_points_in_right)
        h_inv = np.linalg.inv(homography_matrix)

        #MERGE TWO IMAGES
        merged_img = warpAndMerge(img_left, img_right, h_inv)

        #PLOT MERGED IMAGE
        plt.imshow(cv2.cvtColor(merged_img, cv2.COLOR_BGR2RGB))
        plt.show()

        return 0, merged_img
    else:
        return 1, img_left
