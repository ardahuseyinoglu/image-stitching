import numpy as np

def warpAndMerge(img_left, img_right, homography_matrix):
    dst = np.ndarray(shape=(img_left.shape[0], img_left.shape[1] + img_right.shape[1],  3), dtype=img_left.dtype)
    dst[0:img_left.shape[0], 0:img_left.shape[1]] = img_left

    for col in range(img_right.shape[1]):
        for row in range(img_right.shape[0]):
            right_img_coordinates = np.array([col, row, 1])
            right_img_coordinates = right_img_coordinates[:,None]
            right_img_coordinates_prime = np.matmul(homography_matrix, right_img_coordinates)
            right_img_coordinates_prime = right_img_coordinates_prime / right_img_coordinates_prime[2]
            right_img_coordinates_prime = np.round(right_img_coordinates_prime).astype(int)
            if(right_img_coordinates_prime[0][0] > 0 and right_img_coordinates_prime[0][0] < dst.shape[1] and right_img_coordinates_prime[1][0] > 0 and right_img_coordinates_prime[1][0] < dst.shape[0]):
                dst[right_img_coordinates_prime[1][0]][right_img_coordinates_prime[0][0]] = img_right[row][col]


    black_cols = np.where(~dst.any(axis=0))[0]
    if(len(black_cols) != 0):
        start_col_no_black = black_cols[0]
        dst = dst[:,:start_col_no_black]

    return dst
