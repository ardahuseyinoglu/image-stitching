import numpy as np
import random

def findHomographyMatrix(good_matched_points_in_left, good_matched_points_in_right):
    N=50
    min_error_sum= float("inf")
    final_homography_matrix = 0
    for i in range(N):
        random_point_indexes = random.sample(range(1, len(good_matched_points_in_left)), 4)
        x_left_1 = good_matched_points_in_left[random_point_indexes[0]][0]
        y_left_1 = good_matched_points_in_left[random_point_indexes[0]][1]
        x_right_1 = good_matched_points_in_right[random_point_indexes[0]][0]
        y_right_1 = good_matched_points_in_right[random_point_indexes[0]][1]

        x_left_2 = good_matched_points_in_left[random_point_indexes[1]][0]
        y_left_2 = good_matched_points_in_left[random_point_indexes[1]][1]
        x_right_2 = good_matched_points_in_right[random_point_indexes[1]][0]
        y_right_2 = good_matched_points_in_right[random_point_indexes[1]][1]

        x_left_3 = good_matched_points_in_left[random_point_indexes[2]][0]
        y_left_3 = good_matched_points_in_left[random_point_indexes[2]][1]
        x_right_3 = good_matched_points_in_right[random_point_indexes[2]][0]
        y_right_3 = good_matched_points_in_right[random_point_indexes[2]][1]

        x_left_4 = good_matched_points_in_left[random_point_indexes[3]][0]
        y_left_4 = good_matched_points_in_left[random_point_indexes[3]][1]
        x_right_4 = good_matched_points_in_right[random_point_indexes[3]][0]
        y_right_4 = good_matched_points_in_right[random_point_indexes[3]][1]

        A = np.array([[-x_left_1, -y_left_1, -1, 0, 0, 0, x_left_1*x_right_1, y_left_1*x_right_1, x_right_1],
                      [0, 0, 0, -x_left_1, -y_left_1, -1, x_left_1*y_right_1, y_left_1*y_right_1, y_right_1],
                      [-x_left_2, -y_left_2, -1, 0, 0, 0, x_left_2*x_right_2, y_left_2*x_right_2, x_right_2],
                      [0, 0, 0, -x_left_2, -y_left_2, -1, x_left_2*y_right_2, y_left_2*y_right_2, y_right_2],
                      [-x_left_3, -y_left_3, -1, 0, 0, 0, x_left_3*x_right_3, y_left_3*x_right_3, x_right_3],
                      [0, 0, 0, -x_left_3, -y_left_3, -1, x_left_3*y_right_3, y_left_3*y_right_3, y_right_3],
                      [-x_left_4, -y_left_4, -1, 0, 0, 0, x_left_4*x_right_4, y_left_4*x_right_4, x_right_4],
                      [0, 0, 0, -x_left_4, -y_left_4, -1, x_left_4*y_right_4, y_left_4*y_right_4, y_right_4]])

        e_vals, e_vecs = np.linalg.eig(np.dot(A.T, A))
        homography_vector = e_vecs[:, np.argmin(e_vals)]
        homography_matrix = homography_vector.reshape((3,3))

        good_matched_points_in_left_T = np.transpose(good_matched_points_in_left)
        ones = np.ones((1,good_matched_points_in_left_T.shape[1]))
        good_matched_points_in_left_T_with1s = np.append(good_matched_points_in_left_T, ones, axis=0)
        calculated_right_coordinates = np.matmul(homography_matrix, good_matched_points_in_left_T_with1s)
        calculated_right_coordinates_T = np.transpose(calculated_right_coordinates)
        divider = calculated_right_coordinates_T[:,2]
        calculated_right_coordinates_final = calculated_right_coordinates_T / divider[:,None]

        good_matched_points_in_right_T = np.transpose(good_matched_points_in_right)
        good_matched_points_in_right_T_with1s = np.append(good_matched_points_in_right_T, ones, axis=0)
        right_coordinates_final = np.transpose(good_matched_points_in_right_T_with1s)

        euc_dist = right_coordinates_final - calculated_right_coordinates_final
        euc_dist_square = euc_dist**2
        euc_dist_square_sum = np.sum(euc_dist_square, axis=1)
        euc_dist_square_sum_sqrt = np.sqrt(euc_dist_square_sum)
        total_error_sum = np.sum(euc_dist_square_sum_sqrt)

        if(total_error_sum < min_error_sum):
            min_error_sum = total_error_sum
            final_homography_matrix = homography_matrix

    return final_homography_matrix
