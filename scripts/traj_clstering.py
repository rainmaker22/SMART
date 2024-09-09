from smart.utils.geometry import wrap_angle
import numpy as np


def average_distance_vectorized(point_set1, centroids):
    dists = np.sqrt(np.sum((point_set1[:, None, :, :] - centroids[None, :, :, :])**2, axis=-1))
    return np.mean(dists, axis=2)


def assign_clusters(sub_X, centroids):
    distances = average_distance_vectorized(sub_X, centroids)
    return np.argmin(distances, axis=1)


def Kdisk_cluster(X, N=256, tol=0.035, width=0, length=0, a_pos=None):
    S = []
    ret_traj_list = []
    while len(S) < N:
        num_all = X.shape[0]
        # 随机选择第一个簇中心
        choice_index = np.random.choice(num_all)
        x0 = X[choice_index]
        if x0[0, 0] < -10 or x0[0, 0] > 50 or x0[0, 1] > 10 or x0[0, 1] < -10:
            continue
        res_mask = np.sum((X - x0)**2, axis=(1, 2))/4 > (tol**2)
        del_mask = np.sum((X - x0)**2, axis=(1, 2))/4 <= (tol**2)
        if cal_mean_heading:
            del_contour = X[del_mask]
            diff_xy = del_contour[:, 0, :] - del_contour[:, 3, :]
            del_heading = np.arctan2(diff_xy[:, 1], diff_xy[:, 0]).mean()
            x0 = cal_polygon_contour(x0.mean(0)[0], x0.mean(0)[1], del_heading, width, length)
            del_traj = a_pos[del_mask]
            ret_traj = del_traj.mean(0)[None, ...]
            if abs(ret_traj[0, 1, 0] - ret_traj[0, 0, 0]) > 1 and ret_traj[0, 1, 0] < 0:
                print(ret_traj)
                print('1')
        else:
            x0 = x0[None, ...]
            ret_traj = a_pos[choice_index][None, ...]
        X = X[res_mask]
        a_pos = a_pos[res_mask]
        S.append(x0)
        ret_traj_list.append(ret_traj)
    centroids = np.concatenate(S, axis=0)
    ret_traj = np.concatenate(ret_traj_list, axis=0)

    # closest_dist_sq = np.sum((X - centroids[0])**2, axis=(1, 2))

    # for k in range(1, K):
    #     new_dist_sq = np.sum((X - centroids[k - 1])**2, axis=(1, 2))
    #     closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)
    #     probabilities = closest_dist_sq / np.sum(closest_dist_sq)
    #     centroids[k] = X[np.random.choice(N, p=probabilities)]

    return centroids, ret_traj


def cal_polygon_contour(x, y, theta, width, length):

    left_front_x = x + 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
    left_front_y = y + 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)
    left_front = np.column_stack((left_front_x, left_front_y))

    right_front_x = x + 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
    right_front_y = y + 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)
    right_front = np.column_stack((right_front_x, right_front_y))

    right_back_x = x - 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
    right_back_y = y - 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)
    right_back = np.column_stack((right_back_x, right_back_y))

    left_back_x = x - 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
    left_back_y = y - 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)
    left_back = np.column_stack((left_back_x, left_back_y))

    polygon_contour = np.concatenate((left_front[:, None, :], right_front[:, None, :], right_back[:, None, :], left_back[:, None, :]), axis=1)

    return polygon_contour


if __name__ == '__main__':
    shift = 5 # motion token time dimension
    num_cluster = 6 # vocabulary size
    cal_mean_heading = True
    data = {
        "veh": np.random.rand(1000, 6, 3),
        "cyc": np.random.rand(1000, 6, 3),
        "ped": np.random.rand(1000, 6, 3)
    }
    # Collect the trajectories of all traffic participants from the raw data [NumAgent, shift+1, [relative_x, relative_y, relative_theta]]
    nms_res = {}
    res = {'token': {}, 'traj': {}, 'token_all': {}}
    for k, v in data.items():
        # if k != 'veh':
        #     continue
        a_pos = v
        print(a_pos.shape)
        # a_pos = a_pos[:, shift:1+shift, :]
        cal_num = min(int(1e6), a_pos.shape[0])
        a_pos = a_pos[np.random.choice(a_pos.shape[0], cal_num, replace=False)]
        a_pos[:, :, -1] = wrap_angle(a_pos[:, :, -1])
        print(a_pos.shape)
        if shift <= 2:
            if k == 'veh':
                width = 1.0
                length = 2.4
            elif k == 'cyc':
                width = 0.5
                length = 1.5
            else:
                width = 0.5
                length = 0.5
        else:
            if k == 'veh':
                width = 2.0
                length = 4.8
            elif k == 'cyc':
                width = 1.0
                length = 2.0
            else:
                width = 1.0
                length = 1.0
        contour = cal_polygon_contour(a_pos[:, shift, 0], a_pos[:, shift, 1], a_pos[:, shift, 2], width, length)

        # plt.figure(figsize=(10, 10))
        # for rect in contour:
        #     rect_closed = np.vstack([rect, rect[0]])
        #     plt.plot(rect_closed[:, 0], rect_closed[:, 1], linewidth=0.1)

        # plt.title("Plot of 256 Rectangles")
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.axis('equal')
        # plt.savefig(f'src_{k}_new.jpg', dpi=300)

        if k == 'veh':
            tol = 0.05
        elif k == 'cyc':
            tol = 0.004
        else:
            tol = 0.004
        centroids, ret_traj = Kdisk_cluster(contour, num_cluster, tol, width, length, a_pos[:, :shift+1])
        # plt.figure(figsize=(10, 10))
        contour = cal_polygon_contour(ret_traj[:, :, 0].reshape(num_cluster*(shift+1)),
                                      ret_traj[:, :, 1].reshape(num_cluster*(shift+1)),
                                      ret_traj[:, :, 2].reshape(num_cluster*(shift+1)), width, length)

        res['token_all'][k] = contour.reshape(num_cluster, (shift+1), 4, 2)
        res['token'][k] = centroids
        res['traj'][k] = ret_traj
