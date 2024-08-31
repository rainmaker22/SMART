import torch
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
import math
import pickle
from smart.utils import wrap_angle
import os

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

    polygon_contour = np.concatenate(
        (left_front[:, None, :], right_front[:, None, :], right_back[:, None, :], left_back[:, None, :]), axis=1)

    return polygon_contour


def interplating_polyline(polylines, heading, distance=0.5, split_distace=5):
    # Calculate the cumulative distance along the path, up-sample the polyline to 0.5 meter
    dist_along_path_list = [[0]]
    polylines_list = [[polylines[0]]]
    for i in range(1, polylines.shape[0]):
        euclidean_dist = euclidean(polylines[i, :2], polylines[i - 1, :2])
        heading_diff = min(abs(max(heading[i], heading[i - 1]) - min(heading[1], heading[i - 1])),
                           abs(max(heading[i], heading[i - 1]) - min(heading[1], heading[i - 1]) + math.pi))
        if heading_diff > math.pi / 4 and euclidean_dist > 3:
            dist_along_path_list.append([0])
            polylines_list.append([polylines[i]])
        elif heading_diff > math.pi / 8 and euclidean_dist > 3:
            dist_along_path_list.append([0])
            polylines_list.append([polylines[i]])
        elif heading_diff > 0.1 and euclidean_dist > 3:
            dist_along_path_list.append([0])
            polylines_list.append([polylines[i]])
        elif euclidean_dist > 10:
            dist_along_path_list.append([0])
            polylines_list.append([polylines[i]])
        else:
            dist_along_path_list[-1].append(dist_along_path_list[-1][-1] + euclidean_dist)
            polylines_list[-1].append(polylines[i])
    # plt.plot(polylines[:, 0], polylines[:, 1])
    # plt.savefig('tmp.jpg')
    new_x_list = []
    new_y_list = []
    multi_polylines_list = []
    for idx in range(len(dist_along_path_list)):
        if len(dist_along_path_list[idx]) < 2:
            continue
        dist_along_path = np.array(dist_along_path_list[idx])
        polylines_cur = np.array(polylines_list[idx])
        # Create interpolation functions for x and y coordinates
        fx = interp1d(dist_along_path, polylines_cur[:, 0])
        fy = interp1d(dist_along_path, polylines_cur[:, 1])
        # fyaw = interp1d(dist_along_path, heading)

        # Create an array of distances at which to interpolate
        new_dist_along_path = np.arange(0, dist_along_path[-1], distance)
        new_dist_along_path = np.concatenate([new_dist_along_path, dist_along_path[[-1]]])
        # Use the interpolation functions to generate new x and y coordinates
        new_x = fx(new_dist_along_path)
        new_y = fy(new_dist_along_path)
        # new_yaw = fyaw(new_dist_along_path)
        new_x_list.append(new_x)
        new_y_list.append(new_y)

        # Combine the new x and y coordinates into a single array
        new_polylines = np.vstack((new_x, new_y)).T
        polyline_size = int(split_distace / distance)
        if new_polylines.shape[0] >= (polyline_size + 1):
            padding_size = (new_polylines.shape[0] - (polyline_size + 1)) % polyline_size
            final_index = (new_polylines.shape[0] - (polyline_size + 1)) // polyline_size + 1
        else:
            padding_size = new_polylines.shape[0]
            final_index = 0
        multi_polylines = None
        new_polylines = torch.from_numpy(new_polylines)
        new_heading = torch.atan2(new_polylines[1:, 1] - new_polylines[:-1, 1],
                                  new_polylines[1:, 0] - new_polylines[:-1, 0])
        new_heading = torch.cat([new_heading, new_heading[-1:]], -1)[..., None]
        new_polylines = torch.cat([new_polylines, new_heading], -1)
        if new_polylines.shape[0] >= (polyline_size + 1):
            multi_polylines = new_polylines.unfold(dimension=0, size=polyline_size + 1, step=polyline_size)
            multi_polylines = multi_polylines.transpose(1, 2)
            multi_polylines = multi_polylines[:, ::5, :]
        if padding_size >= 3:
            last_polyline = new_polylines[final_index * polyline_size:]
            last_polyline = last_polyline[torch.linspace(0, last_polyline.shape[0] - 1, steps=3).long()]
            if multi_polylines is not None:
                multi_polylines = torch.cat([multi_polylines, last_polyline.unsqueeze(0)], dim=0)
            else:
                multi_polylines = last_polyline.unsqueeze(0)
        if multi_polylines is None:
            continue
        multi_polylines_list.append(multi_polylines)
    if len(multi_polylines_list) > 0:
        multi_polylines_list = torch.cat(multi_polylines_list, dim=0)
    else:
        multi_polylines_list = None
    return multi_polylines_list


def average_distance_vectorized(point_set1, centroids):
    dists = np.sqrt(np.sum((point_set1[:, None, :, :] - centroids[None, :, :, :]) ** 2, axis=-1))
    return np.mean(dists, axis=2)


def assign_clusters(sub_X, centroids):
    distances = average_distance_vectorized(sub_X, centroids)
    return np.argmin(distances, axis=1)


class TokenProcessor:

    def __init__(self, token_size):
        module_dir = os.path.dirname(os.path.dirname(__file__))
        self.agent_token_path = os.path.join(module_dir, f'tokens/cluster_frame_5_{token_size}.pkl')
        self.map_token_traj_path = os.path.join(module_dir, 'tokens/map_traj_token5.pkl')
        self.noise = False
        self.disturb = False
        self.shift = 5
        self.get_trajectory_token()
        self.training = False
        self.current_step = 10

    def preprocess(self, data):
        data = self.tokenize_agent(data)
        data = self.tokenize_map(data)
        del data['city']
        if 'polygon_is_intersection' in data['map_polygon']:
            del data['map_polygon']['polygon_is_intersection']
        if 'route_type' in data['map_polygon']:
            del data['map_polygon']['route_type']
        return data

    def get_trajectory_token(self):
        agent_token_data = pickle.load(open(self.agent_token_path, 'rb'))
        map_token_traj = pickle.load(open(self.map_token_traj_path, 'rb'))
        self.trajectory_token = agent_token_data['token']
        self.trajectory_token_all = agent_token_data['token_all']
        self.map_token = {'traj_src': map_token_traj['traj_src'], }
        self.token_last = {}
        for k, v in self.trajectory_token_all.items():
            token_last = torch.from_numpy(v[:, -2:]).to(torch.float)
            diff_xy = token_last[:, 0, 0] - token_last[:, 0, 3]
            theta = torch.arctan2(diff_xy[:, 1], diff_xy[:, 0])
            cos, sin = theta.cos(), theta.sin()
            rot_mat = theta.new_zeros(token_last.shape[0], 2, 2)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = -sin
            rot_mat[:, 1, 0] = sin
            rot_mat[:, 1, 1] = cos
            agent_token = torch.bmm(token_last[:, 1], rot_mat)
            agent_token -= token_last[:, 0].mean(1)[:, None, :]
            self.token_last[k] = agent_token.numpy()

    def clean_heading(self, data):
        heading = data['agent']['heading']
        valid = data['agent']['valid_mask']
        pi = torch.tensor(torch.pi)
        n_vehicles, n_frames = heading.shape

        heading_diff_raw = heading[:, :-1] - heading[:, 1:]
        heading_diff = torch.remainder(heading_diff_raw + pi, 2 * pi) - pi
        heading_diff[heading_diff > pi] -= 2 * pi
        heading_diff[heading_diff < -pi] += 2 * pi

        valid_pairs = valid[:, :-1] & valid[:, 1:]

        for i in range(n_frames - 1):
            change_needed = (torch.abs(heading_diff[:, i:i + 1]) > 1.0) & valid_pairs[:, i:i + 1]

            heading[:, i + 1][change_needed.squeeze()] = heading[:, i][change_needed.squeeze()]

            if i < n_frames - 2:
                heading_diff_raw = heading[:, i + 1] - heading[:, i + 2]
                heading_diff[:, i + 1] = torch.remainder(heading_diff_raw + pi, 2 * pi) - pi
                heading_diff[heading_diff[:, i + 1] > pi] -= 2 * pi
                heading_diff[heading_diff[:, i + 1] < -pi] += 2 * pi

    def tokenize_agent(self, data):
        if data['agent']["velocity"].shape[1] == 90:
            print(data['scenario_id'], data['agent']["velocity"].shape)
        interplote_mask = (data['agent']['valid_mask'][:, self.current_step] == False) * (
                data['agent']['position'][:, self.current_step, 0] != 0)
        if data['agent']["velocity"].shape[-1] == 2:
            data['agent']["velocity"] = torch.cat([data['agent']["velocity"],
                                                   torch.zeros(data['agent']["velocity"].shape[0],
                                                               data['agent']["velocity"].shape[1], 1)], dim=-1)
        vel = data['agent']["velocity"][interplote_mask, self.current_step]
        data['agent']['position'][interplote_mask, self.current_step - 1, :3] = data['agent']['position'][
                                                                                interplote_mask, self.current_step,
                                                                                :3] - vel * 0.1
        data['agent']['valid_mask'][interplote_mask, self.current_step - 1:self.current_step + 1] = True
        data['agent']['heading'][interplote_mask, self.current_step - 1] = data['agent']['heading'][
            interplote_mask, self.current_step]
        data['agent']["velocity"][interplote_mask, self.current_step - 1] = data['agent']["velocity"][
            interplote_mask, self.current_step]

        data['agent']['type'] = data['agent']['type'].to(torch.uint8)

        self.clean_heading(data)
        matching_extra_mask = (data['agent']['valid_mask'][:, self.current_step] == True) * (
                data['agent']['valid_mask'][:, self.current_step - 5] == False)

        interplote_mask_first = (data['agent']['valid_mask'][:, 0] == False) * (data['agent']['position'][:, 0, 0] != 0)
        data['agent']['valid_mask'][interplote_mask_first, 0] = True

        agent_pos = data['agent']['position'][:, :, :2]
        valid_mask = data['agent']['valid_mask']

        valid_mask_shift = valid_mask.unfold(1, self.shift + 1, self.shift)
        token_valid_mask = valid_mask_shift[:, :, 0] * valid_mask_shift[:, :, -1]
        agent_type = data['agent']['type']
        agent_category = data['agent']['category']
        agent_heading = data['agent']['heading']
        vehicle_mask = agent_type == 0
        cyclist_mask = agent_type == 2
        ped_mask = agent_type == 1

        veh_pos = agent_pos[vehicle_mask, :, :]
        veh_valid_mask = valid_mask[vehicle_mask, :]
        cyc_pos = agent_pos[cyclist_mask, :, :]
        cyc_valid_mask = valid_mask[cyclist_mask, :]
        ped_pos = agent_pos[ped_mask, :, :]
        ped_valid_mask = valid_mask[ped_mask, :]

        veh_token_index, veh_token_contour = self.match_token(veh_pos, veh_valid_mask, agent_heading[vehicle_mask],
                                                              'veh', agent_category[vehicle_mask],
                                                              matching_extra_mask[vehicle_mask])
        ped_token_index, ped_token_contour = self.match_token(ped_pos, ped_valid_mask, agent_heading[ped_mask], 'ped',
                                                              agent_category[ped_mask], matching_extra_mask[ped_mask])
        cyc_token_index, cyc_token_contour = self.match_token(cyc_pos, cyc_valid_mask, agent_heading[cyclist_mask],
                                                              'cyc', agent_category[cyclist_mask],
                                                              matching_extra_mask[cyclist_mask])

        token_index = torch.zeros((agent_pos.shape[0], veh_token_index.shape[1])).to(torch.int64)
        token_index[vehicle_mask] = veh_token_index
        token_index[ped_mask] = ped_token_index
        token_index[cyclist_mask] = cyc_token_index

        token_contour = torch.zeros((agent_pos.shape[0], veh_token_contour.shape[1],
                                     veh_token_contour.shape[2], veh_token_contour.shape[3]))
        token_contour[vehicle_mask] = veh_token_contour
        token_contour[ped_mask] = ped_token_contour
        token_contour[cyclist_mask] = cyc_token_contour

        trajectory_token_veh = torch.from_numpy(self.trajectory_token['veh']).clone().to(torch.float)
        trajectory_token_ped = torch.from_numpy(self.trajectory_token['ped']).clone().to(torch.float)
        trajectory_token_cyc = torch.from_numpy(self.trajectory_token['cyc']).clone().to(torch.float)

        agent_token_traj = torch.zeros((agent_pos.shape[0], trajectory_token_veh.shape[0], 4, 2))
        agent_token_traj[vehicle_mask] = trajectory_token_veh
        agent_token_traj[ped_mask] = trajectory_token_ped
        agent_token_traj[cyclist_mask] = trajectory_token_cyc

        if not self.training:
            token_valid_mask[matching_extra_mask, 1] = True

        data['agent']['token_idx'] = token_index
        data['agent']['token_contour'] = token_contour
        token_pos = token_contour.mean(dim=2)
        data['agent']['token_pos'] = token_pos
        diff_xy = token_contour[:, :, 0, :] - token_contour[:, :, 3, :]
        data['agent']['token_heading'] = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])
        data['agent']['agent_valid_mask'] = token_valid_mask

        vel = torch.cat([token_pos.new_zeros(data['agent']['num_nodes'], 1, 2),
                         ((token_pos[:, 1:] - token_pos[:, :-1]) / (0.1 * self.shift))], dim=1)
        vel_valid_mask = torch.cat([torch.zeros(token_valid_mask.shape[0], 1, dtype=torch.bool),
                                    (token_valid_mask * token_valid_mask.roll(shifts=1, dims=1))[:, 1:]], dim=1)
        vel[~vel_valid_mask] = 0
        vel[data['agent']['valid_mask'][:, self.current_step], 1] = data['agent']['velocity'][
                                                                    data['agent']['valid_mask'][:, self.current_step],
                                                                    self.current_step, :2]

        data['agent']['token_velocity'] = vel

        return data

    def match_token(self, pos, valid_mask, heading, category, agent_category, extra_mask):
        agent_token_src = self.trajectory_token[category]
        token_last = self.token_last[category]
        if self.shift <= 2:
            if category == 'veh':
                width = 1.0
                length = 2.4
            elif category == 'cyc':
                width = 0.5
                length = 1.5
            else:
                width = 0.5
                length = 0.5
        else:
            if category == 'veh':
                width = 2.0
                length = 4.8
            elif category == 'cyc':
                width = 1.0
                length = 2.0
            else:
                width = 1.0
                length = 1.0

        prev_heading = heading[:, 0]
        prev_pos = pos[:, 0]
        agent_num, num_step, feat_dim = pos.shape
        token_num, token_contour_dim, feat_dim = agent_token_src.shape
        agent_token_src = agent_token_src.reshape(1, token_num * token_contour_dim, feat_dim).repeat(agent_num, 0)
        token_last = token_last.reshape(1, token_num * token_contour_dim, feat_dim).repeat(extra_mask.sum(), 0)
        token_index_list = []
        token_contour_list = []
        prev_token_idx = None

        for i in range(self.shift, pos.shape[1], self.shift):
            theta = prev_heading
            cur_heading = heading[:, i]
            cur_pos = pos[:, i]
            cos, sin = theta.cos(), theta.sin()
            rot_mat = theta.new_zeros(agent_num, 2, 2)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos
            agent_token_world = torch.bmm(torch.from_numpy(agent_token_src).to(torch.float), rot_mat).reshape(agent_num,
                                                                                                              token_num,
                                                                                                              token_contour_dim,
                                                                                                              feat_dim)
            agent_token_world += prev_pos[:, None, None, :]

            cur_contour = cal_polygon_contour(cur_pos[:, 0], cur_pos[:, 1], cur_heading, width, length)
            agent_token_index = torch.from_numpy(np.argmin(
                np.mean(np.sqrt(np.sum((cur_contour[:, None, ...] - agent_token_world.numpy()) ** 2, axis=-1)), axis=2),
                axis=-1))
            if prev_token_idx is not None and self.noise:
                same_idx = prev_token_idx == agent_token_index
                same_idx[:] = True
                topk_indices = np.argsort(
                    np.mean(np.sqrt(np.sum((cur_contour[:, None, ...] - agent_token_world.numpy()) ** 2, axis=-1)),
                            axis=2), axis=-1)[:, :5]
                sample_topk = np.random.choice(range(0, topk_indices.shape[1]), topk_indices.shape[0])
                agent_token_index[same_idx] = \
                    torch.from_numpy(topk_indices[np.arange(topk_indices.shape[0]), sample_topk])[same_idx]

            token_contour_select = agent_token_world[torch.arange(agent_num), agent_token_index]

            diff_xy = token_contour_select[:, 0, :] - token_contour_select[:, 3, :]

            prev_heading = heading[:, i].clone()
            prev_heading[valid_mask[:, i - self.shift]] = torch.arctan2(diff_xy[:, 1], diff_xy[:, 0])[
                valid_mask[:, i - self.shift]]

            prev_pos = pos[:, i].clone()
            prev_pos[valid_mask[:, i - self.shift]] = token_contour_select.mean(dim=1)[valid_mask[:, i - self.shift]]
            prev_token_idx = agent_token_index
            token_index_list.append(agent_token_index[:, None])
            token_contour_list.append(token_contour_select[:, None, ...])

        token_index = torch.cat(token_index_list, dim=1)
        token_contour = torch.cat(token_contour_list, dim=1)

        # extra matching
        if not self.training:
            theta = heading[extra_mask, self.current_step - 1]
            prev_pos = pos[extra_mask, self.current_step - 1]
            cur_pos = pos[extra_mask, self.current_step]
            cur_heading = heading[extra_mask, self.current_step]
            cos, sin = theta.cos(), theta.sin()
            rot_mat = theta.new_zeros(extra_mask.sum(), 2, 2)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos
            agent_token_world = torch.bmm(torch.from_numpy(token_last).to(torch.float), rot_mat).reshape(
                extra_mask.sum(), token_num, token_contour_dim, feat_dim)
            agent_token_world += prev_pos[:, None, None, :]

            cur_contour = cal_polygon_contour(cur_pos[:, 0], cur_pos[:, 1], cur_heading, width, length)
            agent_token_index = torch.from_numpy(np.argmin(
                np.mean(np.sqrt(np.sum((cur_contour[:, None, ...] - agent_token_world.numpy()) ** 2, axis=-1)), axis=2),
                axis=-1))
            token_contour_select = agent_token_world[torch.arange(extra_mask.sum()), agent_token_index]

            token_index[extra_mask, 1] = agent_token_index
            token_contour[extra_mask, 1] = token_contour_select

        return token_index, token_contour

    def tokenize_map(self, data):
        data['map_polygon']['type'] = data['map_polygon']['type'].to(torch.uint8)
        data['map_point']['type'] = data['map_point']['type'].to(torch.uint8)
        pt2pl = data[('map_point', 'to', 'map_polygon')]['edge_index']
        pt_type = data['map_point']['type'].to(torch.uint8)
        pt_side = torch.zeros_like(pt_type)
        pt_pos = data['map_point']['position'][:, :2]
        data['map_point']['orientation'] = wrap_angle(data['map_point']['orientation'])
        pt_heading = data['map_point']['orientation']
        split_polyline_type = []
        split_polyline_pos = []
        split_polyline_theta = []
        split_polyline_side = []
        pl_idx_list = []
        split_polygon_type = []
        data['map_point']['type'].unique()

        for i in sorted(np.unique(pt2pl[1])):
            index = pt2pl[0, pt2pl[1] == i]
            polygon_type = data['map_polygon']["type"][i]
            cur_side = pt_side[index]
            cur_type = pt_type[index]
            cur_pos = pt_pos[index]
            cur_heading = pt_heading[index]

            for side_val in np.unique(cur_side):
                for type_val in np.unique(cur_type):
                    if type_val == 13:
                        continue
                    indices = np.where((cur_side == side_val) & (cur_type == type_val))[0]
                    if len(indices) <= 2:
                        continue
                    split_polyline = interplating_polyline(cur_pos[indices].numpy(), cur_heading[indices].numpy())
                    if split_polyline is None:
                        continue
                    new_cur_type = cur_type[indices][0]
                    new_cur_side = cur_side[indices][0]
                    map_polygon_type = polygon_type.repeat(split_polyline.shape[0])
                    new_cur_type = new_cur_type.repeat(split_polyline.shape[0])
                    new_cur_side = new_cur_side.repeat(split_polyline.shape[0])
                    cur_pl_idx = torch.Tensor([i])
                    new_cur_pl_idx = cur_pl_idx.repeat(split_polyline.shape[0])
                    split_polyline_pos.append(split_polyline[..., :2])
                    split_polyline_theta.append(split_polyline[..., 2])
                    split_polyline_type.append(new_cur_type)
                    split_polyline_side.append(new_cur_side)
                    pl_idx_list.append(new_cur_pl_idx)
                    split_polygon_type.append(map_polygon_type)

        split_polyline_pos = torch.cat(split_polyline_pos, dim=0)
        split_polyline_theta = torch.cat(split_polyline_theta, dim=0)
        split_polyline_type = torch.cat(split_polyline_type, dim=0)
        split_polyline_side = torch.cat(split_polyline_side, dim=0)
        split_polygon_type = torch.cat(split_polygon_type, dim=0)
        pl_idx_list = torch.cat(pl_idx_list, dim=0)
        vec = split_polyline_pos[:, 1, :] - split_polyline_pos[:, 0, :]
        data['map_save'] = {}
        data['pt_token'] = {}
        data['map_save']['traj_pos'] = split_polyline_pos
        data['map_save']['traj_theta'] = split_polyline_theta[:, 0]  # torch.arctan2(vec[:, 1], vec[:, 0])
        data['map_save']['pl_idx_list'] = pl_idx_list
        data['pt_token']['type'] = split_polyline_type
        data['pt_token']['side'] = split_polyline_side
        data['pt_token']['pl_type'] = split_polygon_type
        data['pt_token']['num_nodes'] = split_polyline_pos.shape[0]
        return data