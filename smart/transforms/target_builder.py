
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
from smart.utils import wrap_angle
from smart.utils.log import Logging


def to_16(data):
    if isinstance(data, dict):
        for key, value in data.items():
            new_value = to_16(value)
            data[key] = new_value
    if isinstance(data, torch.Tensor):
        if data.dtype == torch.float32:
            data = data.to(torch.float16)
    return data


def tofloat32(data):
    for name in data:
        value = data[name]
        if isinstance(value, dict):
            value = tofloat32(value)
        elif isinstance(value, torch.Tensor) and value.dtype == torch.float64:
            value = value.to(torch.float32)
        data[name] = value
    return data


class WaymoTargetBuilder(BaseTransform):

    def __init__(self,
                 num_historical_steps: int,
                 num_future_steps: int,
                 mode="train") -> None:
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.mode = mode
        self.num_features = 3
        self.augment = False
        self.logger = Logging().log(level='DEBUG')

    def score_ego_agent(self, agent):
        av_index = agent['av_index']
        agent["category"][av_index] = 5
        return agent

    def clip(self, agent, max_num=32):
        av_index = agent["av_index"]
        valid = agent['valid_mask']
        ego_pos = agent["position"][av_index]
        obstacle_mask = agent['type'] == 3
        distance = torch.norm(agent["position"][:, self.num_historical_steps-1, :2] - ego_pos[self.num_historical_steps-1, :2], dim=-1)  # keep the closest 100 vehicles near the ego car
        distance[obstacle_mask] = 10e5
        sort_idx = distance.sort()[1]
        mask = torch.zeros(valid.shape[0])
        mask[sort_idx[:max_num]] = 1
        mask = mask.to(torch.bool)
        mask[av_index] = True
        new_av_index = mask[:av_index].sum()
        agent["num_nodes"] = int(mask.sum())
        agent["av_index"] = int(new_av_index)
        excluded = ["num_nodes", "av_index", "ego"]
        for key, val in agent.items():
            if key in excluded:
                continue
            if key == "id":
                val = list(np.array(val)[mask])
                agent[key] = val
                continue
            if len(val.size()) > 1:
                agent[key] = val[mask, ...]
            else:
                agent[key] = val[mask]
        return agent

    def score_nearby_vehicle(self, agent, max_num=10):
        av_index = agent['av_index']
        agent["category"] = torch.zeros_like(agent["category"])
        obstacle_mask = agent['type'] == 3
        pos = agent["position"][av_index, self.num_historical_steps, :2]
        distance = torch.norm(agent["position"][:, self.num_historical_steps, :2] - pos, dim=-1)
        distance[obstacle_mask] = 10e5
        sort_idx = distance.sort()[1]
        nearby_mask = torch.zeros(distance.shape[0])
        nearby_mask[sort_idx[1:max_num]] = 1
        nearby_mask = nearby_mask.bool()
        agent["category"][nearby_mask] = 3
        agent["category"][obstacle_mask] = 0

    def score_trained_vehicle(self, agent, max_num=10, min_distance=0):
        av_index = agent['av_index']
        agent["category"] = torch.zeros_like(agent["category"])
        pos = agent["position"][av_index, self.num_historical_steps, :2]
        distance = torch.norm(agent["position"][:, self.num_historical_steps, :2] - pos, dim=-1)
        distance_all_time = torch.norm(agent["position"][:, :, :2] - agent["position"][av_index, :, :2], dim=-1)
        invalid_mask = distance_all_time < 150  # we do not believe the perception out of range of 150 meters
        agent["valid_mask"] = agent["valid_mask"] * invalid_mask
        # we do not predict vehicle  too far away from ego car
        closet_vehicle = distance < 100
        valid = agent['valid_mask']
        valid_current = valid[:, (self.num_historical_steps):]
        valid_counts = valid_current.sum(1)
        counts_vehicle = valid_counts >= 1
        no_backgroud = agent['type'] != 3
        vehicle2pred = closet_vehicle & counts_vehicle & no_backgroud
        if vehicle2pred.sum() > max_num:
            # too many still vehicle so that train the model using the moving vehicle as much as possible
            true_indices = torch.nonzero(vehicle2pred).squeeze(1)
            selected_indices = true_indices[torch.randperm(true_indices.size(0))[:max_num]]
            vehicle2pred.fill_(False)
            vehicle2pred[selected_indices] = True
        agent["category"][vehicle2pred] = 3

    def rotate_agents(self, position, heading, num_nodes, num_historical_steps, num_future_steps):
        origin = position[:, num_historical_steps - 1]
        theta = heading[:, num_historical_steps - 1]
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(num_nodes, 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        target = origin.new_zeros(num_nodes, num_future_steps, 4)
        target[..., :2] = torch.bmm(position[:, num_historical_steps:, :2] -
                                    origin[:, :2].unsqueeze(1), rot_mat)
        his = origin.new_zeros(num_nodes, num_historical_steps, 4)
        his[..., :2] = torch.bmm(position[:, :num_historical_steps, :2] -
                                 origin[:, :2].unsqueeze(1), rot_mat)
        if position.size(2) == 3:
            target[..., 2] = (position[:, num_historical_steps:, 2] -
                              origin[:, 2].unsqueeze(-1))
            his[..., 2] = (position[:, :num_historical_steps, 2] -
                           origin[:, 2].unsqueeze(-1))
            target[..., 3] = wrap_angle(heading[:, num_historical_steps:] -
                                        theta.unsqueeze(-1))
            his[..., 3] = wrap_angle(heading[:, :num_historical_steps] -
                                     theta.unsqueeze(-1))
        else:
            target[..., 2] = wrap_angle(heading[:, num_historical_steps:] -
                                        theta.unsqueeze(-1))
            his[..., 2] = wrap_angle(heading[:, :num_historical_steps] -
                                     theta.unsqueeze(-1))
        return his, target

    def __call__(self, data) -> HeteroData:
        agent = data["agent"]
        self.score_ego_agent(agent)
        self.score_trained_vehicle(agent, max_num=32)
        return HeteroData(data)
