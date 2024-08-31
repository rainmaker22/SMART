import numpy as np
import pandas as pd
import os
import torch
from typing import Any, Dict, List, Optional

predict_unseen_agents = False
vector_repr = True
_agent_types = ['vehicle', 'pedestrian', 'cyclist', 'background']
_polygon_types = ['VEHICLE', 'BIKE', 'BUS', 'PEDESTRIAN']
_polygon_light_type = ['LANE_STATE_STOP', 'LANE_STATE_GO', 'LANE_STATE_CAUTION', 'LANE_STATE_UNKNOWN']
_point_types = ['DASH_SOLID_YELLOW', 'DASH_SOLID_WHITE', 'DASHED_WHITE', 'DASHED_YELLOW',
                'DOUBLE_SOLID_YELLOW', 'DOUBLE_SOLID_WHITE', 'DOUBLE_DASH_YELLOW', 'DOUBLE_DASH_WHITE',
                'SOLID_YELLOW', 'SOLID_WHITE', 'SOLID_DASH_WHITE', 'SOLID_DASH_YELLOW', 'EDGE',
                'NONE', 'UNKNOWN', 'CROSSWALK', 'CENTERLINE']
_point_sides = ['LEFT', 'RIGHT', 'CENTER']
_polygon_to_polygon_types = ['NONE', 'PRED', 'SUCC', 'LEFT', 'RIGHT']
_polygon_is_intersections = [True, False, None]


Lane_type_hash = {
    4: "BIKE",
    3: "VEHICLE",
    2: "VEHICLE",
    1: "BUS"
}

boundary_type_hash = {
        5: "UNKNOWN",
        6: "DASHED_WHITE",
        7: "SOLID_WHITE",
        8: "DOUBLE_DASH_WHITE",
        9: "DASHED_YELLOW",
        10: "DOUBLE_DASH_YELLOW",
        11: "SOLID_YELLOW",
        12: "DOUBLE_SOLID_YELLOW",
        13: "DASH_SOLID_YELLOW",
        14: "UNKNOWN",
        15: "EDGE",
        16: "EDGE"
}


def get_agent_features(df: pd.DataFrame, av_id, num_historical_steps=10, dim=3, num_steps=91) -> Dict[str, Any]:
    if not predict_unseen_agents:  # filter out agents that are unseen during the historical time steps
        historical_df = df[df['timestep'] == num_historical_steps-1]
        agent_ids = list(historical_df['track_id'].unique())
        df = df[df['track_id'].isin(agent_ids)]
    else:
        agent_ids = list(df['track_id'].unique())

    num_agents = len(agent_ids)
    # initialization
    valid_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool)
    current_valid_mask = torch.zeros(num_agents, dtype=torch.bool)
    predict_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool)
    agent_id: List[Optional[str]] = [None] * num_agents
    agent_type = torch.zeros(num_agents, dtype=torch.uint8)
    agent_category = torch.zeros(num_agents, dtype=torch.uint8)
    position = torch.zeros(num_agents, num_steps, dim, dtype=torch.float)
    heading = torch.zeros(num_agents, num_steps, dtype=torch.float)
    velocity = torch.zeros(num_agents, num_steps, dim, dtype=torch.float)
    shape = torch.zeros(num_agents, num_steps, dim, dtype=torch.float)

    for track_id, track_df in df.groupby('track_id'):
        agent_idx = agent_ids.index(track_id)
        agent_steps = track_df['timestep'].values

        valid_mask[agent_idx, agent_steps] = True
        current_valid_mask[agent_idx] = valid_mask[agent_idx, num_historical_steps - 1]
        predict_mask[agent_idx, agent_steps] = True
        if vector_repr:  # a time step t is valid only when both t and t-1 are valid
            valid_mask[agent_idx, 1: num_historical_steps] = (
                valid_mask[agent_idx, :num_historical_steps - 1] &
                valid_mask[agent_idx, 1: num_historical_steps])
            valid_mask[agent_idx, 0] = False
        predict_mask[agent_idx, :num_historical_steps] = False
        if not current_valid_mask[agent_idx]:
            predict_mask[agent_idx, num_historical_steps:] = False

        agent_id[agent_idx] = track_id
        agent_type[agent_idx] = _agent_types.index(track_df['object_type'].values[0])
        agent_category[agent_idx] = track_df['object_category'].values[0]
        position[agent_idx, agent_steps, :3] = torch.from_numpy(np.stack([track_df['position_x'].values,
                                                                          track_df['position_y'].values,
                                                                          track_df['position_z'].values],
                                                                         axis=-1)).float()
        heading[agent_idx, agent_steps] = torch.from_numpy(track_df['heading'].values).float()
        velocity[agent_idx, agent_steps, :2] = torch.from_numpy(np.stack([track_df['velocity_x'].values,
                                                                          track_df['velocity_y'].values],
                                                                         axis=-1)).float()
        shape[agent_idx, agent_steps, :3] = torch.from_numpy(np.stack([track_df['length'].values,
                                                                       track_df['width'].values,
                                                                       track_df["height"].values],
                                                                      axis=-1)).float()
    av_idx = agent_id.index(av_id)

    return {
        'num_nodes': num_agents,
        'av_index': av_idx,
        'valid_mask': valid_mask,
        'predict_mask': predict_mask,
        'id': agent_id,
        'type': agent_type,
        'category': agent_category,
        'position': position,
        'heading': heading,
        'velocity': velocity,
        'shape': shape
    }