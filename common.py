import json
import os
import pickle
import re
from collections import defaultdict

import networkx as nx
import numpy as np
from tqdm import tqdm


def coo_dist(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return np.sqrt(dx ** 2 + dy ** 2)


def construct_road_network_urbanvehicle(road_node_xy_dict, edge_dict, road_graph_pkl_file):
    if not os.path.exists(road_graph_pkl_file):
        road_graph = nx.MultiDiGraph()

        for node_id, loc_xy in road_node_xy_dict.items():
            x = loc_xy[0]
            y = loc_xy[1]
            road_graph.add_node(node_id, id=node_id, x=x, y=y)

        for edge_id, node_info in edge_dict.items():
            pre_node_id = node_info[0]
            succ_node_id = node_info[1]
            pre_node_x, pre_node_y = road_node_xy_dict[pre_node_id]
            succ_node_x, succ_node_y = road_node_xy_dict[succ_node_id]
            node_dist = coo_dist(pre_node_x, pre_node_y, succ_node_x, succ_node_y)

            road_graph.add_edge(
                pre_node_id,
                succ_node_id,
                id=edge_id,
                node_id_list=[pre_node_id, succ_node_id],
                level=node_info[2] if len(node_info) > 2 else 5,
                oneway=node_info[3] if len(node_info) > 3 else False,
                length=node_dist,
            )

        with open(road_graph_pkl_file, "wb") as f:
            pickle.dump(road_graph, f)
        print("Generate and save road network graph pickle file successfully!")
    else:
        with open(road_graph_pkl_file, "rb") as f:
            road_graph = pickle.load(f)
        print("Load road network graph successfully!")

    return road_graph


def calc_boundary_and_center_xy(road_node_loc_dict):
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for _, (x, y) in road_node_loc_dict.items():
        if x > max_x:
            max_x = x
        if x < min_x:
            min_x = x
        if y > max_y:
            max_y = y
        if y < min_y:
            min_y = y

    mean_x, mean_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    range_x, range_y = max_x - min_x + 1, max_y - min_y + 1

    print(f"min x: {min_x}, max x: {max_x}, min y: {min_y}, max y: {max_y}")
    print(f"mean x: {mean_x}, mean y: {mean_y}, x range: {range_x}, y range: {range_y}")

    return {
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "mean_x": mean_x,
        "mean_y": mean_y,
        "range_x": range_x,
        "range_y": range_y,
    }


def gen_cam_pos_dict_urbanvehicle(cam_file, cam_dict_file):
    if not os.path.exists(cam_dict_file):
        cam_loc = []
        with open(cam_file) as file:
            for line in file:
                cam_loc.append(json.loads(line))
        print(f"The number of cameras is {len(cam_loc)}")

        cam_dict = {}
        for cam_info in cam_loc:
            cam_id = cam_info["camera_id"]
            x, y = cam_info["position"]
            cam_dict[cam_id] = [x, y]

        with open(cam_dict_file, "wb") as f:
            pickle.dump(cam_dict, f)
        print("save the cameras dictionary as a .pkl file successfully!")
    else:
        with open(cam_dict_file, "rb") as f:
            cam_dict = pickle.load(f)
        print("load the cameras dictionary as a .pkl file successfully!")

    return cam_dict


def load_road_network_urbanvehicle(road_map_file):
    with open(road_map_file, "r", encoding="utf-8") as f:
        road_graph_info = json.load(f)

    node_list = []
    edge_list = []
    for curr_info in road_graph_info:
        if curr_info["type"] == "node":
            node_list.append(curr_info)
        if curr_info["type"] == "way":
            edge_list.append(curr_info)

    road_node_loc_dict = {}
    for curr_node_info in node_list:
        node_id = curr_node_info["id"]
        x, y = curr_node_info["xy"]
        road_node_loc_dict[node_id] = [x, y]

    edge_dict = {}
    road_sec_id = 0
    for curr_edge_info in edge_list:
        oneway = curr_edge_info["oneway"]
        road_level = curr_edge_info["level"]
        for node_idx in range(len(curr_edge_info["nodes"]) - 1):
            pre_node_id = curr_edge_info["nodes"][node_idx]
            succ_node_id = curr_edge_info["nodes"][node_idx + 1]
            edge_dict[road_sec_id] = [pre_node_id, succ_node_id, road_level, oneway]
            road_sec_id += 1

    return road_node_loc_dict, edge_dict


def load_road_node_traj_gt(road_node_traj_gt_file, road_node_traj_gt_path, frame_thres, fps):
    if not os.path.exists(road_node_traj_gt_file):
        traj_gt_file_list = sorted(os.listdir(road_node_traj_gt_path))
        road_node_traj_gt_dict = {}
        for gt_file_name in tqdm(traj_gt_file_list):
            stem, ext = os.path.splitext(gt_file_name)
            if ext.lower() != ".txt" or not stem.lstrip("-").isdigit():
                continue
            vid = int(stem)
            curr_traj_gt_file = os.path.join(road_node_traj_gt_path, gt_file_name)
            traj_gt_list = []
            with open(curr_traj_gt_file, "r") as f:
                for line in f.readlines():
                    line_info = re.split(" |,", line.rstrip("\n"))
                    curr_node_id = int(line_info[0])
                    s_tm = float(line_info[1]) / fps
                    e_tm = float(line_info[2]) / fps
                    if frame_thres is not None and s_tm > frame_thres and e_tm > frame_thres:
                        continue
                    traj_gt_list.append((curr_node_id, s_tm, e_tm))
            road_node_traj_gt_dict[vid] = traj_gt_list

        with open(road_node_traj_gt_file, "wb") as f:
            pickle.dump(road_node_traj_gt_dict, f)
        print("generate the road_node_traj_gt_dict and saved it successfully!")
    else:
        with open(road_node_traj_gt_file, "rb") as f:
            road_node_traj_gt_dict = pickle.load(f)
        print("load the road_node_traj_gt_dict successfully!")
    return road_node_traj_gt_dict


def merge_tm_adj_points(points, adj_range=100):
    """Merge temporally adjacent redundant trajectory points at the same road node.

    Args:
        points: List of trajectory points as [(node_id, timestamp), ...].
        adj_range: Time window threshold in seconds; points at the same node within this window are averaged.

    Returns:
        List of merged trajectory points as [(node_id, merged_timestamp), ...].
    """
    node_to_tms = defaultdict(list)
    for node, tm in points:
        node_to_tms[node].append(tm)

    merge_points = []
    for node, tms in node_to_tms.items():
        if len(tms) == 1:
            merge_points.append((node, tms[0]))
            continue
        tms.sort()
        min_tm = tms[0]
        one_cluster = [tms[0]]
        for tm in tms[1:]:
            if tm - min_tm <= adj_range:
                one_cluster.append(tm)
            else:
                merge_points.append((node, float(np.mean(one_cluster))))
                one_cluster = [tm]
                min_tm = tm
        merge_points.append((node, float(np.mean(one_cluster))))
    return merge_points


def prepare_road_node_index(road_graph):
    """Extract arrays of node IDs and XY coordinates from the road network graph for nearest-neighbor lookup.

    Args:
        road_graph: networkx.MultiDiGraph whose nodes contain 'x' and 'y' attributes.

    Returns:
        (node_ids, node_xy): node_ids as an object-dtype array, and node_xy as an (N, 2) float64 array.
    """
    node_ids, node_xy = [], []
    for node_id, attr in road_graph.nodes(data=True):
        if "x" in attr and "y" in attr:
            node_ids.append(node_id)
            node_xy.append([float(attr["x"]), float(attr["y"])])
    if not node_ids:
        return np.array([], dtype=object), np.zeros((0, 2), dtype=np.float64)
    return np.array(node_ids, dtype=object), np.array(node_xy, dtype=np.float64)


def dense_path_to_node_time_path(dense_path, road_node_ids, road_node_xy):
    """Map a dense XY path to the nearest road network nodes to produce a (node_id, timestamp) sequence.

    Args:
        dense_path: Dense path points as [[x, y, tm], ...].
        road_node_ids: Node ID array returned by prepare_road_node_index.
        road_node_xy: Node coordinate array of shape (N, 2) returned by prepare_road_node_index.

    Returns:
        Time-sorted node-level trajectory as [(node_id, timestamp), ...].
    """
    if not dense_path or road_node_xy.shape[0] == 0:
        return []

    rid_tm = []
    for p in dense_path:
        if not isinstance(p, (list, tuple)) or len(p) < 3:
            continue
        x, y, tm = float(p[0]), float(p[1]), float(p[2])
        dist2 = (road_node_xy[:, 0] - x) ** 2 + (road_node_xy[:, 1] - y) ** 2
        rid = road_node_ids[int(np.argmin(dist2))]
        rid_tm.append((rid, tm))

    rid_tm = merge_tm_adj_points(rid_tm, adj_range=100)
    rid_tm.sort(key=lambda x: x[1])
    return rid_tm
