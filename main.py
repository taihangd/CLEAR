"""Main pipeline entrypoint for cluster-level road-network trajectory recovery (C1 + C2 + C3)."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
import pickle
from typing import Any, Dict

import numpy as np

from cluster_algorithm import SigCluster
from common import (
    calc_boundary_and_center_xy,
    construct_road_network_urbanvehicle,
    gen_cam_pos_dict_urbanvehicle,
    load_road_network_urbanvehicle,
    load_road_node_traj_gt,
    merge_tm_adj_points,
)
from eval import eval_trajs, evaluate, gen_vid2cid
from load_data import load_car_feat
from multi_llm_router import PerformanceAwareMultiLLMRouter, build_case_features_from_trace
from traj_rec_st_cot import LLMClient, build_cluster_track_context
from utils.fixed_expert_core import retrieve_mapping_experience_with_local_fallback
from utils.fixed_expert_recovery import (
    build_camera_mapping_context,
    build_camera_mapping_prompt,
    build_cluster_corridor_subgraph,
    evaluate_cluster_invocation_gate,
    execute_three_tier_corridor_verification,
    parse_camera_mapping_response,
)
from utils.yaml_config_hook import add_config_arguments, flatten_dict, unflatten_dict, yaml_config_hook


def preprocess_paths_with_length(shortest_path_results, road_graph):
    """Precompute cumulative_lengths and total_length for each shortest path."""
    processed_paths = {}
    for key, paths in shortest_path_results.items():
        path_nodes = paths[0]
        cumulative_lengths = [0.0]
        total_length = 0.0
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            edge_data = road_graph[u][v]
            if "length" in edge_data:
                length = float(edge_data["length"])
            elif edge_data:
                length = float(min(
                    (d.get("length", 1.0) for d in edge_data.values() if isinstance(d, dict)),
                    default=1.0,
                ))
            else:
                length = 1.0
            total_length += length
            cumulative_lengths.append(total_length)
        processed_paths[key] = (path_nodes, cumulative_lengths, total_length)
    return processed_paths


def complete_trajectories_fast(cluster_traj_dict, processed_paths):
    """Interpolate along precomputed shortest paths while preserving every observed endpoint."""
    completed = {}
    for label, observations in cluster_traj_dict.items():
        trajectory = [tuple(observations[0])] if observations else []
        for start, end in zip(observations, observations[1:]):
            rid1, t1 = start
            rid2, t2 = end
            entry = processed_paths.get((rid1, rid2))
            if entry is not None:
                nodes, lengths, total = entry
                if (nodes and nodes[0] == rid1 and nodes[-1] == rid2
                        and len(nodes) == len(lengths) and total > 0):
                    trajectory.extend(
                        (node, t1 + distance / total * (t2 - t1))
                        for node, distance in zip(nodes[1:-1], lengths[1:-1])
                    )
            trajectory.append((rid2, t2))
        completed[label] = trajectory
    return completed


def _build_top2_camera_candidates(road_graph, cam_xy_dict, camera_ids):
    """Compute geometric Top-2 candidate road nodes Gamma_2(c_i) per camera (Section 3.3.2)."""
    if not isinstance(cam_xy_dict, dict) or road_graph is None:
        return None
    node_xy = [
        (int(node), float(attr["x"]), float(attr["y"]))
        for node, attr in road_graph.nodes(data=True)
        if "x" in attr and "y" in attr
    ]
    if not node_xy:
        return None
    candidates = {}
    for cam in sorted({int(c) for c in camera_ids}):
        xy = cam_xy_dict.get(cam, cam_xy_dict.get(str(cam)))
        if not isinstance(xy, (list, tuple)) or len(xy) < 2:
            continue
        cx, cy = float(xy[0]), float(xy[1])
        ranked = sorted((math.hypot(cx - nx_val, cy - ny_val), node) for node, nx_val, ny_val in node_xy)[:2]
        if ranked:
            candidates[cam] = [node for _, node in ranked]
    return candidates or None


def _build_public_mapping_context(camera_trace, *, road_graph, camera_to_node, cam_xy_dict):
    """Construct the public camera_mapping_v1 context from an ordered camera trace."""
    if road_graph is None or road_graph.number_of_nodes() < 2 or not isinstance(cam_xy_dict, dict):
        return None
    trace, points, provenance, positions = [], [], [], {}
    for idx, (cam, stamp) in enumerate(camera_trace):
        cam_id, ts = int(cam), float(stamp)
        xy_raw = cam_xy_dict.get(cam_id, cam_xy_dict.get(str(cam_id)))
        if not isinstance(xy_raw, (list, tuple)) or len(xy_raw) < 2:
            return None
        xy = [float(xy_raw[0]), float(xy_raw[1])]
        trace.append([cam_id, ts])
        points.append([xy[0], xy[1], ts])
        positions[str(cam_id)] = xy
        provenance.append({
            "input_index": idx,
            "camera_id": cam_id,
            "prepared_time": ts,
            "raw_record_indices": [idx],
            "transformed_times": [ts],
            "raw_times_s": [ts],
            "aggregation": {"exact_reproduction": True, "method": "one_raw_record_event"},
        })
    if not trace:
        return None
    query = {
        "camera_trace": trace,
        "observed_points": points,
        "observation_provenance": provenance,
        "camera_positions": positions,
    }
    ctx = build_cluster_track_context(
        query,
        road_graph=road_graph,
        camera_to_node=camera_to_node,
        shortest_paths={},
        planning_protocol="road_intents_v1",
        trajectory_scope="single_cluster",
        road_intent_config={"max_waypoints": 2},
    )
    return build_camera_mapping_context(ctx, road_graph=road_graph)


def _make_llm_road_intent_callable(camera_to_node, cam_xy_dict):
    """Build a live C2 road-intent callable backed by LLMClient on corridor subgraph G_q."""
    def _call(*, camera_trace, corridor_subgraph, model, retrieved_experience=None, **_kwargs):
        ctx = _build_public_mapping_context(
            camera_trace,
            road_graph=corridor_subgraph,
            camera_to_node=camera_to_node,
            cam_xy_dict=cam_xy_dict,
        )
        if ctx is None:
            return {"camera_node_choices": {}, "interval_intents": {}}
        prompt = build_camera_mapping_prompt(
            ctx,
            include_pair_evidence=True,
            road_graph=corridor_subgraph,
            retrieved_experience=retrieved_experience,
        )
        raw_text = LLMClient.call_llm(prompt, model=model)
        groups, _, mapping = parse_camera_mapping_response(raw_text, ctx, road_graph=corridor_subgraph)
        obs_id_to_idx = {
            obs["obs_id"]: int(obs["input_index"])
            for obs in ctx.get("observations", [])
            if "obs_id" in obs and "input_index" in obs
        }
        intents = {}
        intents_by_obs_pair = {}
        retained_indices = None
        tracks = groups.get("tracks") or []
        if tracks:
            retained_indices = [
                obs_id_to_idx[oid]
                for oid in (tracks[0].get("obs_ids") or [])
                if oid in obs_id_to_idx
            ]
            for seg_idx, item in enumerate(tracks[0].get("interval_intents") or []):
                via_list = [int(n) for n in (item.get("via_nodes") or [])]
                intents[seg_idx] = via_list
                f_oid, t_oid = item.get("from_obs_id"), item.get("to_obs_id")
                if f_oid in obs_id_to_idx and t_oid in obs_id_to_idx:
                    intents_by_obs_pair[(obs_id_to_idx[f_oid], obs_id_to_idx[t_oid])] = via_list
        return {
            "camera_node_choices": mapping,
            "interval_intents": intents,
            "interval_intents_by_obs_pair": intents_by_obs_pair,
            "retained_obs_indices": retained_indices,
        }

    return _call


def prep_road_network_urbanvehicle(
    road_map_file,
    road_node_xy_info_file,
    road_graph_pkl_file,
    cam_file,
    cam_dict_file,
):
    """Prepare the directed road network graph and planar camera coordinates."""
    road_node_xy_dict, edge_dict = load_road_network_urbanvehicle(road_map_file)
    boundary_info = calc_boundary_and_center_xy(road_node_xy_dict)
    if not os.path.exists(road_node_xy_info_file):
        with open(road_node_xy_info_file, "wb") as f:
            pickle.dump(
                [
                    road_node_xy_dict,
                    [
                        boundary_info["min_x"],
                        boundary_info["min_y"],
                        boundary_info["max_x"],
                        boundary_info["max_y"],
                    ],
                ],
                f,
            )
    road_graph = construct_road_network_urbanvehicle(
        road_node_xy_dict, edge_dict, road_graph_pkl_file
    )
    cam_dict = gen_cam_pos_dict_urbanvehicle(cam_file, cam_dict_file)
    return road_graph, cam_dict, boundary_info


def load_road_assets(args):
    """Load directed road network, camera-to-node bindings, and precomputed shortest paths."""
    road_graph, cam_xy_dict, boundary_info = prep_road_network_urbanvehicle(
        args.road_map_file,
        args.road_node_xy_info_file,
        args.road_graph_file,
        args.cam_file,
        args.cam_xy_dict_file,
    )
    with open(args.cid_rid_correspondence_file, "rb") as f:
        cid_to_rid = pickle.load(f)
    camera_to_node = (
        {int(k): int(v) for k, v in cid_to_rid.items()}
        if isinstance(cid_to_rid, dict)
        else {int(item["id"]): int(item["node_id"]) for item in cid_to_rid}
    )
    with open(args.rid_shortest_path_results_file, "rb") as f:
        shortest_paths = pickle.load(f)
    traj_gt_dict = load_road_node_traj_gt(
        args.road_node_traj_gt_file,
        args.road_node_traj_gt_path,
        getattr(args, "frame_thres", None),
        args.fps,
    )
    return road_graph, camera_to_node, cam_xy_dict, shortest_paths, traj_gt_dict, boundary_info


def recover_cluster_trajectory(
    camera_trace,
    *,
    road_graph,
    camera_to_node,
    shortest_paths,
    cam_xy_dict=None,
    router: PerformanceAwareMultiLLMRouter | None = None,
    llm_callable=None,
    experience_bank: Dict[str, Any] | None = None,
    retrieved_experience: Any | None = None,
    gate_accepted: bool = False,
    interval_intents: Dict[int, list] | None = None,
    sp_cache: Dict[Any, Any] | None = None,
    max_speed_mps: float = 25.0,
    max_span_m: float = 1200.0,
    min_dwell_sec: float = 30.0,
    detour_ratio_threshold: float = 1.05,
) -> Dict[str, Any]:
    """Execute the closed-loop C1 -> C3 -> C2 trajectory recovery pipeline (Algorithm 1).

    Stage 1 (C1): Evaluate cluster-level kinematic conflict gate gamma_inv(O_q).
                  Clean single-corridor clusters (gamma_inv == 0) bypass LLM inference.
                  For triggered clusters, optionally retrieve and gate transition
                  experiences via G5^kin when an experience_bank is supplied.
    Stage 2 (C3): For triggered clusters (gamma_inv == 1), prune the full city graph G
                  to the lossless corridor subgraph G_q (Eq. 8) over Top-2 camera candidates
                  Gamma_2(c_i) with 1-hop corridor halo (k_hop=1) and select model mu*(q)
                  via the gate-aware Pareto router (Eq. 7).
    Stage 3 (C2): Invoke mu*(q) (when llm_callable is provided) to predict camera-node
                  bindings, retained track observations, and interval via-nodes, then
                  execute three-tier bounded Dijkstra verification and edge-proportional
                  timestamp interpolation (Eq. 5).
    """
    merged_cam_trace = sorted(merge_tm_adj_points(list(camera_trace), adj_range=100), key=lambda x: x[1])
    active_mapping = dict(camera_to_node or {})
    rid_trace = [
        (int(active_mapping[cam]), float(ts))
        for cam, ts in merged_cam_trace
        if cam in active_mapping
    ]
    rid_trace = sorted(merge_tm_adj_points(rid_trace, adj_range=100), key=lambda x: x[1])

    # Stage 1 (C1): Cluster-level kinematic invocation gate gamma_inv(O_q)
    gate_info = evaluate_cluster_invocation_gate(
        merged_cam_trace,
        road_graph=road_graph,
        camera_to_node=active_mapping,
        shortest_paths=shortest_paths,
        max_speed_mps=float(max_speed_mps),
        max_span_m=float(max_span_m),
        min_dwell_sec=float(min_dwell_sec),
        detour_ratio_threshold=float(detour_ratio_threshold),
    )
    if not gate_info["invoke_llm"]:
        verified = execute_three_tier_corridor_verification(
            rid_trace,
            road_graph=road_graph,
            shortest_paths=shortest_paths,
            max_speed_mps=float(max_speed_mps),
            max_span_m=float(max_span_m),
            min_dwell_sec=float(min_dwell_sec),
            detour_ratio_threshold=float(detour_ratio_threshold),
        )
        return {
            "trajectory": verified["trajectory"],
            "invoked_llm": False,
            "gate_info": gate_info,
            "retrieval_report": None,
            "subgraph_stats": None,
            "routing_decision": None,
            "verification_audit": verified,
        }

    # Optional C1 Level-2 applicability-gated transition retrieval (G5^kin)
    retrieval_report = None
    active_experience = retrieved_experience
    active_gate_accepted = bool(gate_accepted) or (active_experience is not None)
    if isinstance(experience_bank, dict) and "strict_index" in experience_bank and "local_index" in experience_bank:
        mapping_ctx = _build_public_mapping_context(
            merged_cam_trace,
            road_graph=road_graph,
            camera_to_node=active_mapping,
            cam_xy_dict=cam_xy_dict,
        )
        if mapping_ctx is not None:
            local_cases = experience_bank["local_index"].get("cases", {})
            retrieval_report = retrieve_mapping_experience_with_local_fallback(
                mapping_ctx,
                experience_bank["strict_index"],
                experience_bank["local_index"],
                source_eligible_ids=experience_bank.get("source_eligible_ids", list(local_cases.keys())),
                candidate_rows=experience_bank.get("candidate_rows", []),
                local_gate=experience_bank.get("local_gate", "g5_kinematic_boundary_v1"),
                road_graph=road_graph,
                teacher_kinematics_index=experience_bank.get("teacher_kinematics_index", {}),
                v_stop_mps=float(experience_bank.get("v_stop_mps", 5.0)),
            )
            if retrieval_report.get("gate", {}).get("inject") is True:
                active_gate_accepted = True
                active_experience = retrieval_report.get("payload")

    # Stage 2 (C3): Corridor subgraph pruning (G -> G_q) and difficulty-aware routing
    camera_candidates = _build_top2_camera_candidates(
        road_graph, cam_xy_dict, (cam for cam, _ in merged_cam_trace)
    )
    corridor_subgraph, subgraph_stats = build_cluster_corridor_subgraph(
        merged_cam_trace,
        road_graph=road_graph,
        camera_to_node=active_mapping,
        camera_candidates=camera_candidates,
        shortest_paths=shortest_paths,
        sp_cache=sp_cache if sp_cache is not None else {},
        pair_scope="consecutive_and_two_hop",
        k_hop=1,
    )
    case_features = build_case_features_from_trace(
        merged_cam_trace,
        cam_xy_dict=cam_xy_dict,
        rid_trace=rid_trace,
        input_char_length=subgraph_stats["corridor_node_count"] * 64,
        retrieval_count=1 if active_gate_accepted else 0,
        road_graph=road_graph,
        camera_to_node=active_mapping,
    )
    routing_decision = None
    selected_model = None
    if router is not None:
        routing_decision = router.route(
            case_features,
            gate_aware=True,
            gate_accepted=bool(active_gate_accepted),
        )
        selected_model = routing_decision.get("selected_model")

    # Stage 3 (C2): Road-intent generation (if live callable supplied) + three-tier verification on G_q
    planned_intents = dict(interval_intents or {})
    if callable(llm_callable):
        llm_output = llm_callable(
            camera_trace=merged_cam_trace,
            corridor_subgraph=corridor_subgraph,
            model=selected_model,
            case_features=case_features,
            retrieved_experience=active_experience,
        )
        if isinstance(llm_output, dict):
            node_choices = llm_output.get("camera_node_choices")
            if isinstance(node_choices, dict) and node_choices:
                active_mapping = {**active_mapping, **{int(k): int(v) for k, v in node_choices.items()}}
            retained_indices = llm_output.get("retained_obs_indices")
            obs_pair_intents = llm_output.get("interval_intents_by_obs_pair")
            if isinstance(retained_indices, list):
                indexed_obs = [
                    (orig_idx, int(active_mapping[merged_cam_trace[orig_idx][0]]), float(merged_cam_trace[orig_idx][1]))
                    for orig_idx in retained_indices
                    if 0 <= orig_idx < len(merged_cam_trace) and merged_cam_trace[orig_idx][0] in active_mapping
                ]
                cleaned_anchors = []
                anchor_orig_sets = []
                for orig_idx, node, ts in indexed_obs:
                    if not cleaned_anchors:
                        cleaned_anchors.append((node, ts))
                        anchor_orig_sets.append({orig_idx})
                        continue
                    prev_node, prev_ts = cleaned_anchors[-1]
                    if ts < prev_ts:
                        continue
                    if node == prev_node and (ts - prev_ts) <= float(min_dwell_sec):
                        anchor_orig_sets[-1].add(orig_idx)
                        continue
                    cleaned_anchors.append((node, ts))
                    anchor_orig_sets.append({orig_idx})
                rid_trace = cleaned_anchors
                if isinstance(obs_pair_intents, dict) and obs_pair_intents:
                    planned_intents = {}
                    for seg_idx in range(len(cleaned_anchors) - 1):
                        left_set, right_set = anchor_orig_sets[seg_idx], anchor_orig_sets[seg_idx + 1]
                        for (f_idx, t_idx), via_list in obs_pair_intents.items():
                            if f_idx in left_set and t_idx in right_set:
                                planned_intents[seg_idx] = list(via_list)
                                break
                elif isinstance(llm_output.get("interval_intents"), dict):
                    planned_intents = {int(k): list(v) for k, v in llm_output["interval_intents"].items()}
            else:
                if isinstance(node_choices, dict) and node_choices:
                    rid_trace = [
                        (int(active_mapping[cam]), float(ts))
                        for cam, ts in merged_cam_trace
                        if cam in active_mapping
                    ]
                    rid_trace = sorted(merge_tm_adj_points(rid_trace, adj_range=100), key=lambda x: x[1])
                if isinstance(llm_output.get("interval_intents"), dict):
                    planned_intents = {int(k): list(v) for k, v in llm_output["interval_intents"].items()}

    exec_graph = corridor_subgraph if corridor_subgraph.number_of_nodes() > 0 else road_graph
    verified = execute_three_tier_corridor_verification(
        rid_trace,
        interval_intents=planned_intents,
        road_graph=exec_graph,
        shortest_paths=shortest_paths,
        max_speed_mps=float(max_speed_mps),
        max_span_m=float(max_span_m),
        min_dwell_sec=float(min_dwell_sec),
        detour_ratio_threshold=float(detour_ratio_threshold),
    )
    return {
        "trajectory": verified["trajectory"],
        "invoked_llm": True,
        "gate_info": gate_info,
        "retrieval_report": retrieval_report,
        "subgraph_stats": subgraph_stats,
        "routing_decision": routing_decision,
        "verification_audit": verified,
    }


def run_pipeline(args) -> Dict[str, Any]:
    """Execute upstream clustering, C1/C2/C3 trajectory recovery, and multi-metric evaluation."""
    dataset = str(getattr(args, "dataset", "uv"))
    records_all = load_car_feat(args.record_file, args.fps, getattr(args, "use_plate", True))
    if dataset in {"uv", "uv_z"}:
        for record in records_all:
            record["time"] += 10000

    if getattr(args, "sort_record", True):
        sorted_indices = np.argsort([r["time"] for r in records_all], kind="stable")
        records = [records_all[i] for i in sorted_indices]
    else:
        records = records_all

    clustering_res_file = getattr(args, "clustering_res_file", "")
    if clustering_res_file and os.path.exists(clustering_res_file):
        with open(clustering_res_file, "rb") as f:
            labels = pickle.load(f)
    else:
        sig_cfg = getattr(args, "sig_cluster", {}) or {}
        feat_dim = int(getattr(args, "feat_dim", 256))
        f_car = [x["car_feature"] for x in records]
        f_plate = [x["plate_feature"] for x in records]
        sig_cluster = SigCluster(
            feature_dims=[feat_dim, feat_dim, feat_dim],
            ngpu=int(getattr(args, "ngpu", 1)),
            useFloat16=bool(sig_cfg.get("useFloat16", False)),
        )
        labels = sig_cluster.fit(
            [[a, b, a] for a, b in zip(f_car, f_plate)],
            None,
            weights=sig_cfg.get("weights", [0.5, 0.5, 0.0]),
            similarity_threshold=float(sig_cfg.get("sim_thres", 0.5)),
            topK=int(sig_cfg.get("topK", 50)),
            query_num=int(sig_cfg.get("query_num", 1000)),
            normalization=True,
        )

    clustering_metrics = evaluate(records, labels, log=True)
    road_graph, camera_to_node, cam_xy_dict, shortest_paths, traj_gt_dict, _ = load_road_assets(args)
    processed_paths = preprocess_paths_with_length(shortest_paths, road_graph)
    vid2cid = gen_vid2cid(records, labels)

    cluster_cam_traces = defaultdict(list)
    for rec, cid in zip(records, labels):
        if int(cid) >= 0:
            cluster_cam_traces[int(cid)].append((int(rec["camera_id"]), float(rec["time"])))

    routing_cfg = getattr(args, "llm_routing", {}) or {}
    candidate_models = routing_cfg.get("candidate_models") or ["gpt-4.1-nano", "gpt-4.1", "qwen3.5-plus"]
    default_model = routing_cfg.get("default_model", candidate_models[0])
    quality_tolerance = float(routing_cfg.get("quality_tolerance", 0.01))
    router = PerformanceAwareMultiLLMRouter(
        candidate_models,
        default_model=default_model,
        delta=quality_tolerance,
        seed=int(getattr(args, "seed", 0)),
    )
    ckpt_file = routing_cfg.get("router_ckpt_file")
    if ckpt_file and os.path.exists(ckpt_file):
        router, _ = PerformanceAwareMultiLLMRouter.load(ckpt_file)

    experience_bank = None
    bank_file = getattr(args, "experience_bank_file", "")
    if bank_file and os.path.exists(bank_file):
        with open(bank_file, "rb") as f:
            experience_bank = pickle.load(f)

    llm_callable = None
    if getattr(args, "allow_api", False):
        LLMClient.configure(getattr(args, "llm_client", {}) or {})
        llm_callable = _make_llm_road_intent_callable(camera_to_node, cam_xy_dict)

    recovered_traj_dict: Dict[int, list] = {}
    sp_cache: Dict[Any, Any] = {}
    max_speed = float(getattr(args, "max_speed_mps", 25.0))
    max_span = float(getattr(args, "max_span_m", 1200.0))
    min_dwell = float(getattr(args, "min_dwell_sec", 30.0))
    detour_ratio = float(getattr(args, "detour_ratio_threshold", 1.05))

    for cid, raw_trace in sorted(cluster_cam_traces.items()):
        outcome = recover_cluster_trajectory(
            raw_trace,
            road_graph=road_graph,
            camera_to_node=camera_to_node,
            shortest_paths=processed_paths,
            cam_xy_dict=cam_xy_dict,
            router=router,
            llm_callable=llm_callable,
            experience_bank=experience_bank,
            sp_cache=sp_cache,
            max_speed_mps=max_speed,
            max_span_m=max_span,
            min_dwell_sec=min_dwell,
            detour_ratio_threshold=detour_ratio,
        )
        recovered_traj_dict[cid] = outcome["trajectory"]

    scale_coef_list = [
        float(x) for x in getattr(args, "scale_coef", [0.04267, 0.04267, 0.04267])
    ]
    eval_summary = eval_trajs(
        records,
        labels,
        traj_gt_dict,
        recovered_traj_dict,
        road_graph,
        scale_coef_list,
        vid2cid,
        "recovered",
        True,
    )
    return {
        "clustering": clustering_metrics,
        "trajectory_recovery": eval_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Trajectory Recovery Pipeline (C1 + C2 + C3)")
    parser.add_argument("--cfg_file", type=str, default="config/uv.yaml", help="Path to YAML configuration file")
    parser.add_argument("--allow-api", dest="allow_api", action="store_true", help="Enable live LLM API requests")
    known_args, _ = parser.parse_known_args()
    cfg = yaml_config_hook(known_args.cfg_file)
    add_config_arguments(parser, flatten_dict(cfg))
    args = parser.parse_args()
    for key, value in unflatten_dict(vars(args)).items():
        setattr(args, key, value)
    args.allow_api = known_args.allow_api
    results = run_pipeline(args)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
