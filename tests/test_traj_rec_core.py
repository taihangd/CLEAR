"""Unit tests for the C1 retrieval gate, C2 three-tier corridor verification, and C3 router modules."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import merge_tm_adj_points
from eval import eval_traj_volume_mse_sparse
from main import _make_llm_road_intent_callable, complete_trajectories_fast, recover_cluster_trajectory
from multi_llm_router import PerformanceAwareMultiLLMRouter
from traj_rec_st_cot import LLMClient
from utils.fixed_expert_core import (
    FixedRetriever,
    build_teacher_interval_kinematics_index,
    evaluate_g5_kinematic_boundary_gate,
)
from utils.fixed_expert_recovery import (
    build_cluster_corridor_subgraph,
    evaluate_cluster_invocation_gate,
    execute_three_tier_corridor_verification,
)
from utils.trajectory_features import FeatureScaler, build_signature, structural_similarity
from utils.yaml_config_hook import yaml_config_hook


def _toy_road_graph_and_paths():
    """Construct a directed toy road network with a primary corridor and a spike node."""
    graph = nx.DiGraph()
    coords = {10: (0.0, 0.0), 20: (100.0, 0.0), 30: (200.0, 0.0), 40: (300.0, 0.0), 99: (100.0, 500.0)}
    for node_id, (x, y) in coords.items():
        graph.add_node(node_id, x=x, y=y)
    edges = [
        (10, 20, 100.0),
        (20, 30, 100.0),
        (30, 40, 100.0),
        (10, 99, 600.0),
        (99, 30, 600.0),
    ]
    for u, v, length in edges:
        graph.add_edge(u, v, length=length)

    shortest_paths = {
        (10, 20): ([10, 20], [0.0, 100.0], 100.0),
        (20, 30): ([20, 30], [0.0, 100.0], 100.0),
        (10, 30): ([10, 20, 30], [0.0, 100.0, 200.0], 200.0),
        (30, 40): ([30, 40], [0.0, 100.0], 100.0),
        (10, 99): ([10, 99], [0.0, 600.0], 600.0),
        (99, 30): ([99, 30], [0.0, 600.0], 600.0),
    }
    return graph, shortest_paths


def test_c1_invocation_gate_and_g5_kinematic_clamping():
    """Verify C1 Level-1 invocation gate (gamma_inv) and Level-2 G5/G5^kin boundary clamping."""
    graph, shortest_paths = _toy_road_graph_and_paths()
    cam_to_node = {1: 10, 2: 20, 3: 30, 9: 99}

    clean_gate = evaluate_cluster_invocation_gate(
        [(1, 0.0), (2, 10.0), (3, 20.0)],
        road_graph=graph,
        camera_to_node=cam_to_node,
        shortest_paths=shortest_paths,
        max_speed_mps=25.0,
        max_span_m=1200.0,
    )
    assert clean_gate["invoke_llm"] is False

    detour_gate = evaluate_cluster_invocation_gate(
        [(1, 0.0), (9, 30.0), (3, 60.0)],
        road_graph=graph,
        camera_to_node=cam_to_node,
        shortest_paths=shortest_paths,
        max_speed_mps=25.0,
        max_span_m=1200.0,
        detour_ratio_threshold=1.05,
    )
    assert detour_gate["invoke_llm"] is True
    assert any(seg["trigger"] == "three_point_detour_anomaly" for seg in detour_gate["conflict_segments"])

    teacher_cases = [{
        "split": "train",
        "extraction_eligible": True,
        "events": [{"obs_id": "o0", "camera_id": 1}, {"obs_id": "o1", "camera_id": 2}, {"obs_id": "o2", "camera_id": 3}],
        "intervals": [{
            "from_camera_id": 1,
            "to_camera_id": 2,
            "from_node": 10,
            "to_node": 20,
            "to_obs_id": "o1",
            "training_observed_time_gap_s": 10.0,
        }],
    }]
    kin_index = build_teacher_interval_kinematics_index(teacher_cases)
    context = {
        "observations": [
            {"obs_id": "q0", "camera_id": 1, "road_node": 10, "time": 0.0},
            {"obs_id": "q1", "camera_id": 2, "road_node": 30, "time": 12.0},
        ],
        "camera_mapping_candidates": [
            {"camera_id": 1, "candidates": [{"node": 10}]},
            {"camera_id": 2, "candidates": [{"node": 30}, {"node": 20}]},
        ],
    }
    g5_audit = {
        "allow": True,
        "bindings": [{
            "passes": True,
            "touches_boundary": True,
            "current_obs_id_pairs": [["q0", "q1"]],
            "directed_camera_pair": [1, 2],
            "camera_id": 2,
            "public_prior_node": 30,
            "conditional_supported_node": 20,
        }],
    }
    kin_gate = evaluate_g5_kinematic_boundary_gate(
        context, g5_audit, road_graph=graph, teacher_kinematics_index=kin_index, v_stop_mps=5.0
    )
    assert kin_gate["allow"] is False and kin_gate["policy"] == "g5_kinematic_boundary_v1"

    sig_a = build_signature([(1, 0.0), (2, 10.0), (3, 20.0)], graph, cam_to_node)
    sig_b = build_signature([(1, 0.0), (2, 12.0), (3, 24.0)], graph, cam_to_node)
    scaler = FeatureScaler().fit([sig_a, sig_b])
    sim = structural_similarity(sig_a, sig_b, scaler)
    assert sim is not None and 0.6 < sim < 1.0
    assert math.isclose(structural_similarity(sig_a, sig_a, scaler), 1.0, abs_tol=1e-9)
    retriever = FixedRetriever(
        {"cases": [{"case_id": "c1", "source_group_id": "v1", "signature": sig_b, "camera_trace": [[1, 0.0], [2, 12.0], [3, 24.0]]}]},
        config={"mode": "structural_similarity"},
        road_graph=graph,
        camera_to_node=cam_to_node,
        scaler=scaler,
    )
    matches = retriever.retrieve([(1, 0.0), (2, 10.0), (3, 20.0)], topk=1, query_id="q1", source_group_id="v2")
    assert len(matches) == 1 and matches[0][1]["case_id"] == "c1"


def test_c2_three_tier_corridor_verification_and_interpolation():
    """Verify C2 three-tier corridor verification (Tier 1 direct, Tier 2 speed gate, Tier 3 two-hop bypass)."""
    graph, shortest_paths = _toy_road_graph_and_paths()
    noisy_trace = [(10, 0.0), (99, 10.0), (30, 20.0), (40, 30.0)]
    audit = execute_three_tier_corridor_verification(
        noisy_trace,
        road_graph=graph,
        shortest_paths=shortest_paths,
        max_speed_mps=25.0,
        max_span_m=1200.0,
    )
    verified_traj = audit["trajectory"]
    recovered_nodes = [node for node, _ in verified_traj]
    assert 99 not in recovered_nodes
    assert recovered_nodes == [10, 20, 30, 40]
    assert math.isclose(dict(verified_traj)[20], 10.0, abs_tol=1e-9)
    assert audit["tier_counts"][3] == 1

    # Verify slow detour spike bypass via detour_ratio_threshold and prolonged same-node dwell retention
    slow_detour_trace = [(10, 0.0), (99, 40.0), (30, 80.0), (30, 120.0)]
    detour_audit = execute_three_tier_corridor_verification(
        slow_detour_trace,
        road_graph=graph,
        shortest_paths=shortest_paths,
        max_speed_mps=25.0,
        max_span_m=1200.0,
        min_dwell_sec=30.0,
        detour_ratio_threshold=1.05,
    )
    assert [n for n, _ in detour_audit["trajectory"]] == [10, 20, 30, 30]
    assert math.isclose(detour_audit["trajectory"][-1][1], 120.0, abs_tol=1e-9)
    assert detour_audit["tier_counts"][3] == 1

    merged = merge_tm_adj_points([(10, 0.0), (10, 2.0), (20, 15.0)], adj_range=5.0)
    assert len(merged) == 2 and math.isclose(dict(merged)[10], 1.0, abs_tol=1e-9)
    interp = complete_trajectories_fast({0: [(10, 0.0), (30, 20.0)]}, shortest_paths)[0]
    assert len(interp) == 3 and math.isclose(interp[1][1], 10.0, abs_tol=1e-9)


def test_c3_corridor_subgraph_and_quality_tolerance_router():
    """Verify C3 corridor subgraph construction (G_q), example config template, and Pareto routing."""
    graph, shortest_paths = _toy_road_graph_and_paths()
    cam_to_node = {1: 10, 2: 20, 3: 30}
    subgraph, stats = build_cluster_corridor_subgraph(
        [(1, 0.0), (2, 10.0), (3, 20.0)],
        road_graph=graph,
        camera_to_node=cam_to_node,
        camera_candidates={1: [10, 20], 2: [20, 30], 3: [30, 40]},
        shortest_paths=shortest_paths,
        pair_scope="consecutive_and_two_hop",
    )
    assert subgraph.has_node(20) and subgraph.has_node(40) and not subgraph.has_node(99)
    assert stats["corridor_node_count"] == 4

    router = PerformanceAwareMultiLLMRouter(
        ["fast_model", "strong_model"],
        default_model="fast_model",
        delta=0.02,
    )
    router.quality_predictor = True
    router.latency_predictor = True
    router._predict = lambda _feat, _apis: {
        "fast_model": {"predicted_accuracy": 0.89, "predicted_latency_sec": 1.0},
        "strong_model": {"predicted_accuracy": 0.90, "predicted_latency_sec": 8.0},
    }
    choice = router.route({"retrieval_count": 1}, gate_aware=True, gate_accepted=True)
    assert choice["selected_model"] == "fast_model"
    assert choice["routing_strategy"] == "gate_aware_pareto"

    repo_root = Path(__file__).resolve().parents[1]
    cfg = yaml_config_hook(repo_root / "config" / "uv.yaml")
    assert cfg["llm_routing"]["candidate_models"] == [
        "gpt-4.1-nano",
        "gpt-4.1",
        "qwen3.5-plus",
    ]
    assert math.isclose(float(cfg["llm_routing"]["quality_tolerance"]), 0.01, abs_tol=1e-9)
    assert cfg["local_gate"] == "g5_kinematic_boundary_v1"
    assert len(cfg["scale_coef"]) == 3
    LLMClient.configure(cfg["llm_client"])
    gpt_url, _ = LLMClient._resolve_provider_credentials("gpt-4.1", None, "test-openai-key")
    qwen_url, _ = LLMClient._resolve_provider_credentials("qwen3.5-plus", None, "test-dashscope-key")
    assert gpt_url == "https://api.openai.com/v1/chat/completions"
    assert qwen_url == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    # Verify 3-element scale_coef compatibility with Vol-MSE evaluator
    mean_vol_mse, _, _ = eval_traj_volume_mse_sparse(
        records=[{"vehicle_id": 1}, {"vehicle_id": 1}],
        labels=[0, 0],
        road_node_traj_gt_dict={1: [(10, 0.0, 2.0), (20, 9.0, 11.0), (30, 19.0, 21.0)]},
        pred_traj_dict={0: [(10, 1.0), (20, 10.0), (30, 20.0)]},
        road_graph=graph,
        scale_coef_list=cfg["scale_coef"],
    )
    assert math.isclose(float(mean_vol_mse), 0.0, abs_tol=1e-6)

    # End-to-end Algorithm 1 closed-loop check: clean bypass (gamma_inv=0) vs routed C1+C3+C2 execution (gamma_inv=1)
    clean_res = recover_cluster_trajectory(
        [(1, 0.0), (2, 10.0), (3, 20.0)],
        road_graph=graph,
        camera_to_node={1: 10, 2: 20, 3: 30, 9: 99},
        shortest_paths=shortest_paths,
        router=router,
        llm_callable=lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("Clean cluster must not invoke LLM")),
    )
    assert clean_res["invoked_llm"] is False
    assert [n for n, _ in clean_res["trajectory"]] == [10, 20, 30]

    called_models = []
    orig_call_llm = LLMClient.call_llm
    try:
        def _fake_call_llm(prompt, model=None, **_kw):
            assert "camera_mapping_v1" in prompt
            called_models.append(model)
            return (
                '{"camera_node_choices":['
                '{"camera_id":1,"road_node":10},'
                '{"camera_id":3,"road_node":30},'
                '{"camera_id":9,"road_node":20}'
                '],"tracks":[{"obs_ids":["record:0","record:1","record:2"],"interval_intents":['
                '{"from_obs_id":"record:0","to_obs_id":"record:1","via_nodes":[]},'
                '{"from_obs_id":"record:1","to_obs_id":"record:2","via_nodes":[]}'
                ']}],"unassigned_obs_ids":[]}'
            )

        LLMClient.call_llm = staticmethod(_fake_call_llm)
        cam_map = {1: 10, 2: 20, 3: 30, 9: 99}
        cam_xy = {1: [0.0, 0.0], 9: [100.0, 260.0], 3: [200.0, 0.0]}
        routed_res = recover_cluster_trajectory(
            [(1, 0.0), (9, 30.0), (3, 60.0)],
            road_graph=graph,
            camera_to_node=cam_map,
            cam_xy_dict=cam_xy,
            shortest_paths=shortest_paths,
            router=router,
            gate_accepted=True,
            llm_callable=_make_llm_road_intent_callable(cam_map, cam_xy),
        )
    finally:
        LLMClient.call_llm = orig_call_llm
    assert routed_res["invoked_llm"] is True
    assert called_models == ["fast_model"]
    assert [n for n, _ in routed_res["trajectory"]] == [10, 20, 30]


if __name__ == "__main__":
    test_c1_invocation_gate_and_g5_kinematic_clamping()
    test_c2_three_tier_corridor_verification_and_interpolation()
    test_c3_corridor_subgraph_and_quality_tolerance_router()
    print("All 3 trajectory recovery core algorithmic checks (C1, C2, C3) passed.")
