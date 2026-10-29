import os
import time
import faiss
import psutil
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def normalize(feature):
    return feature / (np.linalg.norm(feature) + 1e-12)

class FlatSearcher:
    def __init__(self, ngpu=1, useFloat16=False, feat_dim=256):
        gpu_available = (
            bool(ngpu)
            and hasattr(faiss, "StandardGpuResources")
            and hasattr(faiss, "get_num_gpus")
            and faiss.get_num_gpus() > 0
        )
        if gpu_available:
            actual_ngpu = min(int(ngpu), int(faiss.get_num_gpus()))
            flat_config = []
            for i in range(actual_ngpu):
                cfg = faiss.GpuIndexFlatConfig()
                cfg.useFloat16 = useFloat16
                cfg.device = i
                flat_config.append(cfg)
            res = [faiss.StandardGpuResources() for _ in range(actual_ngpu)]
            indexes = [
                faiss.GpuIndexFlatIP(res[i], feat_dim, flat_config[i])
                for i in range(actual_ngpu)
            ]
            self.index = faiss.IndexProxy()
            for sub_index in indexes:
                self.index.addIndex(sub_index)
        else:
            self.index = faiss.IndexFlatIP(feat_dim)

    def search_by_topk(self, query, gallery, topk=16):
        if query is None or len(query) == 0:
            return np.array([]), np.array([])
        
        self.index.reset()
        self.index.add(gallery)
        topk_scores, topk_idxs = self.index.search(query, topk)
        self.index.reset()
        
        return topk_scores, topk_idxs
    
    def search_by_topk_by_blocks(self, query, gallery, topk, query_num):
        if query is None or len(query) == 0:
            return np.array([]), np.array([])
        
        self.index.reset()
        self.index.add(gallery)

        topk_scores_list = []
        topk_idxs_list = []
        if query_num is None:
            query_num = len(query)
        block_num = int(np.ceil(len(query) / query_num))
        for i in range(block_num):
            topk_scores, topk_idxs = self.index.search(query[i * query_num: (i + 1) * query_num], topk)
            topk_scores_list.append(topk_scores)
            topk_idxs_list.append(topk_idxs)
        
        self.index.reset()
        
        topk_scores = np.concatenate(topk_scores_list, axis=0)
        topk_idxs = np.concatenate(topk_idxs_list, axis=0)
        
        return topk_scores, topk_idxs

    def add(self, gallery):
        self.index.add(gallery)

    def reset(self):
        self.index.reset()

class SigCluster:
    def __init__(self, feature_dims=[256, 256], ngpu=1, useFloat16=False):
        self.searchers = {i: FlatSearcher(ngpu, useFloat16, i) for i in set(feature_dims)}
        self.f_dims = feature_dims

    def fit(
        self,
        data,
        initial_labels=None,
        weights=[1],
        similarity_threshold=0.88,
        topK=128,
        query_num=8192,
        normalization=True,
    ):
        snapshots_retrieval_time = time.time()
        if isinstance(weights, float) or isinstance(weights, int):
            weights = [weights] * len(self.f_dims)
        else:
            assert len(weights) == len(self.f_dims)
        if isinstance(topK, int):
            topK = [topK] * len(self.f_dims)
        else:
            assert len(topK) == len(self.f_dims)

        record_num = len(data)
        
        if normalization: 
            print("Normalize")
            data = [
                [None if j is None else normalize(j) for j in i] for i in tqdm(data)
            ]

        print("Search topk")
        fs = []
        f_ids = []
        for i in zip(*data):
            tmp = [x for x in i if x is not None]
            if len(tmp) == 0: # if all snapshots for this feature dimension have no valid values
                continue
            f_id, f = zip(*((j, k) for j, k in enumerate(i) if k is not None))
            fs.append(np.array(f))
            f_ids.append(f_id)

        topk_idxs_list = []
        for f, dim, topk in zip(fs, self.f_dims, topK): # for each modal
            topk_scores, topk_idxs = self.searchers[dim].search_by_topk_by_blocks(f, f, topk, query_num)
            topk_idxs_list.append(topk_idxs)
        f_topks = [
            [
                [
                    f_id[curr_topk_idx]
                    for curr_topk_idx in curr_topk_idxs
                    if curr_topk_idx < i
                ]
                for i, curr_topk_idxs in enumerate(topk_idxs)
            ]
            for topk_idxs, f_id in zip(topk_idxs_list, f_ids)
        ]
        assert all(len(i[0]) == 0 for i in f_topks)

        topks = [[] for _ in range(len(data))]
        for f_topk, f_id in zip(f_topks, f_ids):
            for i, topk in zip(f_id, f_topk):
                topks[i] += topk

        if not normalization:
            data = [
                [None if j is None else normalize(j) for j in i] for i in tqdm(data)
            ]
        
        snapshots_retrieval_time = time.time() - snapshots_retrieval_time # record the similarity computation time
        print(f'snapshots retrieval time: {snapshots_retrieval_time}')
        print('memory usage: {:.4} GB'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))

        print("Clustering")
        gen_cluster_time = time.time()
        cf_means = {}
        cfs = {}
        if initial_labels is None:
            cids = [-1] * record_num
        else:
            cids = initial_labels
            cid2records = defaultdict(list)
            for cid, record in zip(cids, data):
                if cid >= 0:
                    cid2records[cid].append(record)
            for cid, rs in cid2records.items():
                tmp = cfs[cid] = [[j for j in i if j is not None] for i in zip(*rs)]
                cf_means[cid] = [
                    normalize(np.mean(t, axis=0)) if len(t) else None for t in tmp
                ]
        
        sim_computation_num = 0
        sim_calc_time = 0
        statistics_update_time = 0
        for i, (record, topk) in enumerate(zip(tqdm(data), topks)):
            start_sim_calc_time = time.time()
            if cids[i] >= 0:
                continue
            cs = {cids[idx] for idx in topk if cids[idx] >= 0}
            best_cid = -1
            best_sim = -1
            for c in cs:
                w_total = 0
                sim = 0
                for w, a, b in zip(weights, record, cf_means[c]):
                    if a is not None and b is not None:
                        sim += a @ b * w
                        w_total += w
                if w_total <= 0:
                    continue
                sim /= w_total
                if sim > best_sim:
                    best_sim = sim
                    best_cid = c
            
            sim_computation_num += len(cs)
            sim_calc_time = time.time() - start_sim_calc_time + sim_calc_time
            start_statistics_update_time = time.time()

            if best_cid >= 0 and best_sim >= similarity_threshold:
                cids[i] = best_cid
                cf = cfs[best_cid]
                cf_mean = cf_means[best_cid]
                for j, k in enumerate(record):
                    if k is not None:
                        if cf[j]:
                            cf[j].append(k)
                            cf_mean[j] = normalize(np.mean(cf[j], axis=0))
                        else:
                            cf[j] = [k]
                            cf_mean[j] = k
            else:
                cid = len(cf_means)
                cids[i] = cid
                cf_means[cid] = record
                cfs[cid] = [[] if j is None else [j] for j in record]

            statistics_update_time = time.time() - start_statistics_update_time + statistics_update_time

        gen_cluster_time = time.time() - gen_cluster_time

        print(f'snapshots retrieval time: {snapshots_retrieval_time}')
        print(f'cluster generation total consuming time: {gen_cluster_time}')
        print(f'similarity computation total consuming time: {sim_calc_time}')
        print(f'cluster statistics update total consuming time: {statistics_update_time}')
        print(f'similarity computation total times: {sim_computation_num}')
        print(f'the number of snapshots is: {len(data)}')
        print('Done!')

        return cids

