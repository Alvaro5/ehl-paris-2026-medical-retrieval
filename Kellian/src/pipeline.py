from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

from .config import PipelineConfig
from .features import feature_matrix, mind_descriptor, resample_to_grid
from .geometry import geometry_distance, geometry_signature
from .io_utils import load_volume, read_table, resolve_image
from .registration import affine_register_target_to_query
from .rerank import cosine_scores, csls, normalized_mutual_information, rankings_from_scores, reciprocal_rank_fusion, sinkhorn


@dataclass(frozen=True)
class Pool:
    dataset: str
    split: str
    queries: pd.DataFrame
    gallery: pd.DataFrame


def load_pools(config: PipelineConfig) -> list[Pool]:
    pools: list[Pool] = []
    for dataset in ("dataset1", "dataset2", "dataset3"):
        for split in ("val", "test"):
            root = config.data_root / dataset
            pools.append(Pool(dataset, split, read_table(root / f"{split}_queries.csv"), read_table(root / f"{split}_gallery.csv")))
    return pools


def rank_descriptor_pool(pool: Pool, config: PipelineConfig) -> list[dict[str, str]]:
    query_ids, query_features = feature_matrix(
        pool.queries, "query_id", "query_image", config.data_root, config.cache_dir / pool.dataset, config.grid, f"{pool.dataset}/{pool.split} query"
    )
    target_ids, target_features = feature_matrix(
        pool.gallery, "target_id", "target_image", config.data_root, config.cache_dir / pool.dataset, config.grid, f"{pool.dataset}/{pool.split} target"
    )
    scores = sinkhorn(csls(cosine_scores(query_features, target_features), config.csls_k), config.sinkhorn_tau, config.sinkhorn_iters)
    rankings = rankings_from_scores(scores, target_ids)
    return [{"query_id": query_id, "target_id_ranking": " ".join(ranking)} for query_id, ranking in zip(query_ids, rankings, strict=True)]


def rank_geometry_pool(pool: Pool, config: PipelineConfig) -> list[dict[str, str]]:
    target_sigs = {
        str(row.target_id): geometry_signature(resolve_image(config.data_root, str(row.target_image)))
        for row in pool.gallery.itertuples(index=False)
    }
    rows: list[dict[str, str]] = []
    for query_row in tqdm(pool.queries.itertuples(index=False), total=len(pool.queries), desc=f"{pool.dataset}/{pool.split} geometry"):
        query_sig = geometry_signature(resolve_image(config.data_root, str(query_row.query_image)))
        ranking = sorted(target_sigs, key=lambda target_id: geometry_distance(query_sig, target_sigs[target_id]))
        rows.append({"query_id": str(query_row.query_id), "target_id_ranking": " ".join(ranking)})
    return rows


def rank_dataset2_pool(pool: Pool, config: PipelineConfig) -> list[dict[str, str]]:
    query_ids, query_features = feature_matrix(
        pool.queries, "query_id", "query_image", config.data_root, config.cache_dir / "dataset2_prefilter", config.grid, f"{pool.dataset}/{pool.split} query prefilter"
    )
    target_ids, target_features = feature_matrix(
        pool.gallery, "target_id", "target_image", config.data_root, config.cache_dir / "dataset2_prefilter", config.grid, f"{pool.dataset}/{pool.split} target prefilter"
    )
    prefilter_scores = sinkhorn(csls(cosine_scores(query_features, target_features), config.csls_k), config.sinkhorn_tau, config.sinkhorn_iters)
    prefilter_rankings = rankings_from_scores(prefilter_scores, target_ids)
    target_by_id = {str(row.target_id): row for row in pool.gallery.itertuples(index=False)}

    rows: list[dict[str, str]] = []
    for q_index, query_row in enumerate(
        tqdm(pool.queries.itertuples(index=False), total=len(pool.queries), desc=f"{pool.dataset}/{pool.split} affine rerank")
    ):
        query_id = str(query_row.query_id)
        query_path = resolve_image(config.data_root, str(query_row.query_image))
        query_lowres = resample_to_grid(load_volume(query_path), config.grid)
        query_desc = mind_descriptor(query_lowres)
        shortlist = prefilter_rankings[q_index][: config.dataset2_topk]

        mind_scores: dict[str, float] = {}
        nmi_scores: dict[str, float] = {}
        for target_id in shortlist:
            target_row = target_by_id[target_id]
            target_path = resolve_image(config.data_root, str(target_row.target_image))
            if config.skip_registration:
                registered = resample_to_grid(load_volume(target_path), config.grid)
            else:
                registered = resample_to_grid(affine_register_target_to_query(query_path, target_path), config.grid)
            mind_scores[target_id] = float(query_desc @ mind_descriptor(registered))
            nmi_scores[target_id] = normalized_mutual_information(query_lowres, registered)

        mind_ranking = sorted(shortlist, key=lambda target_id: -mind_scores[target_id])
        nmi_ranking = sorted(shortlist, key=lambda target_id: -nmi_scores[target_id])
        fused = reciprocal_rank_fusion([prefilter_rankings[q_index], mind_ranking, nmi_ranking], [1.0, 2.0, 2.0])
        reranked = sorted(shortlist, key=lambda target_id: -fused[target_id])
        tail = [target_id for target_id in prefilter_rankings[q_index] if target_id not in set(reranked)]
        rows.append({"query_id": query_id, "target_id_ranking": " ".join(reranked + tail)})
    return rows


def build_submission(config: PipelineConfig) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for pool in load_pools(config):
        if pool.dataset == "dataset3":
            rows.extend(rank_geometry_pool(pool, config))
        elif pool.dataset == "dataset2":
            rows.extend(rank_dataset2_pool(pool, config))
        else:
            rows.extend(rank_descriptor_pool(pool, config))
    return rows
