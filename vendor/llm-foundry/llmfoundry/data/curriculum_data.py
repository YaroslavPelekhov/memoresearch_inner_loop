from copy import deepcopy
import json
import logging
from omegaconf import DictConfig
import os
from pathlib import Path
from typing import Dict, Set, Any

from composer.utils import format_name_with_dist


logger = logging.getLogger(__name__)


def get_curriculum_dataset_stream(cfg: DictConfig) -> Dict[str, Any]:
    save_folder = cfg.get("save_folder", None)
    assert save_folder is not None, "fill save_folder in train config"
    filled_save_directory = format_name_with_dist(save_folder, cfg.run_name)
    curriculum_dataset_path = os.path.join(filled_save_directory, "curriculum_dataset")
    return {
        "local": curriculum_dataset_path,
        "split": "mds",
        "proportion" : 1.0
    }

def update_train_loader_dataset_params(dataset_cfg: DictConfig, streams: Dict[str, Any]):
    dataset_cfg.streams = streams

    streaming_dataset_original_order_params = {
        "num_canonical_nodes" : 1,
        "partition_algo": "orig",
        "shuffle": False
    }

    for k, v in streaming_dataset_original_order_params.items():
        logger.info(f"Force update cfg.train_loader.dataset param '{k}' : {v}")
        setattr(dataset_cfg, k, v)


def create_curriculum_dataset(cfg: DictConfig) -> None:
    streams_dict = cfg.train_loader.dataset.get("streams", None)
    assert streams_dict is not None, "curriculum dataset must have streams in train_loader.dataset"
    curriculum_stream = get_curriculum_dataset_stream(cfg)

    curriculum_stream_dir = Path(curriculum_stream['local']) / curriculum_stream['split']

    os.makedirs(curriculum_stream_dir, exist_ok=True)

    master_index = {
        "shards": [],
        "version": 2,
    }

    curriculum_shard_counter = 0
    sentinel_object = object()

    files_folder_path_set: Set[Any] = {sentinel_object}
    curriculum_dataset_order = []
    for stream_name, stream in streams_dict.items():
        curriculum_dataset_order.append(stream_name)
        files_folder_path_set.add(stream.get('files_folder_path', sentinel_object))

        with open(Path(stream['local']) / stream['split'] / 'index.json', 'r') as f:
            index = json.load(f)

        for i, shard in enumerate(index['shards']):
            dst_shard_name = f"shard.{str(curriculum_shard_counter + i).zfill(5)}.mds"
            src_shard_path = Path(stream['local']) / stream['split'] / shard['raw_data']['basename']
            dst_shard_path = curriculum_stream_dir / dst_shard_name

            shard_link = deepcopy(shard)
            shard_link['raw_data']['basename'] = dst_shard_name

            master_index['shards'].append(shard_link)

            if os.environ["RANK"] == "0":
                if dst_shard_path.exists() or dst_shard_path.is_symlink():
                    dst_shard_path.unlink()
                os.symlink(src_shard_path, dst_shard_path)

        curriculum_shard_counter += len(index['shards'])
        logger.info(f"Symlinked {len(index['shards'])} shards for '{stream_name}' dataset")

    if os.environ["RANK"] == "0":
        with open(curriculum_stream_dir / 'index.json', 'w') as f:
            json.dump(master_index, f)

    logger.info(f"{' -> '.join(curriculum_dataset_order)}")
    logger.info(f"Total curriculum mds shards: {curriculum_shard_counter}")

    files_folder_path_set.remove(sentinel_object)
    assert len(files_folder_path_set) <= 1, "curriculum dataset must have same files_folder_path for all streams"
    if files_folder_path_set:
        curriculum_stream["files_folder_path"] = list(files_folder_path_set)[0]

    update_train_loader_dataset_params(cfg.train_loader.dataset, {"curriculum_stream" : curriculum_stream})
