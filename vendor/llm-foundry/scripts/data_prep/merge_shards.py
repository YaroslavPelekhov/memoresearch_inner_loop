import json
import os
from argparse import ArgumentParser
from functools import partial
from glob import glob
from multiprocessing.pool import ThreadPool

from tqdm import tqdm


def parse_args():
    parser = ArgumentParser(
        description="Merge shards in specified datasets root subfolders into their root folder."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a JSON merge config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run checking if all specified subfolders have a index.json file without performing a merge.",
    )

    return parser.parse_args()


def with_id(basename: str, shard_id: int) -> str:
    """Get a new basename with the given shard_id.

    Args:
        basename (str): Old basename of file.
        shard_id (int): New shard ID.

    Returns:
        str: New basename of file.
    """
    parts = basename.split(".")
    parts[1] = f"{shard_id:08}"

    return ".".join(parts)


def merge_shard_groups(
    root: str, pattern: str = "*", dry_run: bool = False
) -> None:
    """Merge ephemeral sub-datasets created in parallel into one dataset.

    Args:
        root (str): Root directory.
        pattern (str): Glob pattern used to get all the files. Defaults to ``*``.
        dry_run (bool): Flag for performing dry run, that only checks the datasets
            that they have index files (been tokenized). Defaults to ``False``.
    """
    assert os.path.exists(root), f"Provided root path {root} does not exist."

    if os.path.exists(os.path.join(root, "index.json")):
        print(f"Dataset {root} has already been merged, skipping it.")
        return

    pattern = os.path.join(root, pattern)
    subdirs = sorted(glob(pattern))
    shard_id = 0
    infos = []

    # Check for index files in all the subfolders. If index file is not present,
    # then the part of the dataset was not processed correctly.
    is_broken_data = False
    print(f"Checking shards in {root}...")
    for subdir in tqdm(subdirs):
        index_filename = os.path.join(subdir, "index.json")
        if not os.path.exists(index_filename):
            is_broken_data = True
            print("No index file found in", subdir)

    if is_broken_data:
        raise RuntimeError(
            "There appears to be broken data parts, please fix them "
            "(re-launch tokenization for JSONL datasets or re-tokenize for HF datasets)."
        )

    if dry_run:
        return

    print(f"Merging shards in {root}...")
    for subdir in tqdm(subdirs):
        index_filename = os.path.join(subdir, "index.json")
        with open(index_filename) as fin:
            index_cfg = json.load(fin)

        for info in index_cfg["shards"]:
            old_basename = info["raw_data"]["basename"]
            new_basename = with_id(old_basename, shard_id)
            info["raw_data"]["basename"] = new_basename

            if info["zip_data"] is not None:
                old_basename = info["zip_data"]["basename"]
                new_basename = with_id(old_basename, shard_id)
                info["zip_data"]["basename"] = new_basename

            old_filename = os.path.join(subdir, old_basename)
            new_filename = os.path.join(root, new_basename)
            os.rename(old_filename, new_filename)

            shard_id += 1
            infos.append(info)

        os.remove(index_filename)
        os.rmdir(subdir)

    index_filename = os.path.join(root, "index.json")
    index_cfg = {
        "version": 2,
        "shards": infos,
    }
    text = json.dumps(index_cfg, sort_keys=True)
    with open(index_filename, "w") as out:
        out.write(text)

    print("Removing empty directories...")
    os.system(f"find {root} -type d -empty -delete")

    print("Done merging.")


def main(args):
    with open(args.config) as fin:
        config = json.load(fin)

    base_dir = config["base_dir"]
    one_level_dirs = config["one_level_dirs"]
    two_level_dirs = config["two_level_dirs"]
    three_level_dirs = config["three_level_dirs"]
    four_level_dirs = config["four_level_dirs"]

    one_level_dirs_pathes = [
        os.path.join(base_dir, dir_name) for dir_name in one_level_dirs
    ]
    two_level_dirs_pathes = [
        os.path.join(base_dir, dir_name) for dir_name in two_level_dirs
    ]
    three_level_dirs_pathes = [
        os.path.join(base_dir, dir_name) for dir_name in three_level_dirs
    ]
    four_level_dirs_pathes = [
        os.path.join(base_dir, dir_name) for dir_name in four_level_dirs
    ]

    with ThreadPool() as pool:
        pool_fn = partial(merge_shard_groups, pattern="*", dry_run=args.dry_run)
        pool.map(pool_fn, one_level_dirs_pathes, chunksize=1)

    with ThreadPool() as pool:
        pool_fn = partial(
            merge_shard_groups, pattern="*/*", dry_run=args.dry_run
        )
        pool.map(pool_fn, two_level_dirs_pathes, chunksize=1)

    with ThreadPool() as pool:
        pool_fn = partial(
            merge_shard_groups, pattern="*/*/*", dry_run=args.dry_run
        )
        pool.map(pool_fn, three_level_dirs_pathes, chunksize=1)

    with ThreadPool() as pool:
        pool_fn = partial(
            merge_shard_groups, pattern="*/*/*/*", dry_run=args.dry_run
        )
        pool.map(pool_fn, four_level_dirs_pathes, chunksize=1)


if __name__ == "__main__":
    main(parse_args())
