import json
import os
from argparse import ArgumentParser
from glob import glob

from tqdm import tqdm


def parse_args():
    parser = ArgumentParser(
        description="Merge shards in specified datasets root subfolders into their root folder."
    )
    parser.add_argument('--input-dir', '-i', type=str, required=True, help='Input folder.')
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
    parts[1] = f"{shard_id:07}"
    return ".".join(parts)


def merge_shard_groups(root: str, pattern: str = "*", dry_run: bool = False) -> None:
    """Merge ephemeral sub-datasets created in parallel into one dataset.

    Args:
        root (str): Root directory.
    """
    print("merging shards in ", root)
    assert not os.path.exists(
        os.path.join(root, "index.json")
    ), f"dataset {root} has already been re sharded"
    pattern = os.path.join(root, pattern)
    subdirs = sorted(glob(pattern))
    shard_id = 0
    infos = []

    # Check for index files in all the subfolders. If index file is not present,
    # then the part of the dataset was not processed correctly.
    is_broken_data = False
    for subdir in tqdm(subdirs):
        index_filename = os.path.join(subdir, "index.json")
        if not os.path.exists(index_filename):
            is_broken_data = True
            print("no index file in", subdir)
    assert (
        not is_broken_data
    ), "There appears to be broken data parts, please fix using --repair flag during tokenization."
    if dry_run:
        return

    for subdir in tqdm(subdirs):
        index_filename = os.path.join(subdir, "index.json")
        if not os.path.exists(index_filename):
            print("no index file in", subdir)

        obj = json.load(open(index_filename))
        for info in obj["shards"]:
            old_basename = info["raw_data"]["basename"]
            new_basename = with_id(old_basename, shard_id)
            info["raw_data"]["basename"] = new_basename

            if info["zip_data"] is not None:
                old_basename = info["zip_data"]["basename"]
                new_basename = with_id(old_basename, shard_id)
                info["zip_data"]["basename"] = new_basename

            old_filename = os.path.join(subdir, old_basename)
            new_filename = os.path.join(root, new_basename)
            assert not os.rename(old_filename, new_filename)

            shard_id += 1
            infos.append(info)

        assert not os.remove(index_filename)
        assert not os.rmdir(subdir)

    index_filename = os.path.join(root, "index.json")
    obj = {
        "version": 2,
        "shards": infos,
    }
    text = json.dumps(obj, sort_keys=True)
    with open(index_filename, "w") as out:
        out.write(text)


def main(args):
    merge_shard_groups(args.input_dir, pattern="*", dry_run=args.dry_run)


if __name__ == "__main__":
    main(parse_args())
