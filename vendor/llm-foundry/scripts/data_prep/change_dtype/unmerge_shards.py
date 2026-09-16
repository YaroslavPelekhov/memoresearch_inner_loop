import argparse
import os
import json


def split_shard_groups(input_dir: str, output_dir: str, group_size: int):
    """Split shards in one dataset into ephemeral sub-datasets.

    Args:
        input_dir (str): Input directory containing the files and index.json.
        output_dir (str): Output directory to create subdirectories and symbolic links.
        group_size (int): Number of files to group in each subdirectory.
    """
    print("splitting shards in ", input_dir)
    assert os.path.exists(
        os.path.join(input_dir, "index.json")
    ), f"dataset {input_dir} has not been re sharded"

    # Load index file
    index_filename = os.path.join(input_dir, "index.json")
    obj = json.load(open(index_filename))

    # Create subdirectories and group files
    for i in range(0, len(obj["shards"]), group_size):
        subdir = os.path.join(output_dir, f"subdir_{i//group_size:07}")
        os.makedirs(subdir, exist_ok=True)

    # Create symbolic links and update index
    for i, info in enumerate(obj["shards"]):
        old_basename = info["raw_data"]["basename"]
        new_basename = old_basename
        info["raw_data"]["basename"] = new_basename

        old_filename = os.path.join(input_dir, old_basename)
        subdir = os.path.join(output_dir, f"subdir_{i//group_size:07}")
        assert os.path.exists(subdir), f"subdirectory {subdir} does not exist: {i=} {group_size=}"
        new_filename = os.path.join(subdir, new_basename)
        os.symlink(old_filename, new_filename)

        if info["zip_data"] is not None:
            old_basename = info["zip_data"]["basename"]
            new_basename = old_basename
            info["zip_data"]["basename"] = new_basename

            old_filename = os.path.join(input_dir, old_basename)
            new_filename = os.path.join(subdir, new_basename)
            os.symlink(old_filename, new_filename)

    # Write new index files in subdirectories
    for i in range(0, len(obj["shards"]), group_size):
        subdir = os.path.join(output_dir, f"subdir_{i//group_size:07}")
        assert os.path.exists(subdir), f"subdirectory {subdir} does not exist"
        index_filename = os.path.join(subdir, "index.json")
        files_in_subdir = len(os.listdir(subdir))
        assert files_in_subdir > 0, f"subdirectory {subdir} is empty"
        with open(index_filename, "w") as out:
            if (i + group_size) > len(obj["shards"]):
                shards = obj["shards"][i:]
            else:
                shards = obj["shards"][i:i + group_size]
            assert len(shards) == files_in_subdir, (len(shards), group_size, i, len(obj["shards"]), files_in_subdir)
            json.dump({"version": 2, "shards": shards}, out, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", "-i", type=str)
    parser.add_argument("--output-dir", "-o", type=str)
    parser.add_argument("--group-size", type=int, default=40)
    args = parser.parse_args()

    split_shard_groups(args.input_dir, args.output_dir, args.group_size)
