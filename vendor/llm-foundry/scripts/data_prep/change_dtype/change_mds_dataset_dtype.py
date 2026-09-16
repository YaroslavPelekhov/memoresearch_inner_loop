import argparse
import os
import json
from pathlib import Path

import numpy as np
from typing import Dict, List, Generator, Any, Optional

from tqdm import tqdm

from streaming.base.local import LocalDataset
from streaming.base.format.mds.writer import MDSWriter

from llmfoundry.data.data import DTYPE2ENCODING_MAPPING_DICT


def initialize_reader(dirname: str, split: Optional[str]) -> LocalDataset:
    dataset_reader = LocalDataset(dirname, split)
    # print(f"{len(dataset_reader)=}")
    return dataset_reader

def sample_iterator(reader: LocalDataset) -> Generator[Dict[str, Any], None, None]:
    for i in range(len(reader)):
        sample = reader.get_item(i)
        if i % 20_000 == 0:
            print(f"done percentage: {100*i/len(reader)} dataset_dir={reader.local}")
        yield sample

def write_mds_file(samples: Generator[Dict[str, Any], None, None], out: str, dtype_encoding: str, column_names: List[str], column_encodings: List[str]):
    np_dtype_map = {'uint8': np.uint8, 'uint16': np.uint16, 'uint32': np.uint32, 'uint64': np.uint64}
    columns = {name: encoding for name, encoding in zip(column_names, column_encodings)}
    columns['dtype'] = 'int8'
    target_np_dtype = np_dtype_map[dtype_encoding]
    max_value = np.iinfo(target_np_dtype).max

    with MDSWriter(columns=columns, out=out) as writer:
        for sample in samples:
            data = np.frombuffer(sample['tokens'], dtype=np.int64)
            assert data.max() < max_value, \
                "Provided data has a max value larger than the max value of the target dtype. "

            sample['tokens'] = np.array(data, dtype=target_np_dtype).tobytes()
            sample['dtype'] = DTYPE2ENCODING_MAPPING_DICT[target_np_dtype]  # Add the new column with the specified encoding value
            writer.write(sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process MDS files.')
    parser.add_argument('--path', type=str, required=True, help='Input directory containing the MDS file')
    parser.add_argument('--split', type=str, required=False, default=None, help='Split of the MDS file')
    parser.add_argument('--out_root', type=str, required=True, help='Output directory to write the MDS file')
    parser.add_argument('--dtype-encoding', type=str, default='uint16', help='Encoding for the tokens column in the samples')

    args = parser.parse_args()

    assert not Path(args.out_root).exists(), "Output directory already exists"

    reader = initialize_reader(args.path, args.split)
    samples = sample_iterator(reader)
    write_mds_file(samples, args.out_root, args.dtype_encoding, reader.shards[0].column_names, reader.shards[0].column_encodings)
    print(f"{args.out_root} written")
