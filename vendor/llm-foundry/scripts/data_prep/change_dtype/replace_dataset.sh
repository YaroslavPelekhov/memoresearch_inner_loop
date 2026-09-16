#!/bin/bash

set -e

OLD_DATASET_DIR=""
SHARDS_DIR=${OLD_DATASET_DIR/datasets/datasets_reduce_dtype}
REDUCED_DATASET_DIR="${SHARDS_DIR}_reduced"
OLD_DATASET_DIR_BACKUP="${SHARDS_DIR}_backup"

echo "OLD_DATASET_DIR: $OLD_DATASET_DIR"
echo "SHARDS_DIR: $SHARDS_DIR"
echo "REDUCED_DATASET_DIR: $REDUCED_DATASET_DIR"
echo "OLD_DATASET_DIR_BACKUP: $OLD_DATASET_DIR_BACKUP"

python3 merge_shards.py --input-dir ${REDUCED_DATASET_DIR}

python3 << EOF
import sys
sys.path.append('/home/jovyan/vmamedov/llm-foundry-tokenization/')
sys.path.append('/home/jovyan/vmamedov/llm-foundry-tokenization/contrib/streaming')

from streaming import LocalDataset
from streaming import StreamingDataset

import numpy as np

old_dataset = LocalDataset(local='$OLD_DATASET_DIR')
reduced_dataset = LocalDataset(local='$REDUCED_DATASET_DIR')

assert len(old_dataset) == len(reduced_dataset), 'old and reduced datasets should have same length'

old_ds_item = old_dataset.get_item(0)
reduced_ds_item = reduced_dataset.get_item(0)
old_deserialized_bytes = np.frombuffer(old_ds_item['tokens'], dtype=np.int64)

assert 'dtype' in reduced_ds_item, 'new dataset should have dtype'
assert 'dtype' not in old_ds_item, 'old dataset should not have dtype'

reduced_deserialized_bytes = np.frombuffer(reduced_ds_item['tokens'], dtype=np.uint16 if reduced_ds_item.get('dtype') is not None else -1)

assert (reduced_deserialized_bytes == old_deserialized_bytes).all(), 'old and new dataset should be the same'
EOF

rm -r ${SHARDS_DIR} || echo "shards have already been removed"

mv ${OLD_DATASET_DIR} ${OLD_DATASET_DIR_BACKUP}
mv ${REDUCED_DATASET_DIR} ${OLD_DATASET_DIR}

rm -r ${OLD_DATASET_DIR_BACKUP}