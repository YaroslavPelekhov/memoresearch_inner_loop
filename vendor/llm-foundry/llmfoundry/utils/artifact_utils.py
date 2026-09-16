import os
import yaml
import argparse
from importlib import metadata
import shutil
from omegaconf import OmegaConf as om


def copy_artifacts(cfg, artifact_folder, is_dummy=False):
    # Creating artifacts folder
    os.makedirs(artifact_folder, exist_ok=True)

    # Copy config
    config_path = os.path.join(artifact_folder, "llm_foundry_config.yaml")
    with open(config_path, "w") as f:
        om.save(config=cfg, f=f)

    # Copy data meta
    if not is_dummy:
        for data_stream in cfg["train_loader"]["dataset"]["streams"]:
            dataset_stream = cfg["train_loader"]["dataset"]["streams"][data_stream]
            if dataset_stream.get("split", None) is None:
                continue

            split_name = (
                os.path.join(dataset_stream["local"], dataset_stream["split"])
                if "split" in dataset_stream
                else "/"
            )

            folders = []
            for sn in [dataset_stream['local'], split_name]:
                folders.extend([
                    os.path.join(sn, x) \
                    for x in ['configs', 'prod_config', 'prod_meta'] \
                    if os.path.exists(os.path.join(sn, x))
                ])

            dataset_name = os.path.basename(dataset_stream["local"])

            for folder_path in folders:
                folder_name = folder_path.strip("/").split("/")[-1]
                shutil.copytree(
                    folder_path,
                    os.path.join(
                        artifact_folder, "datasets", dataset_name, folder_name
                    ),
                    dirs_exist_ok=True,
                )

    # Copy tokenizer
    if not is_dummy:
        tokenizer_file_names = [
            "added_tokens.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "tokenizer.json",
        ]
        tokenizer_files = [
            os.path.join(cfg["tokenizer"]["name"], x) for x in tokenizer_file_names
        ]

        tokenizer_path = os.path.join(artifact_folder, "tokenizer")
        os.makedirs(tokenizer_path, exist_ok=True)
        for tok_file in tokenizer_files:
            file_name = os.path.basename(tok_file)
            file_destination = os.path.join(tokenizer_path, file_name)
            if os.path.exists(tok_file):
                shutil.copyfile(tok_file, file_destination)

    # Copy packages versions
    packages_destination = os.path.join(artifact_folder, "package_versions.yaml")
    dists = metadata.distributions()
    versions = {}
    for dist in dists:
        name = dist.metadata["Name"]
        version = dist.version
        versions[name] = version

    with open(packages_destination, "w") as f:
        yaml.dump(versions, f, allow_unicode=True, sort_keys=False)
