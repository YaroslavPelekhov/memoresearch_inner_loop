# Data preparation

## Intro

В данной директории содержатся скрипты для обработки данных, их токенизации и сохранение в `StreamingDataset` формате, который используется во время обучения. Для удобства были добавлены скрипты для сабмита задач в Cloud и Slurm. Оригинальное описание представленов разделе [MosaicML intro](#mosaicml-intro) и далее.

## Data preparation submit scripts

### Cloud (client_lib)

Для запуска задачи токенизации на Cloud c помощью `client_lib` необходимо воспользоваться скриптом `submit_tokenization_job.py`, в который необходимо подать конфиг токенизации `tokenization_config.json` (его описание представлено ниже). В рамках сабмита будет подготовлен файл с заданиями токеназации (пары `(input_path, output_path)`) для каждого указанного в конфиге датасета, а затем запущено указанное колчиество задач (executors).

#### Config

Конфиг выглядит следующим образом.

- `base_image` -- путь или название образа, который обудет использоваться для задачи замера (используется только при запуске в Cloud, для Slurm задаётся в скрипте запуска),
- `base_input_path` и `base_output_path` -- пути до базовых директорий с данными, в которых лежат источники или будут находиться результаты токенизации соответственно. Предеполагается, что `base_input_path` содержит датасеты, описанные в разделе `datasets` конфига, а `base_output_path` будет содержать их же.
- `tokenizer_path` -- путь до токенизатора,
- `max_tokenized_len` -- длина сэмпла токенизации,
- `n_gpus` -- количество ГПУ на одной ноде (executor) задачи (используется только для запуска на Cloud),
- `n_proc` -- количество процессов токенизации на одной ноде задачи (используется только для запуска в Slurm),
- `datasets` -- содержит все датасеты и конфиги для них, которые необходимо обработать. Только указанные датасеты будут обработаны, остальные, присутствующие в `base_input_path`, будут игнорированны,
  - `script_path` -- имя или путь до файла, который будет использован для токенизации (один из `covert_dataset_hf_local.py`, `covert_dataset_json.py`, `covert_dataset_parquet.py`),
  - `script_additional_params` -- дополнительные параметры для скрипта обработки, которые не пересылаются из скрипта сабмита задачи. Для уточнения параметров необходимо смотреть help-message скрипта обработки.
  - `per_file_task_split` -- флаг, сигнализирующий, стоит ли выполнять токенизацию по-файлово или по директориям. Применим только к JSONL или Parquet датасетам, для HF датасетов необходимо выставить в `false`.
  - `num_executors` -- количество обработчиков для датасета. Каждый обработчик (нода) будет токенизировать свою часть датасета.

> Важно использовать `convert_dataset_hf_local.py` для токенизации датасета в HF (arrow) формате, т.к. оригинальный скрипт `convert_dataset_hf.py` предполагает только скачивание и обработку `c4` и `The Pile`!

```json
{
  "base_image": "image_name_or_path",
  "base_input_path": "/path/to/datasets/source/directory",
  "base_output_path": "/path/to/datasets/tokenized/directory",
  "tokenizer_path": "/path/to/tokenizer",
  "max_tokenized_len": 8192,
  "n_gpus": 1,
  "n_proc": 32,
  "datasets": {
    "dataset1": {
      "script_path": "/path/to/script1.py",
      "script_additional_params": "tokens_dtype=uint16",
      "per_file_task_split": true,
      "num_executors": 1
    },
    "dataset2": {
      "script_path": "convert_script2.py",
      "script_additional_params": "tokens_dtype=uint32",
      "per_file_task_split": false,
      "num_executors": 1
    }
  }
}
```

В директории `configs/` представлены шаблоны конфигов для номерных сетов, в которых нужно лишь указать соответствующие пути и, при необходимости, поменять параметры. Текущие сеты уже указаны.

### Slurm

Для запуска задачи токенизации через Slurm необходимо воспользоваться скриптом `submit_slurm_tokenization.py`, в который необходимо подать конфиг токенизации `tokenization_config.json` (такой же, что и для Cloud). Также необходимо поправить `slurm-sbatch-tokenization-start.sh`, который используется для непосредственно запуска slurm задачи через команду `sbatch`. Необходимо указать рабочию директорию (`WORKDIR`) и обновить её для логов, путь до образа (`CONTAINER_IMAGE`), на котором будет запускаться токенизация, путь до NFS маунтов для контейнера (`MOUNTS`).

### Untar

Для запуска разархивации шардов в Slurm задаче, сжатых в `.tar.gz` для трансфера между машинами, можно воспользоваться скриптом `submit_slurm_untar.py`, в который необходимо подать конфиг `untar_config.json`. Также необходимо поправить `slurm-sbatch-untar.sh`, задав аналогичные пути и параметры, как и для токенизации. Для запуска вне задачи можно воспользоваться скриптом `uncompress_dataset.sh`.

При необхоидмости нужно обновить исполняемый на нодах файл `slurm-untar-startup.sh`, обновив переменную `PATH`, чтобы она включала в себя путь до директории с утилитой   `parallel`, используемой для распределения обработки, а также обновить функцию `uncompress_file()`, выбрав необходимый уровень вложенности. Уровень задаётся через параметр `--strip-components=N` и по-умолчанию выставлен в 6. Значение обусловлено на то, что сжатые данные хранят абсолютные пути и для сохранения первоначальной структуры директорий их необходимо обрезать. Значение 6 предполагает полную обработку номерного сета (`3.5`, `4.0`, `test_data`). Если же планируется обработать только один конкретный датасет (`3.5/wikipedia`, `test_data/set_name`), то необходимо раскоментировать следующие строки, где значение равно 7. Если используется большая вложенность, обновить соответствюуще.

> **Важно!** Необходимо перепроверить выставленные параметры в конфиге и скриптах запуска, чтобы конечная структура директорий соответствовала ожидаемой. При необходимости протестировать на небольшом сабсете данных.

#### Config

Конфиг для разархивации выглядит следующим образом.

- `source_dir` и `dest_dir` это пути до директорий с исходными сжатыми данными и разжатыми соответственно,
- `task_name` задаёт имя JSON файла с заданиями по расжатию (пары `(input_file, output_file)`), которые затем будут использованы для параллельной обработки,
- `n_proc` -- количество процессов, использующихся для обработки данных,
- `num_executors` -- количество обработчиков, каждый из которых обрабатывает свою часть данных.

```json
{
  "source_dir": "/path/to/source_dir",
  "dest_dir": "/path/to/dest_dir",
  "task_name": "name_of_task_file",
  "n_proc": 1,
  "num_executors": 1
}
```

### Post-processing

После завершения (или зависания) токенизации можно дополнительно убедиться в том, что все шарды обработались без явных проблем, запустив скрипт `check_tokenization_result.py`. Скрипт принимает два аргумента - путь до директории с исходным (нетокенизированным) датасетом, и путь до директории с токенизированными данными, и проверяет
1. Что для каждого шарда в исходных данных имеется токенизированная версия
2. Что соотношение размеров до/после токенизации для всех шардов примерно совпадает. Если для какого-то шарда размер токенизированной версии аномально низкий, это может свидетельствовать о проблемах с его обработкой.

Для пост-обработки, а именно мержа шардов в одну конечную папку, необходимо воспользоваться скриптом `merge_shards.py`, в который подать конфиг `merge_config.json` с указанием всех необходимых полей.

> Рекомендуется сначала выполнить запуск с параметром `--dry-run` для предварительной проверки, что все указанные датасеты были корректно сохранены (содержат `index.json` файлы).

#### Config

Конфиг выглядит следующим образом.

- `base_dir` -- путь до директории с датасетами (`base_output_path` в конфиге токенизации),
- `one_level_dirs`, `two_level_dirs`, `three_level_dirs` -- содержат названия датасетов (раздел `datasets` конфига токенизации), которые необходимо смержить, с соотвествующим уровнем глубины. Уровень глубины определяется по тому, сколько раз нужно переместиться по внутренним директориям датасета прежде, чем будут окрыты шарды.

```json
{
    "base_dir": "/path/to/tokenized/data",
    "one_level_dirs": [
        "dataset1"
    ],
    "two_level_dirs": [
        "dataset2"
    ],
    "three_level_dirs": [
        "dataset3"
    ]
}
```

## Important details

### Tokenization tasks

В ходе работы скриптов сабмита происходит сканирование папки с исходными данными и всех её подпапок и создаётся JSON файл с "заданиями" в подпапке `tasks/` -- разбиением сканированных папок на чанки пар `(source_path, output_path)`. Его название составляется из названия датасета, таргетной длины токенизации и используемого токенизатора. Файл формируется заново при каждом запуске, что позволяет изменять количество нод для обработки (`num_executors`) и перераспределять обработку между ними.

Также в рамках сабмита идёт проверка на то, был ли датасет уже обработан и смержен (в директории датасета присутствуют только файлы, в том числе `index.json`). Если он уже был смёржен, данный датасет пропускается, для него не происходит сабмита задачи.

При начале работы каждого исполнителя (executor'а) происходит фильтрация заданий токенизации для определения возможных уже обработанных частей. Обработанные части определяются как уже существующие папки с шардами и `index.json` файлом. Таким образом, возможен повторный запуск с таким же конфигом для продолжения токенизации, если она была прервана.

> Рекомендуется повторно запустить токенизацию после её завершения, чтобы убедиться, что все данные были корректно обработаны!

### Untar

Для организации параллельной обработки шардов используется утилита `parallel`. Если её нет, то её необходимо скачать и установить ([ссылка]()), а затем обновить переменную `PATH` в startup скрипте для запуска в Slurm (либо же в текущей shell сессии, если используется локальный запуск), чтобы исполняемый бинарник был доступен.

Устанавливать рекомендуется в user режиме (подробнее можно почитать в README в скачанной папке). Тогда результат будет доступен в указанной папке под `bin/` и `share/`, которые затем можно переместить в `local/` или `.local/`.

## MosaicML intro

This folder contains scripts for converting text data from original sources (HF, JSON) to the Mosaic [StreamingDataset](https://github.com/mosaicml/streaming) format for consumption by our training scripts. StreamingDataset is designed to make training on large datasets from cloud storage as fast, cheap, and scalable as possible. In particular, it is custom built for multi-node, distributed training for large models while maximizing correctness guarantees, performance, and ease of use.

They following scripts will run on CPUs (no GPUs needed). Execute them in an environment with `python` and `llm-foundry` dependencies installed. All scripts should run from the `./llm-foundry/scripts/data_prep` directory.

## Converting a pretraining dataset
### HuggingFace data
In this example, we use the `convert_dataset_hf.py` script to convert a HuggingFace `c4` dataset into a `StreamingDataset`, using the `EleutherAI/gpt-neox-20b` tokenizer. The resulting directory is saved at `./llm-foundry/scripts/data_prep/my-copy-c4`.

Currently supports `c4` and `The Pile`.

<!--pytest.mark.skip-->
```bash
# Convert C4 dataset to StreamingDataset format
python convert_dataset_hf.py \
  --dataset c4 --data_subset en \
  --out_root my-copy-c4 --splits train_small val_small \
  --concat_tokens 2048 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>' \
  --compression zstd
```
### JSON data

Using the `convert_dataset_json.py` script...

<!--pytest.mark.skip-->
```bash
# Convert json dataset to StreamingDataset format
python convert_dataset_json.py \
  --path ./example_data/arxiv.jsonl \
  --out_root my-copy-arxiv --split train \
  --concat_tokens 2048 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>' \
  --compression zstd
```

Where `--path` can be a single json file, or a folder containing json files. `--split` denotes the intended split (hf defaults to `train`).

## Converting a finetuning dataset
Using the `convert_finetuning_dataset.py` script you can run a command such as:
<!--pytest.mark.skip-->
```bash
python convert_finetuning_dataset.py --dataset "Muennighoff/P3" \
--splits "train" "validation" \
--preprocessor "llmfoundry.data.finetuning.tasks:p3_preprocessing_function"\
 --out_root "/path/to/your/output_directory"
```

This example assumes:

- `"Muennighoff/P3"` is the dataset you want to convert. Substitute "Muennighoff/P3" with the name or path of your dataset.
- `train` and `validation` are the splits of the dataset to convert.
- `llmfoundry.data.finetuning.tasks:p3_preprocessing_function` is a string that provides the name or import path of the function used to preprocess the dataset. Substitute it with your actual preprocessor. See [tasks](https://github.com/mosaicml/llm-foundry/blob/main/llmfoundry/data/finetuning/tasks.py) for available functions and examples.
- `s3://<bucket>/muennighoff-p3` is the root path of your output directory where MDS shards will be stored. Replace this with the actual path to your output directory.

Please note that you need to fill in actual values for "your_preprocessing_function" and "/path/to/your/output_directory" in the command above for it to work correctly.

Also, if you want to keep a local copy of the output when `out_root` is remote, you can use the `--local` argument:
<!--pytest.mark.skip-->
```bash
python convert_finetuning_dataset.py --dataset "squad" --splits "train" "validation" --preprocessor "your_preprocessing_function" --out_root "s3://your_bucket/output_directory" --local "/path/to/local/directory"
```

Remember that all these command line arguments should be filled with your actual dataset name/path, preprocessing function, and output directories.
