# Sber LLM Foundry

## Полезные ссылки
1) [Чат Mattermost с новостями](https://mm.sberdevices.ru/sberdevices/channels/gigachat-ml-infra-news)
2) [Чат Mattermost для вопросов по LLM Foundry](https://mm.sberdevices.ru/sberdevices/channels/gigachat-ml-infra-bugs)
3) [Confluence](https://confluence.sberdevices.ru/pages/viewpage.action?spaceKey=GigaChatML&title=GigaChat+ML)
4) [Ссылка на F.A.Q. по LLM Foundry](https://confluence.sberdevices.ru/display/GigaChatML/F.A.Q+llm-foundry)


## Intro

Для работы были модифицированы некоторые оригинальные файлы репозитория, а также добавлены дополнительные папки.

* `llmfoundry/` - исходный код для моделей, наборов данных, коллбеков, утилит и т.д.
* `contrib/`
    * [`composer`](https://gitlab.ai.cloud.ru/gigallms/composer)
    * [`gigachat-utils`](https://gitlab.ai.cloud.ru/gigallms/gigachat-utils)
    * [`gigafsdp`](https://gitlab.ai.cloud.ru/gigallms/gigafsdp)
    * [`streaming`](https://gitlab.ai.cloud.ru/gigallms/streaming)
* `sber_benchmarks/` - различные бенчмарки
* `sber_configs/` - конфиги, которые используются для обучения моделей
* `sber_docker/` - содержит файлы, необходимые для сборки образов, использующихся для обучения
* `sber_notebooks/` - некоторые релевантные ноутбуки.
* [`sber_scripts/`](sber_scripts) - скрипты, в том числе для запуска задач обучения и конвертации моделей в HuggingFace формат
* [`sber_tests/`](sber_tests) - интеграционные тесты
* `scripts/`
  * [`data_prep/`](scripts/data_prep) - преобразование текстовых данных из оригинальных источников в формат StreamingDataset
  * `train/train.py`

## Установка

```bash
git clone https://gitlab.ai.cloud.ru/gigallms/llm-foundry.git --recursive
```

**ВАЖНО!**  Для стабильного функционирования LLM Foundry регулярно обновляйте свой локальный репозиторий и **контрибы**. Контрибы обновляются с помощью следующей команды:

```bash
git pull --recurse-submodules && git submodule update --init --recursive
```

## ⚙️ Настройка pre-commit

Один раз после клонирования репозитория:

```bash
pip install pre-commit==4.3.0 ruff==0.14.4
pre-commit install
```

## LLM Foundry задачи

В LLM Foundry есть 2 типа задач: Pytorch2 и Infinite. Pytorch2 задача обычно применяется для какого-то длительного процесса, не требующего вмешательства в код задачи, например, для обучения модели. Infinite задача удобна для дебага и разработки.

| Задача | Плюсы  | Минусы  |
|:-------------:|:-------------:|:---------------:|
| Pytorch2 | User-friendly запуски         | Задача по 5-7 минут может висеть в pending, не удобно для дебага и разработки     |
| Infinite | До нуля сокращается ожидание запуска скрипта | Шаблон запуска периодически меняется, надо обращаться в cloud и обновлять скрипты **infinite_jobs/run.sh** и **infinite_jobs/clean.sh** |

## Запуск задач

Для запуска обоих типов задач используется скрипт `sber_scripts/submit_template.py`. Настройки задачи и окружения конфигурируются через указание параметров в этом скрипте.

- **SUBMISSION_TYPE**: Указывает тип системы, на которую будет отправлена задача. Возможные значения — `"slurm"` или `"cloud"`.
- **REGION**: Укажите регион, в который будет отправлена задача. Например, для SLURM это может быть `o3-64`, `o3-128` или `o4`, для cloud — `SR004`, `SR005` или `SR008`.
- **USER_LOGIN**: Ваш логин. Например, `'fancylogin'`.
- **EXP_NAME**: Название эксперимента. Например, `'MyFancyModel'`.
- **CONFIG_NAME**: Конфигурационный файл эксперимента. Например, `'example.yaml'`.
- **MAX_RETRY**: Максимальное количество попыток перезапуска задачи.
- **N_GPUS и N_WORKERS**: Количество используемых GPU и работников (узлов) соответственно.
- **BASE_IMAGE**: Укажите базовый образ, если не используете значение по умолчанию.
- **SCRIPT_TYPE**: Укажите `"pytorch2"` для запуска PyTorch экспериментов или `"infinity_job"` для inf задачи.
- **SCRIPT_ARGS**: Дополнительные аргументы, передаваемые в скрипт `train.py`.

Настройка параметров обучения осуществляется через отдельный конфигурационный `yaml` файл. В качестве примера можно взять дефолтный конфигурационный файл `sber_configs/default_gigar_config.yaml`. Для запуска тестовой задачи в нем достаточно заполнить обязательные поля `run_name`, `exp_name`, `user`.

Подробнее про задачи можно прочитать [здесь](sber_scripts/).

## MosaicML Docker Images

| Docker Image                                                                                                | Torch Version | Cuda Version | LLM Foundry dependencies installed? |
| ----------------------------------------------------------------------------------------------------------- | ------------- | ------------ | ----------------------------------- |
| `cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/llm-foundry-train:2025_02_03_commit277c3722`           | 2.6           | 12.6         | Yes                                 |
| `cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/llm-foundry-train:2024_12_20_commit3668d9d7`           | 2.5.1         | 12.4         | Yes                                 |
| `cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/llm-foundry-train:2024_09_12_commit47444564`           | 2.4           | 11.8         | Yes                                 |
| `cr.ai.cloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/llm-foundry-train:v000_cuda121_torch211_nccl2205_hpcx` | 2.1.1         | 11.7         | Yes                                 |

## GigaFSDP

В LLM Foundry используется технология GigaFSDP, которая является альтернативой FSDP. GigaFSDP позволяет использовать частичную заморозку весов, обучение LoRA, обучение с tp_size > 1 и многое другое. Подробнее можно прочитать [тут](https://gitlab.ai.cloud.ru/gigallms/gigafsdp).

## Чекпоинты и мапперы

При работе в LLM Foundry мы используем три вида чекпоинтов моделей:
1) Hugging Face -- обычный hugging face чекпоинт #/your/path/model-hf/
2) FSDP -- монолитный чекпоинт, содержащий всю информацию о модели #/your/path/model-rank0.pt
3) [GigaFSDP](https://gitlab.ai.cloud.ru/gigallms/gigafsdp) -- папка с файлами расширения `.distcp`. Шардированный чекпоинт, в нем хранятся только веса.  #/your/path/model/

Для запуска train скрипта используются только fsdp и gigafsdp чекпоинты. Для старта эксперимента с нуля необходимо иметь монолитный fsdp чекпоинт, так как в нем содержится вся информация о модели. Поэтому если у вас есть gigafsdp или hf чекпоинт и вам нужно изменить параметры обучения (архитектуру модели, количество карт и т.д.), то необходимо сконвертировать имеющийся чекпоинт в монолитный.

Для того, чтобы конвертировать один тип чекпоинта в другой, у нас есть мапперы. Как пользоваться мапперами можно почитать [тут](https://gitlab.ai.cloud.ru/gigallms/gigachat-utils/-/tree/master/convertation?ref_type=heads).

## Сохранение моделей и артефакты обучения

Сохранение чекпоинтов во время обучения модели производится в папку `save_folder`, определенную в `yaml` конфиге обучения. Структура у этой папки следующая:

```
checkpoints/${exp_name}/${run_name}
|____artifacts
|____tmp
|____latest-rank0.pt _________
|____ep0_ba2000               |
|____ep0_ba6000               | симлинк
|____ep0_ba7600               |
| |____ep0-ba7600-tp_rank0 <--
| | |______3_0.distcp
| | |______0_0.distcp
| | |______2_0.distcp
| | |______1_0.distcp
| |____artifacts
| | |____llm_foundry_config.yaml
| | |____tokenizer
| | | |____special_tokens_map.json
| | | |____tokenizer_config.json
| | |____package_versions.yaml
```
В папке рана хранятся следующие сущности: папки с чекпоинтами батчей `ep0_ba*`, папка с артефактами `artifacts`, в которой хранятся конфиги текущего запуска, и симлинк `latest-rank0.pt`, который указывает на последний чекпоинт. Папка с батчем состоит из папки с весами модели (в данном случае GigaFSDP с 4 шардами) и копия папки `artifacts`.

Если `tp_size > 1`, количество папок с суффиксом `rank0` равно `tp_size`. Для `tp_size = 2` выглядеть это будет следующим образом:

```
checkpoints/${exp_name}/${run_name}
|____artifacts
|____tmp
|____latest-rank1.pt _____________________
|____latest-rank0.pt _________            |
|____ep0_ba2000               |           |
|____ep0_ba6000               | симлинки  |
|____ep0_ba7600               |           |
| |____ep0-ba7600-tp_rank0 <--            |
| | |...                                  |
| |____ep0-ba7600-tp_rank1 <--------------
| | |...
| |____artifacts
```

При повторных запусках обучения скрипт `train.py` будет подтягивать чекпоинт из `latest-rank0.pt`.

**Будьте осторожны!** Если вы проводите новый эксперимент с новыми параметрами, но сохраняете старое название эксперимента, убедитесь, что у вас не подгрузится неправильный чекпоинт.

## Токенизация и подготовка данных

Для токенизации данных необходимо воспользоваться скриптами в [`scripts/data_prep`](scripts/data_prep/), которая содержит свой README с описанием запуска замера и важными моментами.

## Git

Для того, чтобы сделать Merge Request в main ветку LLM Foundry необходимо выполнить ряд условий:

- Указать себя в assignee реквеста
- Указать главного reviewer в реквесте
- Название соответствует формату: `[тип]: краткое описание` (см. доп. инфо ниже)
    - Если реквест не закончен, добавьте в название `Draft:`
- CI/CD пайплайн должен проходить
- Проверить, что contrib зависимости обновлены (при необходимости)
    - Contrib всегда должны вести на main ветку, сначала сделать merge в контрибах
- Добавить параметр по умолчанию в default конфиг (если реквест посвящен разработке новой фичи, которую можно включить или выключить)
- Добавить интеграционные тесты (при запросе)
- Обновить зависимости в setup.py / contrib (если добавлены новые)
- Проверить на локальном окружении

Подробнее про оформление MR вы узнаете на странице `New merge request`.

## Интеграционные тесты

В нашем репозитории есть CI/CD система тестирования. Для того, чтобы залить изменения в main ветку, необходимо, чтобы ваш MR прошел все тесты. Тесты запускаются автоматически при создании MR. Тесты можно запускать и локально в inf джобе. Для их работы необходимо проставить `PYTHONPATH` в корне вашей папки LLM Foundry с помощью команды:
```bash
export PYTHONPATH=$(pwd)/:$(pwd)/contrib/composer:$(pwd)/contrib/streaming:$(pwd)/contrib/gigafsdp
```

Подробную информацию про наши тесты можно найти [здесь](sber_tests/).


## Profiling

Для включения профайлинга нужно в конфиге обучения в разделе `profiler` для тега `use_profiler` поставить значение `true`. Для лучшей визуализации нужно поставить эксперимент на небольшое число итераций. Также лучше убрать всю валидацию, чтобы профайлинг отработал быстрее.

Выбрать ранги, которые будут записывать трейсы профилирования, можно в `profiler.ranks_to_profile`. Номера рангов задаются списком. При `ranks_to_profile: "all"` или пустом теге выбираются все ранги.

Профайлер выдает `json` файлы с информацией о времени выполнения отдельных операций во время обучения. Вся информация сохраняется в `${tensorboard_path}/${run_name}/torch_profiler`. Полученные файлы можно открыть по ссылке `chrome://tracing` для Chrome и SberBrowser.

Для замеров памяти необходимо активировать следующий коллбек:

```
callbacks:
  ...
  # memory_stats_saver:
  #   collect_memory_snapshot: true
  #   folder: '${tensorboard_path}/{run_name}/memory_stats_saver'
  #   save_interval: 2ba
```

Визуализировать полученные файлы можно [по ссылке](https://pytorch.org/memory_viz).

[Ссылка](https://pytorch.org/docs/stable/torch_cuda_memory.html) на memory profiling от PyTorch.

[Ссылка](https://pytorch.org/tutorials/beginner/profiler.html) на ридми про profiling от PyTorch

### Онлайн-профилирование

Для включения онлайн-профилирования необходимо активировать колбек
```
callbacks:
  ...
  online_profiler:
    signal_dir: ${tensorboard_path}/${run_name}/signal_files
    poll_interval_batches: 10
```

Онлайн профилирование работает следующим образом: при активации, колбек каждые `poll_interval_batches` шагов проверяет папку `signal_dir` на наличие сигнального файла `need_profiler*.json`.

Пример `need_profiler*.json`:
```json
{
  "warmup": 6,
  "active": 2,
  "repeat": 1
}
```

Сразу после обнаружения сигнального файла, колбек перехватывает глобальное расписание профилирования и запускает расписание из сигнального файла. После окончания этого расписания, возвращается глобальное расписание, а сигнальный файл удаляется.

Если в папке `signal_dir` несколько сигнальных файлов, они будут обрабатываться по очереди от самого старого к самому новому.

### Аннотирование трейсов

Аннотирование трейсов реализовано в модуле `composer.utils.profiler_annotation.py`.

Для включения аннотирования, для тега `profiler.annotation.enable` нужно выставить значение `true`.

Для аннотирования доступно 3 бэкенда:

- `record_function`: бэкенд по умолчанию, аннотирует json-трейсы
- `nvtx`: требуется установка пакета `nvtx`, аннотирует nsys-трейсы
- `torch_nvtx`: аннотирует nsys-трейсы

Аннотирование возможно:
- с помощью парных `profiler_annotation.push(msg: str)` и `profiler_annotation.pop(msg: str)`.
- с помощью контекстного менеджера `profiler_annotation.annotate(msg: str)
- с помощью декоратора `profiler_annotation.decorator(msg: Optional[str])`

Эти способы аннотируют трейс только на forward-этапе обучения. Для аннотирования backward-этапа необходимо использовать декоратор `decorator_forward_backward(msg: Optional[str])` на методах `forward`.

### Таймеры

Таймеры позволяют замерить время работы методов (с помощью декоратора) или участков кода (с помощью контекстного менеджера).

Для работы таймеров в конфиге необходимо добавить секцию `timers`:

```yaml
timers:
  timers_threshold: 3
  interval: 50
```

где:

- `timers_threshold` - верхняя граница для всех `timer_level`
- `interval` - интервал агрегации таймеров

Важно: малый `interval` вызывает частые cpu-gpu синхронизации, а при `timer_level = 3` - коммуникации

`timer_level` - уровень логирования таймеров:

 - `timer_level = 0` - все таймеры отключены
 - `timer_level = 1` - агрегируются среднее время и количество вызовов за интервал
 - `timer_level = 2` - агрегируются среднее, минимальное, максимальное время и количество вызовов за интервал
- `timer_level = 3` - агрегируются среднее, минимальное, максимальное время, количество вызовов за интервал и межранговые статистики

Если `timer_level > timers_threshold`, то `timer_level` будет понижен до `timers_threshold`.

Все таймеры хранятся в реестре таймеров `Timers`. `Timers` позволяет агрегировать статистики по вызовам таймеров за интервал.

Таймеры в `Timers` разделены по неймспейсам. Неймспейсы позволяют вычислять общее время всех таймеров за интервал и с момента начала обучения.

Чтобы замерить время работы метода, необходимо повесить на него декоратор

```python
@gigatimer(timer_name, timer_type, timer_level)
```

`timer_name` - имя таймера. В имени через точку нужно задать неймспейс:

```python
timer_name = "custom_namespace.timer_name"
```

`timer_type` - либо TimerType.CPU, либо TimerType.CUDA

Использование контекстных менеджеров аналогично.

Есть отдельный декоратор

```python
@callback_timer(timer_type, timer_level)
```

который можно повесить на методы колбеков и не думать о неймспейсах, автоматически будет выбран нужный.

Для замеров времени работы MoE-слоев на разных рангах есть колбек, построенный на CUDA-таймерах:

```yaml
callbacks:
  ...
  moe_ranks_timing:
    prefix: 'moe_rank_timers'
    moe_layer_name: DeepseekGMMMoeBlock  # recommended: ${model.config_overrides.moe_impl}
    timers_level: 2
```

где:

- `prefix` - будет использоваться в качестве неймспейса таймеров, а также при выводе агрегаций в консоль
- `moe_layer_name` - имя MoE-слоя
- `timers_level` - уровень таймеров, которые будут использоваться для замера времени работы MoE-слоя

# Автозапуск джоб во время обучения

Сервис Job-dispatcher позволяет запускать задачи при появлении чекпойнтов обучения. Для его активации необходимо раскомментировать в конфиге секциюю job_dispatcher_callback_web. Сервис запускает цепочку задач - конвертацию модели в HF -> запуск указанных джоб в конфиге

## 📌 Основные возможности
- **Поддерживаемые задачи**: `sam`, `lm_eval`, `longbench`, `passkey`
- **Параллельное выполнение**: одновременный запуск нескольких задач одного типа Пример:
```yaml
      - lm_eval:
          "model_type": "vllm"
          "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=4,gpu_memory_utilization=0.5"
          "tasks": "bbh"
          "num_fewshot": "3"
          "batch_size": "auto"
          "n_gpus": "4"
      - lm_eval:
          "model_type": "vllm"
          "model_args": "dtype=bfloat16,trust_remote_code=True,max_num_seqs=8,max_model_len=8192,tensor_parallel_size=4,gpu_memory_utilization=0.5"
          "tasks": "arc_easy,arc_challenge"
          "num_fewshot": "25"
          "batch_size": "auto"
          "n_gpus": "4"
```

- **Валидация конфига**: проверка конфигурации перед стартом обучения - при неправильном конфиге обучение не начнется

По всем вопросам связанным с автозапуском задач писать:  @dabrazhnik