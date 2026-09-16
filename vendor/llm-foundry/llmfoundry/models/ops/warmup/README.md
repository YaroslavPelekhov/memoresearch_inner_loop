# Warmup: Triton

---

## Прогрев (снять autotune и собрать `*_best_config.pkl`)

Эта часть документации предназначена для тех случаев, когда вам необходимо получить для ваших новых ядер или уже существующих новые параметры `num_warps`/`num_stages`/`maxnreg` и т.д.

### Пошаговая инструкция

1. В `sber_scripts/submit_warmup.py` выставляем `warmup_type="triton"` и путь до warmup конфига.
2. Внутри warmup конфига указываем, какие ядра мы хотим прогреть. Функции прогрева должны заранее быть написаны и лежать в `llmfoundry/models/ops/warmup/triton_functions.py` и зарегистрированы там же в маппингах: `WARMUP_FNS`, `_get_autotuned_kernels()`,  `GROUP_TO_AUTOTUNE_KERNELS` и `KERNEL_KEY_METADATA`.
  Кратко по каждому из маппингов:
  - `WARMUP_FNS` в нем указываем название warmup-функции, под таким названием она будет вызываться из конфига.
  - `_get_autotuned_kernels()` маппинг между тем, как будет называться прогреваемое ядро и его непосредственной функцией вызова.
  - `GROUP_TO_AUTOTUNE_KERNELS` в одной warmup-функции может прогреваться сразу несколько ядер, поэтому здесь указываем списком названия ядер из предыдущего пункта.
  - `KERNEL_KEY_METADATA` исходя из того, как в предыдущем маппинге назвали ядра, указываем для них ключи (`key`), на которые ориентируется warmup и игнорируемые ключи (`ignore_key`) для тех ключей, которые мы хотим игнорировать при получении best-конфига. На данный момент в `ignore_key` заносятся те ключи, которые связаны с размерностью по токенам.

#### Зарегистрированные warmup-функции


| Ключ в `WARMUP_FNS` (в конфиге: `kernel_groups` / `dense_groups`) | Функция прогрева в `triton_functions.py` | Ядра autotune (`GROUP_TO_AUTOTUNE_KERNELS`)          |
| ----------------------------------------------------------------- | ---------------------------------------- | ---------------------------------------------------- |
| `blocks_quantization`                                             | `warmup_blocks`                          | `block_quantization`, `block_quantization_transpose` |
| `row_quantization`                                                | `warmup_row_quantization`                | `row_quantization`                                   |
| `row2col_te`                                                      | `warmup_row2col_te`                      | `row2col_te`                                         |
| `row2col_deepgemm`                                                | `warmup_row2col_deepgemm`                | `row2col_deepgemm`                                   |
| `fused_swiglu`                                                    | `warmup_fused_swiglu`                    | `swiglu_fwd`, `swiglu_bwd`                           |
| `mla_q_rope`                                                      | `warmup_mla_q_rope`                      | — (только JIT, autotune не сохраняем)                |
| `permute_fwd`                                                     | `warmup_permute_fwd`                     | `permute_fwd`                                        |
| `permute_bwd`                                                     | `warmup_permute_bwd`                     | `permute_bwd`                                        |
| `unpermute_fwd`                                                   | `warmup_unpermute_fwd`                   | `unpermute_fwd`                                      |
| `unpermute_bwd`                                                   | `warmup_unpermute_bwd`                   | `unpermute_bwd`                                      |
| `dense_block_quantize`                                            | `warmup_dense_block_quantize`            | `dense_block_quantize` (в конфиге — `dense_groups`)  |
| `dense_row2col`                                                   | `warmup_dense_row2col`                   | `dense_fp8_transpose`                                |
| `dense_fused_swiglu`                                              | `warmup_dense_swiglu`                    | `dense_swiglu_fwd`, `dense_swiglu_bwd`               |


1. `**lookup_autotuner**` (`llmfoundry/models/ops/triton_autotuner.py`) — над `@triton.jit` вместо «голого» `@triton.autotune`: в `warmup` включается autotune, в `training` — подстановка записей из pickle. (Последнее не так актуально.)
  ```python
   _AUTOTUNE_CONFIGS = [triton.Config(...), ...]

   triton_mode = os.getenv("TRITON_MODE", "training")   # при запуске submit_warmup.py проставляется само
   triton_configs_dir = os.getenv("TRITON_CONFIGS_DIR") # куда сохраняем результат
   # для mode="warmup" достаточно передать _AUTOTUNE_CONFIGS и key как и в обычном triton.autotune, т.к. сохранение pkl и tsv файлов происходит на основе маппингов. Остальные параметры предназначены в случае triton_mode="training". (сейчас не используется.)   
   @lookup_autotuner(
       mode=triton_mode,
       best_config_path=f"{triton_configs_dir}/my_kernel_best_config.pkl" 
           if triton_configs_dir else None,
       best_config_path_bwd = f"{triton_configs_dir}/my_kernel_bwd_best_config.pkl" # если кернел может также использоваться на bwd
           if triton_configs_dir else None,
       configs=_AUTOTUNE_CONFIGS,
       key=["total_token_bucket", "hidden_size"],
       ignore_keys_on_save=["total_token_bucket"] # игнорируемые ключи
   )
   @triton.jit
   def my_kernel(...):
       ...
  ```

# Warmup: DeepGEMM

---

## Прогрев (получить папку с кэшом и положить на s3)

Эта часть документации предназначена для тех случаев, когда по какой-то причине произошла инвалидация существующего кэша для DeepGEMM (переезд на sm100+, новая версия DeepGEMM и тд).

### Пошаговая инструкция

1. В `sber_scripts/submit_warmup.py` выставляем `warmup_type="deep_gemm"` и путь до warmup конфига.
2. Внутри warmup конфига указываем, какие ядра мы хотим прогреть. По классике там указано 6 ядер = 1 fwd + 2 bwd для w1/w2 проекций. В идеальном кейсе мы просто запускаем скрипт `submit_warmup.py` и получаем локально папку с кэшем.
3. Сохраняем на s3 прогретый кэш и при необходимости пересобираем докер. Путь до s3 можно найти в репозитории **CI-CD**, `scripts/download_deep_gemm_cache_s3.sh`.

