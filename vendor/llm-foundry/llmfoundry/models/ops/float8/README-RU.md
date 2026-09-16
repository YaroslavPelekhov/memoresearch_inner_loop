# Float8 training

В репозитории живет две независимые схемы float8 обучения, которые работают независимо друг с другом - квантизация dense и sparse слоев.

⚠️ Обучение возможно только на картах Hopper (H100) в данный момент!

## Sparse MoE FP8

### Quick start

Для включения необходимы два флага:

```yaml
model:
  config_overrides:
    use_float8_grouped_gemm: true
    use_new_expert_weight_layout: true
```

Для эффективного обучения нужен DeepGEMM cache - скомпилированные закешированные ядра, он должен быть в нашем docker image и прогретый с помощью нашего форка DeepGEMM. Сейчас в контейнере будет кзш для размеров:

| hid\_dim | int\_dim | num\_groups | num\_sms | min | max |
|---|---|---|---|---|---|
| 7168 | 2048 | 4, 6 | 96, 116, 132 | 128 | 64000 |
| 2560 | 1536 | 9, 18, 36, 72 | 96, 116 | 128 | 64000 |


Если ваших размерностей нет в табличке, то прогрейте кэш самостоятельно (см. ниже).

### In Details

#### Deep GEMM

Эффективное обучение в float8 реализовано за счет библиотеки DeepGEMM, которая реализует умножение матриц групп экспертов на матрицу групп токенов - Grouped GEMM в float8. DeepGEMM использует Just In Time (-JIT) компиляцию и для конкретных размеров матрицы, параметров видеокарты и других параметров определяет наилучшую кофнигурацию кернела, которое будет делать умножение, после делает его компиляцию. В начале обучения это может вызвать часовые простои, поскольку разные ранги будут ждать компиляцию на других рангах, поэтому для максимально эффективной работы необходимо сделать прогрев кэша DeepGEMM и затем передать его на каждый ранг обучения.

#### Quick deep gemm cache warmup

У нас написаны скрипты прогрева кэша для cloud и для сущностей, для этого нужно заполнить конфиг прогрева

[`warmup_config.yaml`](../../../../sber_configs/warmup/)

после чего запустить

[`submit_warmup.py`](../../../../contrib/job_submitter/job_submitter/jobs/llmf/warmup.py)

Пример для Ultra, где `num_groups = num_routed_experts // ep_size`, `min_size` - минимальное количество токенов на одного эксперта, `max_size` - максимальное количество токенов, которое может прийти на эксперта - можно ориентироваться на параметр expert capacity в конфиге, если он есть и если нет, оставить значение по умолчанию - 64'000. 

```yaml
deep_gemm:
  cache_dir: "/path/to/cache/folder"
  gemm_groups:
    - "w1_m_grouped"
    - "w1_m_grouped_nt_for_dgrad_transposed"
    - "w1_k_grouped"
    - "w2_m_grouped"
    - "w2_m_grouped_nt_for_dgrad_transposed"
    - "w2_k_grouped"
  hid_dim: 7168
  int_dim: 2048
  num_groups: 6
  min_size: 128
  max_size: 51328
  num_sms: 116
```

После чего запустить скрипт и в переменные обучения прокинуть

```python
"PROPAGATE_DG_CACHE": 1,  
"PROPAGATE_DG_CACHE_DIR": "/path/to/cache/folder",
```

**UPD:** Теперь отсутствие кэша по дефолту вызывает **ошибку**:
```console
RuntimeError: Assertion error (csrc/apis/../jit_kernels/impls/../../jit/compiler.hpp:169): false and "CACHE MISS: set DG_CACHE_FORCE_SKIP=1 to allow runtime compilation, or warm up the cache before running"
``` 
Если вы **не хотите** прогревать кэш по тем или иным причинам, добавьте в переменные обучения ```DG_CACHE_FORCE_SKIP=1```, тогда недостающие ядра будут компилироваться в рантайме и ошибки не возникнет.  

<!-- ### Forward pass

```text
     input  (bf16, [batch_size, seq_len, hidden_dim])
       │
       ▼
  ┌──────────────────┐
  │      router      │
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ DeepEP dispatch  │   row-quant + all-to-all   (bf16 → FP8)
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ permute + padding│   group tokens by expert, pad each group to ×128
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ Grouped linear 1 │   Fprop   (FP8 × FP8  →  bf16)
  │   (up / gate)    │
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │   activation +   │   fused SwiGLU + row-quant   (bf16 → FP8)
  │     row-quant    │
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ Grouped linear 2 │   Fprop   (FP8 × FP8  →  bf16)
  │     (down)       │
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │  un-padding +    │   strip padding, un-permute
  │    un-permute    │
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │  DeepEP combine  │   reverse all-to-all   (bf16)
  └────────┬─────────┘
           ▼
     output (bf16, [batch_size, seq_len, hidden_dim])
```

### Backward pass

```text
  output_grad  (bf16, [batch_size, seq_len, hidden_dim])
       │
       ▼
  ┌──────────────────┐
  │ DeepEP combine   │   reverse all-to-all of grads   (bf16)
  │      (bwd)       │
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │  un-padding +    │   re-pad + permute grads back
  │ un-permute (bwd) │
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ Grouped linear 2 │   Dgrad + Wgrad   (both FP8 GEMMs,
  │     (down)       │                    share the same weight)
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │  activation +    │   SwiGLU backward   (bf16, STE through quant)
  │ row-quant (bwd)  │
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ Grouped linear 1 │   Dgrad + Wgrad   (both FP8 GEMMs,
  │   (up / gate)    │                    share the same weight)
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ permute + padding│   un-permute + strip padding of grads
  │      (bwd)       │
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ DeepEP dispatch  │   reverse all-to-all of grads   (bf16)
  │      (bwd)       │
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │      router      │   scatter grads back to original token slots
  │       (bwd)      │
  └────────┬─────────┘
           ▼
  input_grad   (bf16, [batch_size, seq_len, hidden_dim])
``` -->

## Dense MLP FP8

Dense FP8 — это полностью отдельная интеграция по сравнению с MoE. Здесь используется **FP8 модули из Transformer Engine**, применяемый к линейным модулям целиком (`gate_proj`, `up_proj`, `down_proj`, а также при необходимости к attention-проекциям).

### Quick start

Dense FP8 включается через секцию `fp8_config:`, которая сообщает `apply_fp8_config()` в [`scripts/train/train.py`](../../../../scripts/train/train.py), что нужно заменить каждый подходящий `nn.Linear` на `TEFP8Linear` из Transformer Engine.

```yaml
fp8_config:
  include_linear_layers: ".*\\.(down_proj|gate_proj|up_proj|q_proj|k_proj|v_proj|o_proj)$"
  include_shared_quantizers: ".*\\.(mlp)$"
```
- В `include_linear_layers` указываются те паттерны, по которым `nn.Linear` будет заменен на `TEFP8Linear`. 
- В `include_shared_quantizers` можно указать те слои, для которых будет вызван pre-forward hook для квантизации активаций. Применимо для тех блоков, у которых несколько линейных слоев на вход принимают одну и ту же активацию (Например, в `mlp` – `gate_proj` и `up_proj` линейные слои).       

### Requirements

- **GPU**: `sm89` (Ada) или новее.
- **`tp_size=1`**: dense FP8 сейчас поддерживается только для запусков без tensor parallelism (это проверяется через assert в `train.py` в месте вызова `apply_fp8_config`).
- **Несовместимо с `fused_mlp=True`**: TE заменяет отдельные линейные модули, из-за чего ломается путь с fused-MLP-ядром. Это проверяется assert’ом при старте в [`ffn.py`](../../layers/ffn.py).
- **Несовместимо с `fused_mlp_checkpoint_lvl > 0`**: по той же причине.

### Без прогрева кэша

В отличие от sparse, для dense FP8 **не нужен прогрев кэша ядер**. Forward/backward в TE работают поверх FP8 GEMM из cuBLAS, а не на JIT-компилируемых шаблонах DeepGEMM под конкретные формы, поэтому здесь нечего заранее прогревать офлайн. Достаточно включить `fp8_config:` и запустить обучение — первый шаг сразу будет выполняться на полной скорости.
