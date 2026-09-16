# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from llmfoundry.models.hf import (ComposerHFCausalLM, ComposerHFPrefixLM,
                                  ComposerHFT5)
from llmfoundry.models.gigar import ComposerGigarCausalLM
from llmfoundry.models.giga_mix import ComposerGigaMixCausalLM
from llmfoundry.models.gigavision import ComposerGigaVisionCausalLM
from llmfoundry.models.gigaspeech import ComposerGigaSpeechCausalLM
from llmfoundry.models.gigatts import ComposerGigaTTSCausalLM
from llmfoundry.models.distil import ComposerDistilTeacherStudentModel
from llmfoundry.models.dpo import ComposerDPOModel


COMPOSER_MODEL_REGISTRY = {
    'gigar_causal_lm': ComposerGigarCausalLM,
    'giga_mix_causal_lm': ComposerGigaMixCausalLM,
    'gigavision_causal_lm': ComposerGigaVisionCausalLM,
    'gigaspeech_causal_lm': ComposerGigaSpeechCausalLM,
    'gigatts_causal_lm': ComposerGigaTTSCausalLM,
    'hf_causal_lm': ComposerHFCausalLM,
    'hf_prefix_lm': ComposerHFPrefixLM,
    'hf_t5': ComposerHFT5,
    'distil': ComposerDistilTeacherStudentModel,
    'dpo': ComposerDPOModel
}

