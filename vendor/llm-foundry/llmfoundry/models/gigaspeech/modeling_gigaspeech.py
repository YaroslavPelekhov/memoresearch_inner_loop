import typing as tp
from contextlib import nullcontext

import torch
from composer.utils import dist, model_eval_mode
from transformers import PretrainedConfig
from transformers.generation import GenerationConfig

from llmfoundry.models.gigar.modelling_gigar import LlamaPreTrainedModel
from llmfoundry.models.utils.param_init_fns import MODEL_INIT_REGISTRY

from .modules import build_acoustic_encoder
from .modules.modality_adapter import build_modality_adapter
from .configuration_gigaspeech import GigaSpeechConfig
from .registry import build_decoder


class GigaSpeechMixin(LlamaPreTrainedModel):

    def __init__(self, config: PretrainedConfig, *args, **kwargs) -> None: # type: ignore
        super().__init__(config, *args, **kwargs)

    def encode_spectrograms(
        self,
        specs: torch.Tensor,
        spec_lengths: torch.Tensor,
    )  -> tp.Tuple[torch.Tensor, ...]:
        """Process spectrograms from raw to acoustic_embedings. Then subsampled them and project
        into LLM prefix space.

        Args:
            specs (torch.Tensor): bs x seq_len x n_mel
            spec_lengths (torch.Tensor): bs

        Returns:
            projected_specs (torch.Tensor): bs x seq_len' x d_model
            projected_lengths (torch.Tensor): bs
        """
        encoder_no_grad_ctx = torch.no_grad() if self.freeze_encoder else nullcontext()
        encoder_eval_ctx = model_eval_mode(self.encoder) if self.freeze_encoder else nullcontext()
        with encoder_no_grad_ctx, encoder_eval_ctx:
            if (
                self.chunk_input_audio_size is not None and
                specs.shape[1] > self.chunk_input_audio_size
            ):
                chunks = list(specs.split(self.chunk_input_audio_size, dim=1))
                batch_size, num_chunks = specs.shape[0], len(chunks)
                chunked_lengths, lengths_cpy = [], spec_lengths.clone()
                for chunk in chunks:
                    max_len = chunk.shape[1]
                    curr_lengths = torch.clamp(lengths_cpy, min=0, max=max_len)

                    lengths_cpy -= curr_lengths
                    chunked_lengths.append(curr_lengths)

                last_chunk, penult_chunk = chunks[-1], chunks[-2]
                if last_chunk.shape[1] != penult_chunk.shape[1]:
                    assert last_chunk.shape[1] < penult_chunk.shape[1]

                    # Padding via `torch.cat` to preserve dtype
                    pad_chunk = torch.zeros(
                        last_chunk.shape[0],
                        self.chunk_input_audio_size - last_chunk.shape[1],
                        last_chunk.shape[2],
                        dtype=last_chunk.dtype,
                        device=last_chunk.device
                    )
                    chunks[-1] = torch.cat([last_chunk, pad_chunk], dim=1)

                chunked_spec, chunked_lengths = torch.cat(chunks, dim=0), torch.cat(chunked_lengths)
                assert chunked_spec.shape[0] == chunked_lengths.shape[0]
                assert chunked_spec.shape[1] == self.chunk_input_audio_size

                chunked_encoded_spec, chunked_encoded_lengths = self.encoder(
                    chunked_spec, chunked_lengths
                )

                subsampled_chunk_size = chunked_encoded_spec.shape[1]
                encoded_spec = chunked_encoded_spec.reshape(
                    batch_size, subsampled_chunk_size * num_chunks, -1
                )
                encoded_lengths = torch.zeros_like(spec_lengths)
                for lengths_chunk in chunked_encoded_lengths.split(batch_size):
                    assert len(lengths_chunk.shape) == 1
                    assert lengths_chunk.shape[0] == batch_size

                    encoded_lengths += lengths_chunk
            else:
                encoded_spec, encoded_lengths = self.encoder(specs, spec_lengths)
        projected_specs, projected_lengths = self.modality_adapter(encoded_spec, encoded_lengths)
        return projected_specs, projected_lengths

    def split_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        pad_sizes: torch.LongTensor,
        generation_config: GenerationConfig
    ) -> tp.Tuple[torch.Tensor, ...]:
        """
        Args:
            input_ids (torch.LongTensor): bs x seq_len
            pad_sizes (torch.LongTensor): bs
            generation_config (GenerationConfig): generation params

        Returns:
            tp.Tuple[torch.Tensor, ...]: _description_
        """
        pad_token_id = generation_config.pad_token_id

        pad_sizes = pad_sizes.clone()

        input_ids_list, labels_list = [], []
        max_len_input_ids, max_len_labels = 0, 0
        for i in range(input_ids.shape[0]):
            role_sep_mask = (input_ids[i] == self.role_sep_id)
            role_sep_idx_list = torch.nonzero(role_sep_mask).squeeze(1)

            msg_sep_mask = (input_ids[i] == pad_token_id)
            msg_sep_idx_list = torch.nonzero(msg_sep_mask).squeeze(1)

            assert torch.numel(role_sep_idx_list) > 0, f"Encountered sample with no role_sep tokens"
            role_sep_idx = role_sep_idx_list[-2]
            msg_sep_idx = msg_sep_idx_list[-2]

            input_ids_split = input_ids[i, :role_sep_idx + 1]
            labels_split = input_ids[i, role_sep_idx + 1:msg_sep_idx + 1]
            
            max_len_input_ids = max(max_len_input_ids, input_ids_split.shape[0])
            max_len_labels = max(max_len_labels, labels_split.shape[0])
            input_ids_list.append(input_ids_split)
            labels_list.append(labels_split)

        assert len(input_ids_list) == len(labels_list)

        for i in range(len(input_ids_list)):
            input_ids_pad_size = max(0, max_len_input_ids - len(input_ids_list[i]))
            input_ids_list[i] = torch.nn.functional.pad(
                input_ids_list[i], (input_ids_pad_size, 0), value=pad_token_id
            )
            pad_sizes[i] += input_ids_pad_size

            labels_pad_size = max(0, max_len_labels - len(labels_list[i]))
            labels_list[i] = torch.nn.functional.pad(
                labels_list[i], (0, labels_pad_size), value=pad_token_id
            )

        input_ids = torch.vstack(input_ids_list)
        labels = torch.vstack(labels_list)

        assert pad_sizes.max() < input_ids.shape[1], f"input_ids: {input_ids.shape}, pad_sizes: {pad_sizes}"

        return input_ids, labels, pad_sizes

    def prepare_inputs_for_acoustic_modality(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        spectrograms: tp.Optional[torch.Tensor],
        spectrograms_lengths: tp.Optional[torch.Tensor],
        pad_sizes: tp.Optional[torch.LongTensor] = None,
        left_padding: bool = False,
    ) -> tp.Tuple[torch.Tensor, ...]:
        """
        Inserts audio features into input token embeddings `left_padding` mode is created for generation
        You also should provide pad_sizes for every input sequence in input_ids to correctly
        build attention mask.
        """
        # TODO (fedorovgv) : переделать эту простыню
        combined_seq_lengths = torch.full(
            size=(input_ids.shape[0],),
            fill_value=input_ids.shape[1],
            dtype=torch.long,
            device=input_ids.device
        )

        has_audio = spectrograms is not None

        if has_audio:
            assert spectrograms_lengths is not None

            acoustic_embeds, encoded_lengths = self.encode_spectrograms(spectrograms, spectrograms_lengths)

            acoustic_idx = 0
            for i in range(input_ids.shape[0]):
                audio_token_mask = (input_ids[i] == self.audio_token_id)
                num_audio_tokens = torch.sum(audio_token_mask)

                for _ in range(num_audio_tokens.item()):
                    # input_ids already contain one audio token
                    combined_seq_lengths[i] += encoded_lengths[acoustic_idx] - 1
                    acoustic_idx += 1
            assert acoustic_idx == acoustic_embeds.shape[0], \
                "Different number of audio tokens and spectrograms"
        else:
            # For every worker to call `forward` on acoustic encoder we create dummy
            # spectrogram inputs
            dummy_spec = torch.randn(1, 20, 64, dtype=torch.float16, device=input_ids.device)
            dummy_lengths = torch.tensor([20], dtype=torch.long, device=input_ids.device)
            acoustic_embeds, encoded_lengths = self.encode_spectrograms(dummy_spec, dummy_lengths)

        max_combined_seq_length = combined_seq_lengths.max()

        if left_padding:
            assert pad_sizes is not None
            pad_sizes = pad_sizes.clone()

        input_ids_list, labels_list, attention_list = [], [], []
        acoustic_idx = 0
        for i in range(input_ids.shape[0]):
            audio_token_mask = (input_ids[i] == self.audio_token_id)
            num_audio_tokens = torch.sum(audio_token_mask)

            pad_size = max_combined_seq_length - combined_seq_lengths[i]
            empty_pad = torch.tensor([], device=input_ids[i].device, dtype=torch.long)
            if num_audio_tokens > 0:
                assert has_audio

                audio_token_idx_list = torch.nonzero(audio_token_mask).squeeze(1)

                curr_input_ids, curr_labels = [], []
                if left_padding:
                    curr_input_ids.append(torch.zeros((pad_size,), dtype=torch.long, device=input_ids[i].device))
                    curr_labels.append(torch.full((pad_size,), -100, dtype=torch.long, device=labels[i].device))

                prev_audio_token_idx = None
                for j in range(len(audio_token_idx_list)):
                    curr_audio_token_idx = audio_token_idx_list[j]
                    curr_input_ids.extend([
                        input_ids[i][prev_audio_token_idx:curr_audio_token_idx],
                        torch.full((encoded_lengths[acoustic_idx],), self.audio_token_id, dtype=torch.long, device=input_ids[i].device),
                    ])
                    curr_labels.extend([
                        labels[i][prev_audio_token_idx:curr_audio_token_idx],
                        torch.full((encoded_lengths[acoustic_idx],), -100, dtype=torch.long, device=labels[i].device),
                    ])

                    if j == len(audio_token_idx_list) - 1:
                        curr_input_ids.append(input_ids[i][curr_audio_token_idx + 1:])
                        curr_labels.append(labels[i][curr_audio_token_idx + 1:])
                    
                    prev_audio_token_idx = curr_audio_token_idx + 1
                    acoustic_idx += 1

                if not left_padding:
                    curr_input_ids.append(torch.zeros((pad_size,), dtype=torch.long, device=input_ids[i].device))
                    curr_labels.append(torch.full((pad_size,), -100, dtype=torch.long, device=labels[i].device))

                input_ids_list.append(torch.cat(curr_input_ids))
                labels_list.append(torch.cat(curr_labels))
            else:
                input_ids_list.append(torch.cat([
                    torch.zeros((pad_size,), dtype=torch.long, device=input_ids[i].device) if left_padding else empty_pad,
                    input_ids[i],
                    empty_pad if left_padding else torch.zeros((pad_size,), dtype=torch.long, device=input_ids[i].device)
                ]))

                labels_list.append(torch.cat([
                    torch.full((pad_size,), -100, dtype=torch.long, device=labels[i].device) if left_padding else empty_pad,
                    labels[i],
                    empty_pad if left_padding else torch.full((pad_size,), -100, dtype=torch.long, device=labels[i].device),
                ]))

            if left_padding:
                pad_part = torch.full((pad_size + pad_sizes[i],), 0, dtype=torch.long, device=labels[i].device)
                value_part = torch.full((combined_seq_lengths[i] - pad_sizes[i],), 1, dtype=torch.long, device=labels[i].device)
                attention_part = [pad_part, value_part] if left_padding else [value_part, pad_part]
                attention_list.append(torch.cat(attention_part))

                pad_sizes[i] += pad_size

        if has_audio:
            assert acoustic_idx == acoustic_embeds.shape[0], \
                "Different number of audio tokens and spectrograms"
        else:
            assert acoustic_idx == 0, "input contains audio tokens without input spectrograms"

        input_ids = torch.vstack(input_ids_list)
        labels = torch.vstack(labels_list)
        attention_mask = None

        assert input_ids.shape == labels.shape, \
            f"input_ids.shape != labels.shape: {input_ids.shape, labels.shape}"

        if left_padding:
            attention_mask = torch.vstack(attention_list)
            assert input_ids.shape == attention_mask.shape, \
                f"input_ids.shape != attention_mask.shape: {input_ids.shape, attention_mask.shape}"

        if left_padding:
            min_pad_size = pad_sizes.min()

            input_ids = input_ids[:, min_pad_size:]
            labels = labels[:, min_pad_size:]
            attention_mask = attention_mask[:, min_pad_size:]

            pad_sizes -= min_pad_size

        audio_token_mask = (input_ids == self.audio_token_id)
        # if self.audio_token_id >= self.base_vocab_size:
        #     input_ids[audio_token_mask] = SWAP_TOKEN_ID

        # Create emptry mask if no real audio slots are presented in input_embeds
        if not has_audio:
            encoded_lengths = torch.tensor([0], dtype=torch.long, device=input_ids.device)
        acoustic_feature_mask = torch.arange(acoustic_embeds.shape[1], device=input_ids.device)
        acoustic_feature_mask = acoustic_feature_mask < encoded_lengths.unsqueeze(1)

        sp_group_size = dist.get_sp_group_size() if dist.get_sp_group() is not None else 1
        bs_, seq_len_ = input_ids.size()
        if seq_len_ % sp_group_size != 0 and not left_padding:
            padding_size = sp_group_size - (seq_len_ % sp_group_size)
            if padding_size > 0:
                pad_tensor = torch.zeros(
                    (bs_, padding_size), device=input_ids.device, dtype=input_ids.dtype,
                )
                pad_tensor.fill_(self.generation_config.pad_token_id)
                input_ids = torch.cat([input_ids, pad_tensor], dim=1)
                pad_tensor.fill_(-100)
                labels = torch.cat([labels, pad_tensor], dim=1)

        inputs_embeds = self.decoder.get_input_embeddings()(input_ids)
        inputs_embeds = inputs_embeds.clone()
        if seq_len_ % sp_group_size != 0 and not left_padding:
            inputs_embeds[:,:seq_len_,:][audio_token_mask] = acoustic_embeds[acoustic_feature_mask]
        else:
            inputs_embeds[audio_token_mask] = acoustic_embeds[acoustic_feature_mask]

        if sp_group_size > 1 and not left_padding:
            assert inputs_embeds.size(1) % sp_group_size == 0
            assert labels.size(1) == inputs_embeds.size(1)

        return None, labels, inputs_embeds, attention_mask


class GigaSpeechForCausalLM(GigaSpeechMixin):

    def __init__(
            self,
            config: GigaSpeechConfig, 
            generation_config: GenerationConfig,
            *args: tp.List,
            **kwargs: tp.Dict,
        ) -> None:
        super().__init__(config, *args, **kwargs)
        self._config = config

        self.encoder = build_acoustic_encoder(config.encoder_config)
        self.modality_adapter = build_modality_adapter(config.modality_adapter_config)
        self.decoder = build_decoder(config.decoder_config)

        self.encoder.apply(lambda x: setattr(x, "_is_hf_initialized", True))
        self.modality_adapter.apply(lambda x: setattr(x, "_is_hf_initialized", True))

        self.freeze_encoder = config.freeze_encoder

        self.audio_token_id = config.audio_token_id
        self.message_sep_id = config.message_sep_id
        self.role_sep_id = config.role_sep_id

        self.generation_config = generation_config
        self.base_vocab_size = config.vocab_size
        self.chunk_input_audio_size = config.chunk_input_audio_size

        self.return_dict = config.return_dict

    def param_init_fn(self, module: torch.nn.Module) -> None:
        if hasattr(module, "_lora_b_zero_init"):
            torch.nn.init.zeros_(module.weight)
            return

        gain = 1.0
        init_config = {
            'name': 'xavier_normal_',
            'init_gain': gain,
            'init_div_is_residual': False,
            'emb_init_std': None,
            'emb_init_uniform_lim': None,
        }

        init_fn_name = init_config["name"]
        MODEL_INIT_REGISTRY[init_fn_name](
            module=module,
            n_layers=self._config.decoder_config.num_hidden_layers,
            **init_config,
        )

    def reset_parameters(self):
        return super().reset_parameters()

    def set_generation_config(self, generation_config: GenerationConfig) -> None:
        self.generation_config = generation_config

    def get_decoder(self):
        return self.decoder

    def get_encoder(self):
        return self.encoder

    def get_modality_adapter(self):
        return self.modality_adapter

    def forward(
        self,
        input_ids: tp.Optional[torch.LongTensor] = None,
        attention_mask: tp.Optional[torch.Tensor] = None,
        position_ids: tp.Optional[torch.LongTensor] = None,
        past_key_values: tp.Optional[tp.List[torch.FloatTensor]] = None,
        inputs_embeds: tp.Optional[torch.FloatTensor] = None,
        labels: tp.Optional[torch.LongTensor] = None,
        use_cache: tp.Optional[bool] = None,
        output_attentions: tp.Optional[bool] = None,
        output_hidden_states: tp.Optional[bool] = None,
        spectrograms: tp.Optional[torch.FloatTensor] = None,
        spectrogram_lengths: tp.Optional[torch.LongTensor] = None,
        return_dict: tp.Optional[bool] = None,
    ) -> tp.Tuple:
        """
        Args:
            spectrograms (tp.Optional[torch.FloatTensor], optional): _description_. Defaults to None.
            spectrogram_lengths (tp.Optional[torch.LongTensor], optional): _description_. Defaults to None.
        """
        if inputs_embeds is None:
            _, labels, inputs_embeds, _ = self.prepare_inputs_for_acoustic_modality(
                input_ids,
                labels,
                spectrograms,
                spectrogram_lengths
            )
        input_ids = None if inputs_embeds is not None else input_ids
 
        return_dict = return_dict if return_dict is not None else self.return_dict

        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs

    @torch.no_grad()
    def generate_continuation(
        self, 
        input_ids: torch.Tensor,
        spectrograms: tp.Optional[torch.FloatTensor],
        spectrogram_lengths: tp.Optional[torch.LongTensor],
        pad_sizes: torch.LongTensor,
        generation_config: GenerationConfig,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """Generate fn.
        """
        assert generation_config.eos_token_id == generation_config.pad_token_id
        assert generation_config.num_beams == 1

        # Pass `input_ids` as `labels` for simplicity. We dont need them anyway
        _, _, inputs_embeds, attention_mask = self.prepare_inputs_for_acoustic_modality(
            input_ids, input_ids, spectrograms, spectrogram_lengths, pad_sizes, left_padding=True
        )

        def _has_unfinished_sequences(this_peer_finished: bool, device: torch.device) -> bool:
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(device)
            dist.all_reduce(this_peer_finished_flag, "SUM")
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                return False
            return True

        temperature = generation_config.temperature if generation_config.do_sample else None
        top_p = generation_config.top_p if generation_config.do_sample else None
        repetition_penalty = generation_config.repetition_penalty
        use_cache = generation_config.use_cache
        past_key_values = None
        position_ids = None

        batch_size = input_ids.shape[0]
        pad_eos_token = torch.tensor([generation_config.pad_token_id], device=input_ids.device, dtype=torch.long)
        found_eos = torch.zeros(batch_size, device=input_ids.device, dtype=torch.bool)
        generated_input_ids = torch.tensor([], device=input_ids.device, dtype=torch.long)

        logits: tp.Optional[torch.tensor] = None

        while _has_unfinished_sequences(
            len(generated_input_ids.shape) > 1 and
            generated_input_ids.shape[1] >= generation_config.max_length or
            found_eos.all().item(), input_ids.device
        ):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

            outputs = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=self.return_dict,
                is_left_padded_eval=True,
            )

            if not self.return_dict:
                logits_for_next_step = outputs[0][:, -1, :]
                past_key_values = outputs[1] if use_cache else None
            else:
                logits_for_next_step = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

            if return_logits:
                logits = (
                    torch.cat([logits, outputs[0][:, -1, :][:, None, :]], dim=1)
                    if logits is not None else outputs[0]
                )

            # Repetition penalty
            extracted_scores = torch.gather(logits_for_next_step, 1, generated_input_ids)
            extracted_scores = torch.where(
                extracted_scores < 0,
                extracted_scores * repetition_penalty,
                extracted_scores / repetition_penalty
            )
            logits_for_next_step = logits_for_next_step.scatter(1, generated_input_ids, extracted_scores)

            if generation_config.do_sample:
                # Temperature
                logits_for_next_step = logits_for_next_step / temperature

                # Top p
                sorted_logits, sorted_indices = torch.sort(logits_for_next_step, descending=False)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

                # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
                sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
                # Keep at least 1 token
                sorted_indices_to_remove[..., -1:] = 0

                # scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits_for_next_step = logits_for_next_step.masked_fill(indices_to_remove, -float("Inf"))

                probs = torch.nn.functional.softmax(logits_for_next_step, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                if use_rank_zero_tokens:
                    next_tokens = dist.all_gather(next_tokens, dist.get_tp_group())[0]
            else:
                next_tokens = torch.argmax(logits_for_next_step, dim=-1)

            found_eos = torch.logical_or(found_eos, next_tokens == pad_eos_token)
            next_tokens = next_tokens * (1 - found_eos.long()) + pad_eos_token * found_eos.long()
            generated_input_ids = torch.cat([generated_input_ids, next_tokens[:, None]], dim=-1)

            # NOTE (Sbr, fedorovgv): for gigafsdp compatibility, do fake forward the dummy spec through the encoder 
            # and modality adapter
            dummy_spec = torch.randn(1, 20, 64, dtype=torch.float16, device=input_ids.device)
            dummy_lengths = torch.tensor([20], dtype=torch.long, device=input_ids.device)
            _, _ = self.encode_spectrograms(dummy_spec, dummy_lengths)

            next_input_embed = self.decoder.get_input_embeddings()(next_tokens.unsqueeze(1))
            inputs_embeds = next_input_embed if use_cache else torch.cat([inputs_embeds, next_input_embed], dim=1)

            attention_mask = torch.hstack([
                attention_mask,
                torch.ones(batch_size, 1, dtype=torch.long, device=attention_mask.device)
            ])

        # NOTE (Sbr, fedorovgv): call false self.decoder for gigafsdp buffer release
        # rewrite it more efficiently
        dummy_input_embeds = torch.zeros(
            (1, 2, inputs_embeds.size(-1)), device=inputs_embeds.device, dtype=inputs_embeds.dtype,
        )
        _ = self.decoder(inputs_embeds=dummy_input_embeds, return_dict=False)

        return generated_input_ids, logits
