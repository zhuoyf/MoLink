import torch
import warnings
from vllm import envs
from vllm.worker.model_runner import ModelRunner, ModelInputForGPUWithSamplingMetadata
from vllm.config import CompilationLevel
from vllm.logger import init_logger
from vllm.utils import (DeviceMemoryProfiler, supports_dynamo)
from vllm.model_executor.models import supports_lora, supports_multimodal
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.prompt_adapter.worker_manager import (
    LRUCacheWorkerPromptAdapterManager)
from vllm.platforms import current_platform
from molink.model_executor.model_loader import get_model
from typing import (List, Optional, Union)
from vllm.distributed import broadcast_tensor_dict, get_pp_group
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.sampler import (Sampler, SamplerOutput)
from vllm.multimodal import MultiModalKwargs
from vllm.sequence import IntermediateTensors
import time


logger = init_logger(__name__)

class MolinkGPUModelRunner(ModelRunner):
    def load_model(self) -> None:
        #zyflog：真实加载模型函数
        logger.info("Starting to load model %s...", self.model_config.model)
        with DeviceMemoryProfiler() as m:
            self.model = get_model(vllm_config=self.vllm_config)

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))

        # TODO test
        self.offload_weight()

        if self.lora_config:
            assert supports_lora(
                self.model
            ), f"{self.model.__class__.__name__} does not support LoRA yet."

            if supports_multimodal(self.model):
                logger.warning("Regarding multimodal models, vLLM currently "
                               "only supports adding LoRA to language model.")
            # It's necessary to distinguish between the max_position_embeddings
            # of VLMs and LLMs.
            if hasattr(self.model.config, "max_position_embeddings"):
                max_pos_embeddings = self.model.config.max_position_embeddings
            else:
                max_pos_embeddings = (
                    self.model.config.text_config.max_position_embeddings)

            self.lora_manager = LRUCacheWorkerLoRAManager(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
                self.vocab_size,
                self.lora_config,
                self.device,
                self.model.embedding_modules,
                self.model.embedding_padding_modules,
                max_position_embeddings=max_pos_embeddings,
            )
            self.model = self.lora_manager.create_lora_manager(self.model)

        if self.prompt_adapter_config:
            self.prompt_adapter_manager = LRUCacheWorkerPromptAdapterManager(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens, self.device,
                self.prompt_adapter_config)
            self.model = (
                self.prompt_adapter_manager.create_prompt_adapter_manager(
                    self.model))

        if self.kv_cache_dtype == "fp8" and (current_platform.is_rocm()
                                             or current_platform.is_cuda()):
            # Currently only ROCm accepts kv-cache scaling factors
            # via quantization_param_path and this will be deprecated
            # in the future.
            if self.model_config.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    warnings.warn(
                        "Loading kv cache scaling factor from JSON is "
                        "deprecated and will be removed. Please include "
                        "kv cache scaling factors in the model checkpoint.",
                        FutureWarning,
                        stacklevel=2)
                    self.model.load_kv_cache_scales(
                        self.model_config.quantization_param_path)
                    logger.info("Loaded KV cache scaling factors from %s",
                                self.model_config.quantization_param_path)
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        "model %s does not support loading scaling factors.",
                        self.model.__class__)
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors "
                    "provided. Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!")

        if self.vllm_config.compilation_config.level ==\
            CompilationLevel.DYNAMO_AS_IS and supports_dynamo():
            backend = self.vllm_config.compilation_config.init_backend(
                self.vllm_config)
            self.model = torch.compile(
                self.model,
                fullgraph=envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                backend=backend)
            

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
        **kwargs,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
        # print(f"here in execute model!!  {time.time()}", flush=True)
        # with open('/env/offloading.log', 'a') as f:
        #     f.write(f"here in execute model!!  {time.time()}\n")
        if num_steps > 1:
            raise ValueError("num_steps > 1 is not supported in ModelRunner")

        if self.lora_config:
            assert model_input.lora_requests is not None
            assert model_input.lora_mapping is not None
            self.set_active_loras(model_input.lora_requests,
                                  model_input.lora_mapping)

        if self.prompt_adapter_config:
            assert model_input.prompt_adapter_requests is not None
            assert model_input.prompt_adapter_mapping is not None
            self.set_active_prompt_adapters(
                model_input.prompt_adapter_requests,
                model_input.prompt_adapter_mapping)

        self.attn_state.begin_forward(model_input)

        # Currently cuda graph is only supported by the decode phase.
        assert model_input.attn_metadata is not None
        prefill_meta = model_input.attn_metadata.prefill_metadata
        decode_meta = model_input.attn_metadata.decode_metadata
        # TODO(andoorve): We can remove this once all
        # virtual engines share the same kv cache.
        virtual_engine = model_input.virtual_engine
        previous_hidden_states = kwargs.get("previous_hidden_states")
        if prefill_meta is None and decode_meta.use_cuda_graph:
            assert model_input.input_tokens is not None
            graph_batch_size = model_input.input_tokens.shape[0]
            use_inputs_embeds = model_input.inputs_embeds is not None
            model_executable = self.graph_runners[virtual_engine][(
                graph_batch_size, use_inputs_embeds)]
            if previous_hidden_states is not None:
                previous_hidden_states = torch.cat([
                    previous_hidden_states,
                    torch.empty([
                        graph_batch_size - previous_hidden_states.shape[0],
                        *previous_hidden_states.shape[1:]
                    ],
                                dtype=previous_hidden_states.dtype,
                                device=previous_hidden_states.device)
                ])
        else:
            model_executable = self.model

        # Receive KV cache in distributed KV cache transfer setting
        # In disagg prefill setting, it will also recv hidden states and bypass
        # model forwarding
        # In KV cache database setting, it will change the model input so that
        # we can skip prefilling on tokens that successfully received KV caches
        # NOTE: The receive operation is blocking
        bypass_model_exec = False
        if self.need_recv_kv(model_input, kv_caches):
            hidden_or_intermediate_states, bypass_model_exec, model_input = \
                get_kv_transfer_group().recv_kv_caches_and_hidden_states(
                    # model is used to know which layer the current worker
                    # is working on, so that we can receive KV for only those
                    # layers.
                    model_executable,
                    model_input,
                    kv_caches=kv_caches
                )

        multi_modal_kwargs = model_input.multi_modal_kwargs or {}
        seqlen_agnostic_kwargs = {
            "finished_requests_ids": model_input.finished_requests_ids,
            "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
        } if self.has_inner_state else {}
        model_kwargs = {}
        if previous_hidden_states is not None:
            model_kwargs["previous_hidden_states"] = previous_hidden_states
        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_start = torch.cuda.Event(enable_timing=True)
            model_forward_end = torch.cuda.Event(enable_timing=True)
            model_forward_start.record()

        if not bypass_model_exec:
            # todo 检查模型层是否在显存

            with set_forward_context(model_input.attn_metadata,
                                    self.vllm_config, virtual_engine):
                hidden_or_intermediate_states = model_executable(
                    input_ids=model_input.input_tokens,
                    inputs_embeds=model_input.inputs_embeds,
                    positions=model_input.input_positions,
                    intermediate_tensors=intermediate_tensors,
                    **MultiModalKwargs.as_kwargs(
                        multi_modal_kwargs,
                        device=self.device,
                    ),
                    **seqlen_agnostic_kwargs,
                    **model_kwargs,
                )

        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_end.record()

        # Sending KV cache in distributed KV cache transfer setting
        # NOTE: the send operation is non-blocking
        if self.need_send_kv(model_input, kv_caches):
            get_kv_transfer_group().send_kv_caches_and_hidden_states(
                # model_executable is used to know which layer the current
                # worker is working on, so that we can send KV for only those
                # layers.
                model_executable,
                model_input,
                kv_caches,
                hidden_or_intermediate_states,
            )

        # Compute the logits in the last pipeline stage.
        if not get_pp_group().is_last_rank:
            if (self.is_driver_worker
                    and hidden_or_intermediate_states is not None
                    and isinstance(hidden_or_intermediate_states,
                                   IntermediateTensors)
                    and self.observability_config is not None
                    and self.observability_config.collect_model_forward_time):
                model_forward_end.synchronize()
                model_forward_time = model_forward_start.elapsed_time(
                    model_forward_end)
                orig_model_forward_time = 0.0
                if intermediate_tensors is not None:
                    orig_model_forward_time = intermediate_tensors.tensors.get(
                        "model_forward_time", torch.tensor(0.0)).item()
                hidden_or_intermediate_states.tensors["model_forward_time"] = (
                    torch.tensor(model_forward_time + orig_model_forward_time))
            return hidden_or_intermediate_states

        logits = self.model.compute_logits(hidden_or_intermediate_states,
                                           model_input.sampling_metadata)

        if self.is_driver_worker:
            if model_input.async_callback is not None:
                model_input.async_callback()

            # Sample the next token.
            assert isinstance(self.sampler, Sampler)
            orig_include_gpu_probs = self.sampler.include_gpu_probs_tensor
            if model_input.inputs_embeds is not None:
                self.sampler.include_gpu_probs_tensor = True

            output: SamplerOutput = self.sampler(
                logits=logits,
                sampling_metadata=model_input.sampling_metadata,
            )
            if (self.observability_config is not None
                    and self.observability_config.collect_model_forward_time
                    and output is not None):
                model_forward_end.synchronize()
                model_forward_time = model_forward_start.elapsed_time(
                    model_forward_end)
                orig_model_forward_time = 0.0
                if intermediate_tensors is not None:
                    orig_model_forward_time = intermediate_tensors.tensors.get(
                        "model_forward_time", torch.tensor(0.0)).item()
                # If there are multiple workers, we are still tracking the
                # latency from the start time of the driver worker to the end
                # time of the driver worker. The model forward time will then
                # end up covering the communication time as well.
                output.model_forward_time = (orig_model_forward_time +
                                             model_forward_time)

        if model_input.inputs_embeds is not None:
            if self.is_driver_worker:
                sampled = broadcast_tensor_dict(
                    {"token_ids": output.sampled_token_ids})
            else:
                sampled = broadcast_tensor_dict()
            if sampled["token_ids"] is not None:
                sampled_token_embeds = self.model.get_input_embeddings(
                    sampled["token_ids"].squeeze(1))
                if self.is_driver_worker:
                    self.sampler.include_gpu_probs_tensor = \
                        orig_include_gpu_probs

                    output.sampled_token_embeds = sampled_token_embeds

                    for token_embed, sequence_group_output in zip(
                            output.sampled_token_embeds, output.outputs):
                        assert len(sequence_group_output.samples) == 1
                        sequence_group_output.samples[
                            0].output_embed = token_embed

        if not self.is_driver_worker:
            return []

        if self.return_hidden_states:
            # we only need to pass hidden states of most recent token
            assert model_input.sampling_metadata is not None
            indices = model_input.sampling_metadata.selected_token_indices
            if model_input.is_prompt:
                hidden_states = hidden_or_intermediate_states.index_select(
                    0, indices)
                output.prefill_hidden_states = hidden_or_intermediate_states
            elif decode_meta.use_cuda_graph:
                hidden_states = hidden_or_intermediate_states[:len(indices)]
            else:
                hidden_states = hidden_or_intermediate_states

            output.hidden_states = hidden_states

        return [output]
    

    def offload_weight(self):
        # todo offload
        
        self.device_cpu = torch.device('cpu')
        self.device_gpu = torch.device('cuda')

        print(f'before move memory allocated: {torch.cuda.memory_allocated() / 1024**3} GB')
        for(i, layer) in enumerate(self.model.model.layers):
            if not isinstance(layer, PPMissingLayer) and i == 5:
                print(f"current layer: {i}")
                torch.cuda.synchronize()
                layer.to(self.device_cpu)
                torch.cuda.synchronize()
                print(f"finish move")
        print(f'after move memory allocated: {torch.cuda.memory_allocated() / 1024**3} GB')

        # for(i, layer) in enumerate(self.model.model.layers):
        #     if not isinstance(layer, PPMissingLayer):
        #         print(f"current layer: {i}")
        #         layer.to(device_gpu)
        #         print(f"finish move")
        # print(f'after move back memory allocated: {torch.cuda.memory_allocated() / 1024**3} GB')

    def layer_to_GPU(self, idx):
        self.model.model.layers[idx].to(self.device_gpu)