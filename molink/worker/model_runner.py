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
from molink.model_executor.models.utils import set_cpu_offload_max_bytes
from vllm.config import VllmConfig
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.multimodal import (MULTIMODAL_REGISTRY, MultiModalRegistry)


logger = init_logger(__name__)

class MolinkGPUModelRunner(ModelRunner):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        return_hidden_states: bool = False,
        input_registry: InputRegistry = INPUT_REGISTRY,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ):        
        super().__init__(vllm_config, kv_cache_dtype, is_driver_worker, return_hidden_states, input_registry, mm_registry)
        set_cpu_offload_max_bytes(int(self.cache_config.cpu_offload_gb * 1024**3))


    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        with DeviceMemoryProfiler() as m:
            self.model = get_model(vllm_config=self.vllm_config)

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))

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
            