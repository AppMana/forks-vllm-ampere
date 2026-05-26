# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_events import (
    BlockStored,
    KVCacheEvent,
    KVConnectorKVEvents,
    KVEventAggregator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _deepseek_mtp_draft_layers(vllm_config: "VllmConfig") -> int | None:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is None:
        return None

    method = getattr(speculative_config, "method", None)
    if method not in ("deepseek_mtp", "mtp"):
        return None

    model_config = vllm_config.model_config
    return int(
        getattr(speculative_config, "num_speculative_tokens", 0)
        or getattr(model_config.hf_config, "num_nextn_predict_layers", 0)
        or 0
    )


def _patch_lmcache_draft_layers() -> None:
    """Keep packaged LMCache metadata aligned with native DSV4 MTP."""
    try:
        from lmcache.integration.vllm import utils as lmcache_vllm_utils
    except ImportError:
        return

    original = lmcache_vllm_utils.calculate_draft_layers
    if getattr(original, "_vllm_dsv4_mtp_patch", False):
        return

    def calculate_draft_layers(vllm_config: "VllmConfig") -> int:
        mtp_layers = _deepseek_mtp_draft_layers(vllm_config)
        if mtp_layers is not None:
            return mtp_layers
        return original(vllm_config)

    calculate_draft_layers._vllm_dsv4_mtp_patch = True  # type: ignore[attr-defined]
    lmcache_vllm_utils.calculate_draft_layers = calculate_draft_layers


def _patch_lmcache_v3_grouped_transfers() -> None:
    """Use LMCache's block-level grouped copy path for mixed DSV4 KV groups."""
    try:
        from lmcache.integration.vllm.utils import get_size_bytes
        from lmcache.v1.protocol import (
            get_remote_metadata_bytes,
            init_remote_metadata_info,
        )
        from lmcache.v1.storage_backend.connector import base_connector
        from lmcache.v1.gpu_connector import gpu_connectors
        from lmcache.v1.memory_management import MemoryFormat
        from lmcache.v1.metadata import LMCacheMetadata
        import lmcache.c_ops as lmc_ops
    except ImportError:
        return

    remote_connector_cls = getattr(base_connector, "RemoteConnector", None)
    if remote_connector_cls is not None and not getattr(
        remote_connector_cls, "_vllm_dsv4_grouped_mla_init_patch", False
    ):
        original_remote_init = remote_connector_cls.__init__

        def remote_init(self, config, metadata):
            try:
                return original_remote_init(self, config, metadata)
            except AssertionError:
                if metadata is None or not getattr(metadata, "use_mla", False):
                    raise
                manager = getattr(metadata, "kv_layer_groups_manager", None)
                groups = getattr(manager, "kv_layer_groups", None)
                if not groups:
                    raise
                shapes = metadata.get_shapes()
                dtypes = metadata.get_dtypes()
                full_chunk_size_bytes = get_size_bytes(shapes, dtypes)
                if full_chunk_size_bytes % metadata.chunk_size == 0:
                    raise

                self.save_chunk_meta = (
                    config.extra_config is None
                    or config.extra_config.get("save_chunk_meta", True)
                    or config.use_layerwise
                )
                self.meta_shapes = shapes
                self.meta_dtypes = dtypes
                self.meta_fmt = MemoryFormat.KV_MLA_FMT
                self.full_chunk_size_bytes = full_chunk_size_bytes
                self.single_token_size = max(
                    1,
                    (full_chunk_size_bytes + metadata.chunk_size - 1)
                    // metadata.chunk_size,
                )

                init_remote_metadata_info(metadata.get_num_groups())
                self.remote_metadata_bytes = get_remote_metadata_bytes()
                logger.info(
                    "Initialized LMCache grouped MLA remote connector with "
                    "non-uniform logical token bytes: shapes=%s, dtypes=%s, "
                    "full chunk size=%s, approximate single token size=%s, "
                    "remote metadata bytes=%s",
                    self.meta_shapes,
                    self.meta_dtypes,
                    self.full_chunk_size_bytes,
                    self.single_token_size,
                    self.remote_metadata_bytes,
                )

        remote_connector_cls.__init__ = remote_init
        remote_connector_cls._vllm_dsv4_grouped_mla_init_patch = True

    if not getattr(LMCacheMetadata, "_vllm_dsv4_physical_shapes_patch", False):
        original_get_shapes = LMCacheMetadata.get_shapes

        def get_shapes(self, num_tokens: int | None = None) -> list[torch.Size]:
            if num_tokens is None:
                num_tokens = self.chunk_size
            manager = self.kv_layer_groups_manager
            if manager is None or not manager.kv_layer_groups:
                return original_get_shapes(self, num_tokens)

            shapes: list[torch.Size] = []
            for group in manager.kv_layer_groups:
                compress_ratio = getattr(group, "compress_ratio", 1)
                if compress_ratio <= 1:
                    group_tokens = num_tokens
                else:
                    group_tokens = (num_tokens + compress_ratio - 1) // compress_ratio
                shapes.append(
                    torch.Size(
                        [
                            group.shape_desc.kv_size,
                            group.num_layers,
                            group_tokens,
                            group.hidden_dim_size,
                        ]
                    )
                )
            return shapes

        LMCacheMetadata.get_shapes = get_shapes
        LMCacheMetadata._vllm_dsv4_physical_shapes_patch = True

    connector_cls = getattr(gpu_connectors, "VLLMPagedMemGPUConnectorV3", None)
    if connector_cls is None or getattr(
        connector_cls, "_vllm_dsv4_grouped_transfer_patch", False
    ):
        return

    original_from_gpu = connector_cls.from_gpu
    original_to_gpu = connector_cls.to_gpu

    def _heterogeneous_group_manager(connector: object) -> object | None:
        manager = getattr(connector.metadata, "kv_layer_groups_manager", None)
        if manager is None:
            return None
        groups = getattr(manager, "kv_layer_groups", ())
        block_sizes = {group.shape_desc.bs for group in groups}
        return manager if len(block_sizes) > 1 else None

    def _logical_block_ids(
        connector: object,
        slot_mapping: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        manager = getattr(connector.metadata, "kv_layer_groups_manager")
        logical_block_size = manager.inference_engine_logical_block_size
        if (end - start) % logical_block_size != 0:
            raise ValueError(
                "LMCache V3 grouped transfer requires chunk-aligned slots: "
                f"start={start}, end={end}, block_size={logical_block_size}"
            )
        chunk_slots = slot_mapping[start:end:logical_block_size]
        return torch.div(
            chunk_slots, logical_block_size, rounding_mode="floor"
        ).to(dtype=torch.long).contiguous()

    def _group_tmp_buffer(connector: object, group_idx: int, token_count: int):
        tmp_buffers = getattr(connector, "group_tmp_buffer", None)
        if tmp_buffers is None:
            raise RuntimeError(
                "LMCache V3 grouped transfer requires GPU temporary buffers"
            )
        return tmp_buffers[group_idx][:, :, :token_count, :]

    def _physical_group_tensor(
        memory_obj_tensor: torch.Tensor, physical_chunk_size: int
    ) -> torch.Tensor:
        if memory_obj_tensor.shape[2] < physical_chunk_size:
            raise RuntimeError(
                "LMCache grouped memory object is smaller than the physical "
                f"chunk: shape={tuple(memory_obj_tensor.shape)}, "
                f"physical_chunk_size={physical_chunk_size}"
            )
        return memory_obj_tensor[:, :, :physical_chunk_size, :]

    def from_gpu(self, memory_obj, start: int, end: int, **kwargs):
        assert "slot_mapping" in kwargs
        self.initialize_kvcaches_ptr(**kwargs)
        self._initialize_kv_cache_pointers()
        manager = _heterogeneous_group_manager(self)
        if manager is None:
            return original_from_gpu(self, memory_obj, start, end, **kwargs)

        slot_mapping = kwargs["slot_mapping"]
        block_ids = _logical_block_ids(self, slot_mapping, start, end)
        with torch.cuda.stream(self.store_stream):
            for group_idx, kv_cache_pointer in enumerate(
                self.group_kv_cache_pointers_on_gpu
            ):
                physical_chunk_size = manager.get_physical_chunk_size(group_idx)
                tmp_gpu_buffer = _group_tmp_buffer(
                    self, group_idx, physical_chunk_size
                )
                lmc_ops.multi_layer_block_kv_transfer(
                    kv_cache_pointer,
                    [tmp_gpu_buffer.data_ptr()],
                    block_ids,
                    self.device,
                    lmc_ops.TransferDirection.D2H,
                    manager.get_shape_desc(group_idx),
                    physical_chunk_size,
                    self.gpu_kv_format,
                    0,
                )
                memory_obj_tensor = memory_obj.get_tensor(group_idx)
                assert memory_obj_tensor is not None
                _physical_group_tensor(
                    memory_obj_tensor, physical_chunk_size
                ).copy_(tmp_gpu_buffer, non_blocking=True)

        if not memory_obj.raw_tensor.is_cuda:
            self.store_stream.synchronize()
        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def to_gpu(self, memory_obj, start: int, end: int, **kwargs):
        assert "slot_mapping" in kwargs
        self.initialize_kvcaches_ptr(**kwargs)
        self._initialize_kv_cache_pointers()
        manager = _heterogeneous_group_manager(self)
        if manager is None:
            return original_to_gpu(self, memory_obj, start, end, **kwargs)

        if self.use_mla:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_MLA_FMT
        block_ids = _logical_block_ids(self, kwargs["slot_mapping"], start, end)
        logical_block_size = manager.inference_engine_logical_block_size
        vllm_cached = kwargs.get("vllm_cached_tokens", 0)
        skip_prefix_n_tokens = min(end - start, max(0, vllm_cached - start))
        skip_blocks_in_chunk = skip_prefix_n_tokens // logical_block_size

        for group_idx, kv_cache_pointer in enumerate(
            self.group_kv_cache_pointers_on_gpu
        ):
            physical_chunk_size = manager.get_physical_chunk_size(group_idx)
            tmp_gpu_buffer = _group_tmp_buffer(self, group_idx, physical_chunk_size)
            memory_obj_tensor = memory_obj.get_tensor(group_idx)
            assert memory_obj_tensor is not None
            tmp_gpu_buffer.copy_(
                _physical_group_tensor(memory_obj_tensor, physical_chunk_size),
                non_blocking=True,
            )
            lmc_ops.multi_layer_block_kv_transfer(
                kv_cache_pointer,
                [tmp_gpu_buffer.data_ptr()],
                block_ids,
                self.device,
                lmc_ops.TransferDirection.H2D,
                manager.get_shape_desc(group_idx),
                physical_chunk_size,
                self.gpu_kv_format,
                skip_blocks_in_chunk,
            )

    connector_cls.from_gpu = from_gpu
    connector_cls.to_gpu = to_gpu
    connector_cls._vllm_dsv4_grouped_transfer_patch = True


def _lmcache_expected_kv_cache_count(vllm_config: "VllmConfig") -> int:
    base_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
    draft_layers = _deepseek_mtp_draft_layers(vllm_config) or 0
    return base_layers + draft_layers


def _filter_lmcache_kv_caches(
    kv_caches: dict[str, torch.Tensor], vllm_config: "VllmConfig"
) -> dict[str, torch.Tensor]:
    expected = _lmcache_expected_kv_cache_count(vllm_config)
    if expected <= 0 or len(kv_caches) <= expected:
        return kv_caches

    filtered_items = list(kv_caches.items())[:expected]
    dropped = list(kv_caches.keys())[expected:]
    logger.warning(
        "LMCache registered %d KV tensors but allocated space for %d on this "
        "PP rank; dropping trailing tensors from LMCache registration: %s",
        len(kv_caches),
        expected,
        dropped,
    )
    return dict(filtered_items)


def _lmcache_grouped_gpu_connector(lmcache_impl: object) -> object | None:
    lmcache_engine = getattr(lmcache_impl, "lmcache_engine", None)
    gpu_connector = getattr(lmcache_engine, "gpu_connector", None)
    if gpu_connector is None:
        return None
    if hasattr(gpu_connector, "_initialize_kv_cache_pointers"):
        return gpu_connector
    return None


def _prime_lmcache_grouped_metadata(
    lmcache_impl: object, kv_caches: dict[str, torch.Tensor]
) -> None:
    gpu_connector = _lmcache_grouped_gpu_connector(lmcache_impl)
    if gpu_connector is None:
        return

    kvcaches = list(kv_caches.values())
    gpu_connector.initialize_kvcaches_ptr(kvcaches=kvcaches)
    gpu_connector._initialize_kv_cache_pointers()
    _repair_lmcache_grouped_compression_metadata(gpu_connector)


def _repair_lmcache_grouped_compression_metadata(gpu_connector: object) -> None:
    metadata = getattr(gpu_connector, "metadata", None)
    manager = getattr(metadata, "kv_layer_groups_manager", None)
    groups = tuple(getattr(manager, "kv_layer_groups", ()))
    if not groups:
        return

    block_sizes = {group.shape_desc.bs for group in groups}
    if len(block_sizes) <= 1:
        return
    if any(getattr(group, "compress_ratio", 1) != 1 for group in groups):
        return

    layout_hints = getattr(gpu_connector, "layout_hints", None)
    if not isinstance(layout_hints, dict):
        return

    logical_block_size = max(block_sizes)
    layout_hints["inference_engine_logical_block_size"] = logical_block_size
    manager_factory = getattr(gpu_connector, "_kv_layer_groups_manager_factory", None)
    if manager_factory is None:
        try:
            from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
        except ImportError:
            return
        manager_factory = KVLayerGroupsManager

    kvcaches = getattr(gpu_connector, "kvcaches", None)
    gpu_kv_format = getattr(gpu_connector, "gpu_kv_format", None)
    num_blocks = getattr(gpu_connector, "num_blocks", None)
    chunk_size = getattr(gpu_connector, "chunk_size", 256)
    if kvcaches is None or gpu_kv_format is None or not isinstance(num_blocks, int):
        return

    rebuilt_manager = manager_factory(
        kvcaches,
        gpu_kv_format=gpu_kv_format,
        num_blocks=num_blocks,
        layout_hints=layout_hints,
        lmcache_logical_chunk_size=chunk_size,
    )
    setattr(metadata, "kv_layer_groups_manager", rebuilt_manager)
    if hasattr(gpu_connector, "init"):
        setattr(gpu_connector, "init", False)

    logger.info(
        "Rebuilding LMCache grouped KV metadata with inferred logical block "
        "size %d from heterogeneous physical block sizes %s",
        logical_block_size,
        sorted(block_sizes),
    )
    gpu_connector._initialize_kv_cache_pointers()


def _register_lmcache_grouped_kv_caches(
    lmcache_impl: object, kv_caches: dict[str, torch.Tensor]
) -> bool:
    if _lmcache_grouped_gpu_connector(lmcache_impl) is None:
        return False
    if not hasattr(lmcache_impl, "kv_caches"):
        return False

    registered_kv_caches = getattr(lmcache_impl, "kv_caches")
    assert len(registered_kv_caches) == 0 and len(kv_caches) > 0
    setattr(lmcache_impl, "kv_caches", kv_caches)

    # The packaged LMCache adapter initializes storage metadata during
    # post_init(). DSV4 sparse MLA needs grouped metadata available before
    # that point so remote Redis chunks use the same heterogeneous layout as
    # the GPU connector.
    _prime_lmcache_grouped_metadata(lmcache_impl, kv_caches)

    lmcache_engine = getattr(lmcache_impl, "lmcache_engine", None)
    if lmcache_engine is not None:
        lmcache_engine.post_init(kvcaches=list(kv_caches.values()))
    return True


class LMCacheKVEvents(KVConnectorKVEvents):
    """
    Concrete implementation of KVConnectorKVEvents using KVEventAggregator.
    """

    def __init__(self, num_workers: int) -> None:
        self._aggregator = KVEventAggregator(num_workers)

    def add_events(self, events: list[KVCacheEvent]) -> None:
        self._aggregator.add_events(events)

    def aggregate(self) -> "LMCacheKVEvents":
        """
        Aggregate KV events and retain only common events.
        """
        common_events = self._aggregator.get_common_events()
        self._aggregator.clear_events()
        self._aggregator.add_events(common_events)
        self._aggregator.reset_workers()
        return self

    def increment_workers(self, count: int = 1) -> None:
        self._aggregator.increment_workers(count)

    def get_all_events(self) -> list[KVCacheEvent]:
        return self._aggregator.get_all_events()

    def get_number_of_workers(self) -> int:
        return self._aggregator.get_number_of_workers()

    def clear_events(self) -> None:
        self._aggregator.clear_events()
        self._aggregator.reset_workers()

    def __repr__(self) -> str:
        return f"<LMCacheKVEvents events={self.get_all_events()}>"


class LMCacheConnectorV1(KVConnectorBase_V1):
    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        """
        LMCache requires PIECEWISE CUDA graph mode when layerwise
        operations are enabled. The wait_for_layer_load and save_kv_layer
        methods perform actual async synchronization that cannot be
        captured in CUDA graphs.
        """
        return extra_config.get("use_layerwise", False)

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        assert vllm_config.kv_transfer_config is not None
        use_native = vllm_config.kv_transfer_config.get_from_extra_config(
            "use_native", False
        )
        if use_native:
            logger.info("Initializing native LMCache connector")
            # lazy import
            from vllm.distributed.kv_transfer.kv_connector.v1 import lmcache_integration

            _adapter = lmcache_integration.vllm_v1_adapter

            cls = _adapter.LMCacheConnectorV1Impl
        else:
            logger.info("Initializing latest dev LMCache connector")
            _patch_lmcache_draft_layers()
            _patch_lmcache_v3_grouped_transfers()
            # lazy import
            from lmcache.integration.vllm.vllm_v1_adapter import (
                LMCacheConnectorV1Impl as LMCacheConnectorLatestImpl,
            )

            cls = LMCacheConnectorLatestImpl

        self._lmcache_engine = cls(vllm_config, role, self)

        self._kv_cache_events: LMCacheKVEvents | None = None

    # ==============================
    # Worker-side methods
    # ==============================
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args:
            kv_caches: dictionary of layer names, kv cache
        """
        if _register_lmcache_grouped_kv_caches(self._lmcache_engine, kv_caches):
            return

        if _lmcache_grouped_gpu_connector(self._lmcache_engine) is None:
            kv_caches = _filter_lmcache_kv_caches(kv_caches, self._vllm_config)
        if hasattr(self._lmcache_engine, "register_kv_caches"):
            self._lmcache_engine.register_kv_caches(kv_caches)
            _prime_lmcache_grouped_metadata(self._lmcache_engine, kv_caches)
        else:
            logger.warning(
                "LMCache engine does not support register_kv_caches, "
                "please check and use the latest version"
            )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        self._lmcache_engine.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        self._lmcache_engine.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """
        Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        self._lmcache_engine.save_kv_layer(
            layer_name, kv_layer, attn_metadata, **kwargs
        )

    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        self._lmcache_engine.wait_for_save()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        return self._lmcache_engine.get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        Returns:
            Set of block IDs that encountered load errors.
            Empty set if no load errors occurred.
        """
        method = getattr(self._lmcache_engine, "get_block_ids_with_load_errors", None)
        if callable(method):
            return method()

        # Fallback for older versions that don't support this method
        return set()

    def get_kv_connector_kv_cache_events(self) -> LMCacheKVEvents | None:
        """
        Get the KV connector kv cache events collected during the last interval.
        """

        events = self._lmcache_engine.get_kv_events()  # type: ignore [attr-defined]
        if not events:
            return None

        blocks: list[BlockStored] = [
            BlockStored(
                block_hashes=e.block_hashes,
                parent_block_hash=e.parent_block_hash,
                token_ids=e.token_ids,
                lora_id=e.lora_id,
                block_size=e.block_size,
                medium=e.medium,
                lora_name=getattr(e, "lora_name", None),
            )
            for e in events
        ]

        lmcache_kv_events = LMCacheKVEvents(num_workers=1)
        lmcache_kv_events.add_events(blocks)
        return lmcache_kv_events

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        return self._lmcache_engine.get_num_new_matched_tokens(
            request, num_computed_tokens
        ), False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.
        """
        self._lmcache_engine.update_state_after_alloc(request, num_external_tokens)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        return self._lmcache_engine.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        # Get the KV events
        kv_cache_events = connector_output.kv_cache_events
        if not kv_cache_events or not isinstance(kv_cache_events, LMCacheKVEvents):
            return

        if self._kv_cache_events is None:
            self._kv_cache_events = kv_cache_events
        else:
            self._kv_cache_events.add_events(kv_cache_events.get_all_events())
            self._kv_cache_events.increment_workers(
                kv_cache_events.get_number_of_workers()
            )
        return

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        return self._lmcache_engine.request_finished(request, block_ids)

    def take_events(self) -> Iterable["KVCacheEvent"]:
        """
        Take the KV cache events from the connector.

        Yields:
            New KV cache events since the last call.
        """
        if self._kv_cache_events is not None:
            self._kv_cache_events.aggregate()
            kv_cache_events = self._kv_cache_events.get_all_events()
            yield from kv_cache_events
            self._kv_cache_events.clear_events()
            self._kv_cache_events = None
