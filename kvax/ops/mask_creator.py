import functools
import logging

import chex
import jax
import jax.numpy as jnp
import jax_triton as jt
import numpy as np
import triton
import triton.language as tl
from jax.sharding import Mesh

from kvax.utils.common import (
    PADDING_SEGMENT_ID,
    FlashAttentionParamsConfig,
    get_default_flash_attention_params,
    pad_to_block_size,
)
from kvax.utils.sharding import shard_map
from kvax.utils.specs import Axes, get_attention_specs
from kvax.utils.typing import AttentionMask, DeviceArray, Specs

logger = logging.getLogger(__name__)


@triton.jit
def compute_attention_mask_kernel(
    outer_positions_ref,
    outer_segment_id_ref,
    inner_positions_ref,
    inner_segment_ids_ref,
    lower_block_ref,
    upper_block_ref,
    lower_full_block_ref,
    upper_full_block_ref,
    inner_block_size: tl.constexpr,
    inner_seq_len: tl.constexpr,
    outer_seq_len: tl.constexpr,
    outer_block_size: tl.constexpr,
    padding_segment_id: tl.constexpr,
    skip_pad_tokens: tl.constexpr,
    use_segment_mask: tl.constexpr,
    query_is_outer: tl.constexpr,
):
    outer_block_id = tl.program_id(0)
    batch_size_id = tl.program_id(1)
    num_outer_block_programs = tl.num_programs(0)

    # inputs
    outer_positions_ref += batch_size_id * outer_seq_len
    outer_segment_id_ref += batch_size_id * outer_seq_len
    inner_positions_ref += batch_size_id * inner_seq_len
    inner_segment_ids_ref += batch_size_id * inner_seq_len

    # outputs
    lower_block_ref += batch_size_id * num_outer_block_programs + outer_block_id
    upper_block_ref += batch_size_id * num_outer_block_programs + outer_block_id
    lower_full_block_ref += batch_size_id * num_outer_block_programs + outer_block_id
    upper_full_block_ref += batch_size_id * num_outer_block_programs + outer_block_id

    outer_arange = outer_block_id * outer_block_size + tl.arange(0, outer_block_size)

    outer_positions_block = tl.load(outer_positions_ref + outer_arange)
    outer_segments_block = tl.load(outer_segment_id_ref + outer_arange)

    outer_max_seg_id = tl.max(outer_segments_block)
    outer_min_seg_id = tl.min(outer_segments_block)
    outer_same_segment = outer_max_seg_id == outer_min_seg_id
    if skip_pad_tokens and (
        outer_same_segment and outer_min_seg_id == padding_segment_id
    ):
        tl.store(lower_block_ref, 0)
        tl.store(upper_block_ref, 0)
        tl.store(lower_full_block_ref, 0)
        tl.store(upper_full_block_ref, 0)
        return

    # Maximum position in the current query block
    # (used to determine which KV blocks to attend)
    max_outer_position = tl.max(outer_positions_block)
    min_outer_position = tl.min(outer_positions_block)

    upper_block_to_attend = 0
    lower_block_to_attend = inner_seq_len // inner_block_size

    upper_full_block = 0
    lower_full_block = inner_seq_len // inner_block_size

    for inner_idx in range(0, inner_seq_len // inner_block_size):
        inner_offset = inner_idx * inner_block_size + tl.arange(0, inner_block_size)
        inner_positions_block = tl.load(inner_positions_ref + inner_offset)
        inner_segments_block = tl.load(inner_segment_ids_ref + inner_offset)
        inner_min_seg_id = tl.min(inner_segments_block)
        inner_max_seg_id = tl.max(inner_segments_block)

        inner_same_segment = inner_max_seg_id == inner_min_seg_id

        should_attend_segments = (
            inner_min_seg_id <= outer_max_seg_id
            and outer_min_seg_id <= inner_max_seg_id
        )
        full_block_segments = outer_same_segment and inner_same_segment

        # The current query block will attend to a KV block if there is any query
        # position that is greater than or equal than any KV position in the KV block.
        min_inner_position = tl.min(inner_positions_block)
        max_inner_position = tl.max(inner_positions_block)

        # max_query >= min_kv
        if query_is_outer:
            should_attend_positions = max_outer_position >= min_inner_position
            full_block_positions = min_outer_position >= max_inner_position
        else:
            should_attend_positions = max_inner_position >= min_outer_position
            full_block_positions = min_inner_position >= max_outer_position

        if use_segment_mask:
            should_attend = should_attend_positions and should_attend_segments
        else:
            should_attend = should_attend_positions

        if skip_pad_tokens:
            is_pad_tokens = inner_min_seg_id == padding_segment_id
            should_attend = should_attend and not is_pad_tokens
        else:
            should_attend = should_attend

        should_not_attend = not should_attend

        upper_block_to_attend = tl.maximum(
            upper_block_to_attend, should_attend * (inner_idx + 1)
        )

        lower_block_to_attend = tl.minimum(
            lower_block_to_attend,
            should_attend * inner_idx + should_not_attend * lower_block_to_attend,
        )

        full_block = (full_block_segments and full_block_positions) and should_attend
        not_full_block = not full_block

        upper_full_block = tl.maximum(upper_full_block, full_block * (inner_idx + 1))

        lower_full_block = tl.minimum(
            lower_full_block,
            full_block * inner_idx + not_full_block * lower_full_block,
        )

    tl.store(lower_block_ref, lower_block_to_attend)
    tl.store(upper_block_ref, upper_block_to_attend)
    tl.store(lower_full_block_ref, lower_full_block)
    tl.store(upper_full_block_ref, upper_full_block)


def make_attention_mask_spec(dkdv_mask: bool) -> Specs:
    specs = get_attention_specs()
    kv_specs = specs.kv_specs
    query_specs = specs.query_specs
    # (batch, sharding axis, real axis)
    if dkdv_mask:
        out_spec = (
            query_specs[Axes.batch],
            query_specs[Axes.query_sequence],
            kv_specs[Axes.kv_sequence],
        )
    else:
        out_spec = (
            query_specs[Axes.batch],
            kv_specs[Axes.kv_sequence],
            query_specs[Axes.query_sequence],
        )

    return (
        out_spec,
        out_spec,
        out_spec,
        out_spec,
    )


def make_attention_mask_specs(
    dkdv_mask: bool = False,
) -> tuple[tuple[Specs, ...], tuple[Specs, ...]]:
    specs = get_attention_specs()
    kv_specs = specs.kv_specs
    query_specs = specs.query_specs

    in_specs = (
        (query_specs[Axes.batch], query_specs[Axes.query_sequence]),  # query_positions
        (
            query_specs[Axes.batch],
            query_specs[Axes.query_sequence],
        ),  # query_segment_ids
        (kv_specs[Axes.batch], kv_specs[Axes.kv_sequence]),  # kv_positions
        (kv_specs[Axes.batch], kv_specs[Axes.kv_sequence]),  # kv_segment_ids
    )
    out_specs = make_attention_mask_spec(dkdv_mask=dkdv_mask)

    return in_specs, out_specs


def compute_attention_mask_single_device(
    query_positions: DeviceArray,
    query_segment_ids: DeviceArray,
    kv_positions: DeviceArray,
    kv_segment_ids: DeviceArray,
    kv_block_size: int,
    query_block_size: int,
    skip_pad_tokens: bool = True,
    calculate_dkdv_mask: bool = False,
) -> AttentionMask:
    _, query_positions, query_segment_ids = pad_to_block_size(
        embeddings=None,
        positions=query_positions,
        segment_ids=query_segment_ids,
        block_size=query_block_size,
        pos_fill_value=-1,  # Any filler should work
    )
    _, kv_positions, kv_segment_ids = pad_to_block_size(
        embeddings=None,
        positions=kv_positions,
        segment_ids=kv_segment_ids,
        block_size=kv_block_size,
        pos_fill_value=jnp.iinfo(jnp.int32).max,  # These should not be attended to
    )

    batch_size, query_seq_len = query_positions.shape
    _, kv_seq_len = kv_positions.shape

    chex.assert_shape([kv_positions, kv_segment_ids], [batch_size, kv_seq_len])
    num_query_blocks = jt.cdiv(query_seq_len, query_block_size)
    num_kv_blocks = jt.cdiv(kv_seq_len, kv_block_size)

    if calculate_dkdv_mask:
        # The secon is the sharding axis.
        # It uses only to shard mask between GPUs during CP.
        output_shape = jax.ShapeDtypeStruct(
            shape=(batch_size, 1, num_kv_blocks),
            dtype=jnp.int32,
        )
    else:
        output_shape = jax.ShapeDtypeStruct(
            shape=(batch_size, 1, num_query_blocks),
            dtype=jnp.int32,
        )

    # Replace all -1 (PADDING_SEGMENT_ID) by kv_seq_len + 1 to make the all logic work
    INNER_PADDING_SEGMENT_ID = kv_seq_len + 1

    query_segment_ids = jnp.where(
        query_segment_ids == PADDING_SEGMENT_ID,
        INNER_PADDING_SEGMENT_ID,
        query_segment_ids,
    )
    kv_segment_ids = jnp.where(
        kv_segment_ids == PADDING_SEGMENT_ID,
        INNER_PADDING_SEGMENT_ID,
        kv_segment_ids,
    )

    common_params = dict(
        kernel=compute_attention_mask_kernel,
        out_shape=(
            output_shape,  # lower_kv_block_to_attend
            output_shape,  # upper_kv_block_to_attend
            output_shape,  # lower_full_kv_block_to_attend
            output_shape,  # upper_full_kv_block_to_attend
        ),
        debug=False,
        padding_segment_id=INNER_PADDING_SEGMENT_ID,
        skip_pad_tokens=skip_pad_tokens,
        use_segment_mask=True,
        num_warps=2,
        num_stages=4,
    )

    if calculate_dkdv_mask:
        # outer loop: kv
        # inner loop: query
        results = jt.triton_call(
            kv_positions,
            kv_segment_ids,
            query_positions,
            query_segment_ids,
            inner_block_size=query_block_size,
            outer_block_size=kv_block_size,
            inner_seq_len=query_seq_len,
            outer_seq_len=kv_seq_len,
            query_is_outer=False,
            grid=(num_kv_blocks, batch_size),
            **common_params,
        )
    else:
        # outer loop: query
        # inner loop: kv
        results = jt.triton_call(
            query_positions,
            query_segment_ids,
            kv_positions,
            kv_segment_ids,
            inner_block_size=kv_block_size,
            outer_block_size=query_block_size,
            inner_seq_len=kv_seq_len,
            outer_seq_len=query_seq_len,
            query_is_outer=True,
            grid=(num_query_blocks, batch_size),
            **common_params,
        )

    (
        lower_block,
        upper_block,
        lower_full_block,
        upper_full_block,
    ) = results

    # Postprocessing
    # Work around the case when wasn't found a block to process
    if calculate_dkdv_mask:
        lower_block = jnp.where(
            lower_block == num_query_blocks, upper_block, lower_block
        )
        lower_full_block = jnp.where(
            lower_full_block == num_query_blocks, upper_block, lower_full_block
        )
    else:
        lower_block = jnp.where(lower_block == num_kv_blocks, upper_block, lower_block)
        lower_full_block = jnp.where(
            lower_full_block == num_kv_blocks, lower_block, lower_full_block
        )
    upper_full_block = jnp.where(
        upper_full_block < lower_full_block, lower_full_block, upper_full_block
    )

    return (lower_block, upper_block, lower_full_block, upper_full_block)


def compute_attention_mask(
    query_positions: DeviceArray,
    query_segment_ids: DeviceArray,
    kv_positions: DeviceArray,
    kv_segment_ids: DeviceArray,
    kv_block_size: int,
    query_block_size: int,
    skip_pad_tokens: bool = True,
    calculate_dkdv_mask: bool = False,
    mesh: Mesh | None = None,
):
    in_specs, out_specs = make_attention_mask_specs(calculate_dkdv_mask)

    compute_attention_mask_fn = functools.partial(
        compute_attention_mask_single_device,
        kv_block_size=kv_block_size,
        query_block_size=query_block_size,
        skip_pad_tokens=skip_pad_tokens,
        calculate_dkdv_mask=calculate_dkdv_mask,
    )

    compute_attention_mask_fn = shard_map(
        compute_attention_mask_fn,
        in_specs=in_specs,
        out_specs=out_specs,
        mesh=mesh,
    )

    flatten_mask = compute_attention_mask_fn(
        query_positions,
        query_segment_ids,
        kv_positions,
        kv_segment_ids,
    )

    attn_mask = AttentionMask.unflatten(flatten_mask)
    return attn_mask


def create_attention_mask(
    query_positions: DeviceArray,
    query_segment_ids: DeviceArray,
    kv_positions: DeviceArray,
    kv_segment_ids: DeviceArray,
    fwd_params: FlashAttentionParamsConfig | None = None,
    bwd_params: FlashAttentionParamsConfig | None = None,
    calc_bwd_mask: bool = False,
    skip_pad_tokens: bool = True,
    mesh: Mesh | None = None,
) -> tuple[AttentionMask, ...]:
    """
    Creates attention masks for forward and (optionally) backward flash attention
    kernels.

    This function generates the required attention masks based on the query and
    key-value (KV) positions and segment ids. The masks are used for both
    the forward and backward passes in flash attention to improve computational
    efficiency while respecting segment boundaries.

    Args:
        query_positions (DeviceArray): The positions of the query tokens of shape:
            (batch_size, query_seq_length).
        query_segment_ids (DeviceArray): Segment ids for query tokens of shape:
            (batch_size, query_seq_length).
        kv_positions (DeviceArray): The positions of the key and value tokens of shape:
            (batch_size, kv_seq_length).
        kv_segment_ids (DeviceArray): Segment ids for key abd value tokens of shape:
            (batch_size, kv_seq_length).
        fwd_params (FlashAttentionParamsConfig | None, optional): Parameters for the
            forward pass of the flash attention kernel. Defaults to parameters defined
            via `get_default_flash_attention_params(backward=False)`.
        bwd_params (FlashAttentionParamsConfig | None, optional): Parameters for the
            backward pass of the flash attention kernel. Defaults to parameters defined
            via `get_default_flash_attention_params(backward=True)`.
        calc_bwd_mask (bool, optional): Whether to calculate the backward attention
            masks. Defaults is False.
        skip_pad_tokens (bool, optional): Whether to skip padding tokens in the
            attention mask. Defaults is True.
        mesh (Mesh | None, optional): Device mesh configuration for distributed
            execution. If None, it takes the mesh from the global context.
            Defaults is None.

    Returns:
        tuple[AttentionMask, ...]: A tuple containing:
            - The forward attention mask.
            - (Optional) The backward mask for dquery (if `calc_bwd_mask` is True).
            - (Optional) The backward mask for dkey and dvalue
                (if `calc_bwd_mask` is True).

    Notes:
        - If `calc_bwd_mask` is True, masks for dquery, dkey, and dvalue are computed.
        - Defaults for `fwd_params` and `bwd_params` are set using
            `get_default_flash_attention_params`.

    """
    # Define default parameters for flash attention if not provided.
    if fwd_params is None:
        fwd_params = get_default_flash_attention_params(backward=False)

    # Compute attention mask for forward kernel
    mask_fwd = compute_attention_mask(
        query_positions,
        query_segment_ids,
        kv_positions,
        kv_segment_ids,
        kv_block_size=fwd_params.kv_block_size,
        query_block_size=fwd_params.query_block_size,
        skip_pad_tokens=skip_pad_tokens,
        mesh=mesh,
    )
    outputs = [
        mask_fwd,
    ]

    # Compute attention mask for backward kernel if needed
    if calc_bwd_mask:
        if bwd_params is None:
            bwd_params = get_default_flash_attention_params(backward=True)

        query_block_size_dkdv = bwd_params.query_block_size
        kv_block_size_dkdv = bwd_params.kv_block_size
        query_block_size_dq = bwd_params.kv_block_size
        kv_block_size_dq = bwd_params.query_block_size

        # Compute attention mask for calculation dquery
        mask_dq = compute_attention_mask(
            query_positions,
            query_segment_ids,
            kv_positions,
            kv_segment_ids,
            kv_block_size=kv_block_size_dq,
            query_block_size=query_block_size_dq,
            skip_pad_tokens=skip_pad_tokens,
            mesh=mesh,
        )

        # Compute attention mask for calculation dkey and dvalue
        mask_dkdv = compute_attention_mask(
            query_positions,
            query_segment_ids,
            kv_positions,
            kv_segment_ids,
            kv_block_size=kv_block_size_dkdv,
            query_block_size=query_block_size_dkdv,
            skip_pad_tokens=skip_pad_tokens,
            calculate_dkdv_mask=True,
            mesh=mesh,
        )

        outputs.append(mask_dq)
        outputs.append(mask_dkdv)

    return tuple(outputs)


def print_mask(
    mask: AttentionMask,
    num_inner_blocks: int,
    print_only_first_in_batch: bool = True,
    fit_in_screen: bool = True,
) -> None:
    batch_size = mask.lower_bounds.shape[0]

    masks_range = 1 if print_only_first_in_batch else batch_size
    MAX_NUM_ROWS = 32
    MAX_NUM_COLUMNS = 64

    def _build_mask(
        lower: DeviceArray,
        lower_full: DeviceArray,
        upper_full: DeviceArray,
        upper: DeviceArray,
        num_inner_blocks: int,
    ) -> np.array:
        num_shards = lower.shape[0]
        num_rows = lower.shape[-1]

        mask = np.zeros((num_rows, num_inner_blocks), dtype=np.int32)
        for row in range(num_rows):
            for shard in range(num_shards):
                shift = shard * num_inner_blocks // num_shards
                mask[row, shift : shift + lower[shard, row]] = 0
                mask[
                    row, shift + lower[shard, row] : shift + lower_full[shard, row]
                ] = 1
                mask[
                    row, shift + lower_full[shard, row] : shift + upper_full[shard, row]
                ] = 2
                mask[
                    row, shift + upper_full[shard, row] : shift + upper[shard, row]
                ] = 1
                mask[
                    row,
                    shift + upper[shard, row] : shift + num_inner_blocks // num_shards,
                ] = 0
        return mask

    def _average_mask_blocks(
        mask_np: np.array,
        row_step: int,
        col_step: int,
    ) -> np.array:
        pad_rows = (row_step - mask_np.shape[0] % row_step) % row_step
        pad_cols = (col_step - mask_np.shape[1] % col_step) % col_step

        padded_mask = np.pad(
            mask_np, ((0, pad_rows), (0, pad_cols)), mode="constant", constant_values=0
        )

        reshaped = padded_mask.reshape(
            padded_mask.shape[0] // row_step,
            row_step,
            padded_mask.shape[1] // col_step,
            col_step,
        )
        result_mask = reshaped.sum(axis=(1, 3)) / (col_step * row_step)
        return result_mask

    def _mask_to_list_of_string(
        mask_np: np.array,
    ) -> list[str]:
        lines = []
        for row in mask_np:
            tmp_line = ""
            for item in row:
                if item == 2:
                    tmp_line += "██"
                elif item == 0:
                    tmp_line += "  "
                else:
                    tmp_line += "░░"
            lines.append(tmp_line)
        return lines

    final_str = "Attention masks:\n"
    for i in range(masks_range):
        lower = mask.lower_bounds[i, :, :]
        lower_full = mask.lower_full_bounds[i, :, :]
        upper_full = mask.upper_full_bounds[i, :, :]
        upper = mask.upper_bounds[i, :, :]

        num_rows = lower.shape[-1]
        mask_np = _build_mask(lower, lower_full, upper_full, upper, num_inner_blocks)

        columns_step = np.maximum(
            np.ceil(num_inner_blocks / MAX_NUM_COLUMNS), 1
        ).astype(np.int32)
        rows_step = np.maximum(np.ceil(num_rows / MAX_NUM_ROWS), 1).astype(np.int32)
        if fit_in_screen and (columns_step > 1 or rows_step > 1):
            mask_np = _average_mask_blocks(mask_np, rows_step, columns_step)

        lines = _mask_to_list_of_string(mask_np)
        num_inner_blocks_to_display = len(lines[0]) // 2

        # Print mask
        final_str += "\n  " + "==" * num_inner_blocks_to_display + "  \n"
        for line in lines:
            final_str += "||" + line + "||\n"
        final_str += "  " + "==" * num_inner_blocks_to_display + "  \n\n"

    logger.info(final_str)


def calculate_mask_sparsity(
    mask: AttentionMask,
    query_block_size: int,
    kv_block_size: int,
    query_seq_len: int,
    kv_seq_len: int,
) -> float:
    batch_size = mask.lower_bounds.shape[0]
    num_blocks_to_calculate = 0
    for i in range(batch_size):
        lower = mask.lower_bounds[i, :, :]
        upper = mask.upper_bounds[i, :, :]

        num_shards = lower.shape[0]
        num_rows = lower.shape[-1]

        for row in range(num_rows):
            for shard in range(num_shards):
                num_blocks_to_calculate += upper[shard, row] - lower[shard, row]

    num_blocks_to_calculate /= query_seq_len // query_block_size
    num_blocks_to_calculate /= kv_seq_len // kv_block_size
    num_blocks_to_calculate /= batch_size

    return num_blocks_to_calculate
