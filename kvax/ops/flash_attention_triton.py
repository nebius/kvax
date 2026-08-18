import functools
import logging
from typing import Sequence

import chex
import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh

from kvax.ops.mask_creator import make_attention_mask_spec
from kvax.utils.common import (
    FlashAttentionParamsConfig,
    get_default_flash_attention_params,
    pad_to_block_size,
)
from kvax.utils.permutation import unpermute_tokens_context_parallelism
from kvax.utils.sharding import get_query_context_mesh_axis_name, shard_map
from kvax.utils.specs import Axes, get_attention_specs
from kvax.utils.typing import AttentionMask, DeviceArray, Specs

logger = logging.getLogger(__name__)

LOG2_CONST = tl.constexpr(1.4426950408889634)  # = 1.0 / ln(2)
NEG_INF = tl.constexpr(jnp.iinfo(jnp.int32).min)


@triton.jit
def make_segment_mask(
    query_segment_ids,
    kv_segment_ids,
    transposed: tl.constexpr,
):
    if transposed:
        res = query_segment_ids[None, :] == kv_segment_ids[:, None]
    else:
        res = query_segment_ids[:, None] == kv_segment_ids[None, :]
    return res


@triton.jit
def make_causal_mask(
    query_positions,
    kv_positions,
    transposed: tl.constexpr,
):
    if transposed:
        causal_mask = query_positions[None, :] >= kv_positions[:, None]
    else:
        causal_mask = query_positions[:, None] >= kv_positions[None, :]
    return causal_mask


@triton.jit
def _flash_attention_forward_kernel_inner(
    acc,
    l,
    m,
    query_block,
    query_segment_ids,
    query_positions_block,
    key_transpose_block_ptr,
    value_block_ptr,
    kv_segment_ids_ref,
    kv_positions_ref,
    lower,
    upper,
    scale,
    query_span: tl.constexpr,
    kv_span: tl.constexpr,
    even_qk_head_dims: tl.constexpr,
    query_block_size: tl.constexpr,
    kv_block_size: tl.constexpr,
    use_causal_mask: tl.constexpr,
    use_segment_mask: tl.constexpr,
    assume_sequential_positions: tl.constexpr,
):
    key_transpose_block_ptr = tl.advance(key_transpose_block_ptr, (0, lower))
    value_block_ptr = tl.advance(value_block_ptr, (lower, 0))
    kv_arange = tl.arange(0, kv_block_size)

    for kv_block_offset in range(lower, upper, kv_block_size):
        if even_qk_head_dims:
            key_transpose_block = tl.load(key_transpose_block_ptr)
        else:
            key_transpose_block = tl.load(
                key_transpose_block_ptr, boundary_check=(0,), padding_option="zero"
            )
        attn_weights = tl.zeros([query_block_size, kv_block_size], dtype=tl.float32)
        attn_weights += tl.dot(query_block, key_transpose_block)

        kv_arange_offsetted = kv_block_offset + kv_arange
        kv_span_offsetted = kv_block_offset + kv_span

        if use_segment_mask:
            kv_segment_ids = tl.load(kv_segment_ids_ref + kv_arange_offsetted)
            mask = make_segment_mask(query_segment_ids, kv_segment_ids, False)
        if use_causal_mask:
            if assume_sequential_positions:
                causal_mask = make_causal_mask(query_span, kv_span_offsetted, False)
            else:
                kv_positions_block = tl.load(kv_positions_ref + kv_arange_offsetted)
                causal_mask = make_causal_mask(
                    query_positions_block, kv_positions_block, False
                )
            if use_segment_mask:
                mask = causal_mask and mask
            else:
                mask = causal_mask

        if use_segment_mask or use_causal_mask:
            attn_weights = tl.where(mask, attn_weights * scale, NEG_INF)
        else:
            attn_weights = attn_weights * scale
        m_new = tl.maximum(m, tl.max(attn_weights, axis=1))
        attn_weights -= m_new[:, None]

        # Flash Attention 2 accumulation
        p = tl.math.exp2(attn_weights)
        l_new = tl.sum(p, axis=1)
        alpha = tl.math.exp2(m - m_new)
        l = l * alpha + l_new  # noqa: E741
        acc *= alpha[:, None]
        value = tl.load(value_block_ptr)
        acc += tl.dot(p.to(value_block_ptr.type.element_ty), value)
        m = m_new
        value_block_ptr = tl.advance(value_block_ptr, (kv_block_size, 0))
        key_transpose_block_ptr = tl.advance(
            key_transpose_block_ptr, (0, kv_block_size)
        )
    return acc, l, m


@triton.jit
def flash_attention_forward_kernel(
    query_ref,
    key_ref,
    value_ref,
    query_positions_ref,
    query_segment_ids_ref,
    kv_positions_ref,
    kv_segment_ids_ref,
    lower_blocks_ref,
    upper_blocks_ref,
    lower_full_blocks_ref,
    upper_full_blocks_ref,
    query_global_offset_ref,
    scale,
    stride_q_batch,
    stride_q_heads,
    stride_q_seq_len,
    stride_q_dims,
    stride_k_batch,
    stride_k_heads,
    stride_k_seq_len,
    stride_k_dims,
    stride_v_batch,
    stride_v_heads,
    stride_v_seq_len,
    stride_v_dims,
    stride_o_batch,
    stride_o_heads,
    stride_o_seq_len,
    stride_o_dims,
    out_ref,
    logsumexp_ref,
    num_heads: tl.constexpr,
    query_seq_len: tl.constexpr,
    kv_seq_len: tl.constexpr,
    qk_head_dim: tl.constexpr,
    qk_head_dim_pad: tl.constexpr,
    even_qk_head_dims: tl.constexpr,
    value_head_dim: tl.constexpr,
    query_block_size: tl.constexpr,
    kv_block_size: tl.constexpr,
    assume_sequential_positions: tl.constexpr,
    num_groups: tl.constexpr,
    is_context_parallelism: tl.constexpr,
):
    query_block_id = tl.program_id(0)
    batch_heads_size_id = tl.program_id(1).to(tl.int64)
    batch_size_id = (batch_heads_size_id // num_heads).to(tl.int64)
    num_query_block_programs = tl.num_programs(0)

    mask_offset = batch_size_id * num_query_block_programs + query_block_id
    lower_bound = tl.load(lower_blocks_ref + mask_offset)
    lower_full_bound = tl.load(lower_full_blocks_ref + mask_offset)
    upper_full_bound = tl.load(upper_full_blocks_ref + mask_offset)
    upper_bound = tl.load(upper_blocks_ref + mask_offset)

    query_offset = batch_heads_size_id * stride_q_heads
    # Note: tl.cdiv don't work with program_ids
    k_offset = (batch_heads_size_id // num_groups).to(tl.int64) * stride_k_heads
    v_offset = (batch_heads_size_id // num_groups).to(tl.int64) * stride_v_heads
    out_offset = batch_heads_size_id * stride_o_heads

    # block pointers
    query_block_ptr = tl.make_block_ptr(
        base=query_ref + query_offset,
        shape=(query_seq_len, qk_head_dim),
        strides=(stride_q_seq_len, stride_q_dims),
        offsets=(query_block_id * query_block_size, 0),
        block_shape=(query_block_size, qk_head_dim_pad),
        order=(1, 0),
    )
    # Load transpose version of key because we need only
    # transpose version during calculations
    key_transpose_block_ptr = tl.make_block_ptr(
        base=key_ref + k_offset,
        shape=(qk_head_dim, kv_seq_len),
        strides=(stride_k_dims, stride_k_seq_len),
        offsets=(0, 0),
        block_shape=(qk_head_dim_pad, kv_block_size),
        order=(0, 1),
    )
    value_block_ptr = tl.make_block_ptr(
        base=value_ref + v_offset,
        shape=(kv_seq_len, value_head_dim),
        strides=(stride_v_seq_len, stride_v_dims),
        offsets=(0, 0),
        block_shape=(kv_block_size, value_head_dim),
        order=(1, 0),
    )
    out_block_ptr = tl.make_block_ptr(
        base=out_ref + out_offset,
        shape=(query_seq_len, value_head_dim),
        strides=(stride_o_seq_len, stride_o_dims),
        offsets=(query_block_id * query_block_size, 0),
        block_shape=(query_block_size, value_head_dim),
        order=(1, 0),
    )

    if is_context_parallelism:
        query_global_offset = tl.load(query_global_offset_ref + query_block_id)
    else:
        query_global_offset = 0

    # initialize offsets
    query_span = query_block_id * query_block_size + tl.arange(0, query_block_size)
    kv_span = tl.arange(0, kv_block_size)

    logsumexp_offset = batch_heads_size_id * query_seq_len
    logsumexp_offset += query_span

    # initialize pointer to m and l
    m = tl.zeros([query_block_size], dtype=tl.float32) - float("inf")
    l = tl.zeros([query_block_size], dtype=tl.float32)  # noqa: E741
    acc = tl.zeros([query_block_size, value_head_dim], dtype=tl.float32)

    num_blocks_to_attend = upper_bound - lower_bound
    if num_blocks_to_attend == 0:
        # Store zeros to outputs
        tl.store(out_block_ptr, acc.to(out_ref.type.element_ty))
        tl.store(logsumexp_ref + logsumexp_offset, l)
        return

    # Load scales
    q_scale = scale
    q_scale *= LOG2_CONST
    q_scale = q_scale.to(tl.float32)

    if even_qk_head_dims:
        query_block = tl.load(query_block_ptr)
    else:
        query_block = tl.load(
            query_block_ptr, boundary_check=(1,), padding_option="zero"
        )

    # Offset segmant ids pointers
    query_segment_ids_ref += batch_size_id * query_seq_len
    kv_segment_ids_ref += batch_size_id * kv_seq_len

    # Offset positions pointers
    if not assume_sequential_positions:
        query_positions_ref += batch_size_id * query_seq_len
        kv_positions_ref += batch_size_id * kv_seq_len

    # Load query segmant ids
    query_segment_ids = tl.load(query_segment_ids_ref + query_span)
    if assume_sequential_positions:
        # Isn't used when assume_sequential_positions == True
        query_positions_block = query_segment_ids
    else:
        query_positions_block = tl.load(query_positions_ref + query_span)

    query_global_span = query_global_offset + query_span

    # Calculate partial blocks on the left side
    acc, l, m = _flash_attention_forward_kernel_inner(
        acc,
        l,
        m,
        query_block,
        query_segment_ids,
        query_positions_block,
        key_transpose_block_ptr,
        value_block_ptr,
        kv_segment_ids_ref,
        kv_positions_ref,
        lower_bound * kv_block_size,
        lower_full_bound * kv_block_size,
        q_scale,
        query_global_span,
        kv_span,
        even_qk_head_dims=even_qk_head_dims,
        query_block_size=query_block_size,
        kv_block_size=kv_block_size,
        use_causal_mask=False,
        use_segment_mask=True,
        assume_sequential_positions=assume_sequential_positions,
    )

    # Calculate full blocks
    acc, l, m = _flash_attention_forward_kernel_inner(
        acc,
        l,
        m,
        query_block,
        query_segment_ids,
        query_positions_block,
        key_transpose_block_ptr,
        value_block_ptr,
        kv_segment_ids_ref,
        kv_positions_ref,
        lower_full_bound * kv_block_size,
        upper_full_bound * kv_block_size,
        q_scale,
        query_global_span,
        kv_span,
        even_qk_head_dims=even_qk_head_dims,
        query_block_size=query_block_size,
        kv_block_size=kv_block_size,
        use_causal_mask=False,
        use_segment_mask=False,
        assume_sequential_positions=assume_sequential_positions,
    )

    # Calculate partial blocks on the right side
    acc, l, m = _flash_attention_forward_kernel_inner(
        acc,
        l,
        m,
        query_block,
        query_segment_ids,
        query_positions_block,
        key_transpose_block_ptr,
        value_block_ptr,
        kv_segment_ids_ref,
        kv_positions_ref,
        upper_full_bound * kv_block_size,
        upper_bound * kv_block_size,
        q_scale,
        query_global_span,
        kv_span,
        even_qk_head_dims=even_qk_head_dims,
        query_block_size=query_block_size,
        kv_block_size=kv_block_size,
        use_causal_mask=True,
        use_segment_mask=True,
        assume_sequential_positions=assume_sequential_positions,
    )

    m += tl.math.log2(l)
    acc /= l[:, None]

    tl.store(logsumexp_ref + logsumexp_offset, m)
    tl.store(out_block_ptr, acc.to(out_ref.type.element_ty))


@triton.jit
def attn_bwd_preprocess(
    out_ref,
    dout_ref,
    stride_batch,
    stride_heads,
    stride_seq_len,
    stride_dims,
    delta_ref,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
):
    block_id = tl.program_id(0).to(tl.int64)
    batch_size_id = tl.program_id(1).to(tl.int64)
    num_heads_id = tl.program_id(2).to(tl.int64)

    init_offset = batch_size_id * stride_batch + num_heads_id * stride_heads

    block_arange = block_id * block_size + tl.arange(0, block_size)
    head_dim_arange = tl.arange(0, head_dim)

    offsets = (
        init_offset
        + block_arange[:, None] * stride_seq_len
        + head_dim_arange[None, :] * stride_dims
    )
    out = tl.load(out_ref + offsets)
    dout = tl.load(dout_ref + offsets).to(tl.float32)

    # It's better to calculate it in fp32 to reduce number of casts
    delta = tl.sum(out * dout, axis=1)

    delta_offsets = batch_size_id * (num_heads * seq_len) + num_heads_id * seq_len
    delta_offsets += block_arange

    tl.store(delta_ref + delta_offsets, delta)


@triton.jit
def _flash_attention_backward_kernel_dkdv_inner(
    dkey,
    dvalue,
    kv_segment_ids,
    kv_positions_block,
    query_ref,
    query_segment_ids_ref,
    query_positions_ref,
    dout_ref,
    logsumexp_ref,
    delta_ref,
    key,
    value,
    qk_scale,
    stride_q_seq_len,
    stride_q_dims,
    stride_o_seq_len,
    stride_o_dims,
    stride_lse_seq_len,
    lower_bound,
    upper_bound,
    query_block_size: tl.constexpr,
    qk_head_dim: tl.constexpr,
    qk_head_dim_pad: tl.constexpr,
    value_head_dim: tl.constexpr,
    even_qk_head_dims: tl.constexpr,
    query_global_offset: tl.constexpr,
    kv_span: tl.constexpr,
    use_causal_mask: tl.constexpr,
    use_segment_mask: tl.constexpr,
    assume_sequential_positions: tl.constexpr,
):
    query_arange = lower_bound * query_block_size + tl.arange(0, query_block_size)
    value_head_dim_arange = tl.arange(0, value_head_dim)
    qk_head_dim_pad_arange = tl.arange(0, qk_head_dim_pad)

    query_transpose_ref = (
        query_ref
        + query_arange[None, :] * stride_q_seq_len
        + qk_head_dim_pad_arange[:, None] * stride_q_dims
    )
    dout_ref += (
        query_arange[:, None] * stride_o_seq_len
        + value_head_dim_arange[None, :] * stride_o_dims
    )

    lse_offset = query_arange * stride_lse_seq_len
    query_offset = 0
    output_offset = 0

    for query_block_id in range(lower_bound, upper_bound):
        if even_qk_head_dims:
            query_transpose = tl.load(query_transpose_ref + query_offset)
        else:
            qk_head_dim_mask = qk_head_dim_pad_arange[:, None] < qk_head_dim
            query_transpose = tl.load(
                query_transpose_ref + query_offset, mask=qk_head_dim_mask, other=0.0
            )

        attn_weights_transpose = tl.dot(key, query_transpose)
        attn_weights_transpose = (attn_weights_transpose * qk_scale).to(tl.float32)

        logsumexp = tl.load(logsumexp_ref + lse_offset)
        pT = tl.math.exp2(attn_weights_transpose - logsumexp[None, :])

        query_seq_len_offset = query_block_id * query_block_size + tl.arange(
            0, query_block_size
        )
        query_span = query_global_offset + query_seq_len_offset

        if use_segment_mask:
            query_segment_ids = tl.load(query_segment_ids_ref + query_seq_len_offset)
            mask = make_segment_mask(query_segment_ids, kv_segment_ids, True)
        if use_causal_mask:
            if assume_sequential_positions:
                causal_mask = query_span[None, :] >= kv_span[:, None]
            else:
                query_positions_block = tl.load(
                    query_positions_ref + query_seq_len_offset
                )
                causal_mask = make_causal_mask(
                    query_positions_block, kv_positions_block, True
                )
            if use_segment_mask:
                mask = causal_mask and mask
            else:
                mask = causal_mask

        if use_segment_mask or use_causal_mask:
            pT = tl.where(mask, pT, 0.0)

        dout = tl.load(dout_ref + output_offset)
        # Compute dvalue
        dvalue += tl.dot(pT.to(dout.type.element_ty), dout)

        delta = tl.load(delta_ref + lse_offset)
        # Compute dp and ds
        dpT = tl.dot(value, tl.trans(dout))
        dsT = pT * (dpT - delta[None, :])
        dsT = dsT.to(query_ref.type.element_ty)

        # Compute dkey
        dkey += tl.dot(dsT, tl.trans(query_transpose))

        # Increment pointers.
        query_offset += query_block_size * stride_q_seq_len
        output_offset += query_block_size * stride_o_seq_len
        lse_offset += query_block_size * stride_lse_seq_len

    return dkey, dvalue


@triton.jit
def _flash_attention_backward_kernel_dquery_inner(
    dquery,
    query_segment_ids,
    query_positions,
    key_ref,
    value_ref,
    kv_segment_ids_ref,
    kv_positions_ref,
    query,
    dout,
    logsumexp,
    delta,
    stride_k_seq_len,
    stride_k_dims,
    stride_v_seq_len,
    stride_v_dims,
    lower_bound,
    upper_bound,
    kv_block_size: tl.constexpr,
    qk_head_dim: tl.constexpr,
    qk_head_dim_pad: tl.constexpr,
    value_head_dim: tl.constexpr,
    even_qk_head_dims: tl.constexpr,
    kv_global_offset: tl.constexpr,
    query_span: tl.constexpr,
    use_causal_mask: tl.constexpr,
    use_segment_mask: tl.constexpr,
    assume_sequential_positions: tl.constexpr,
):
    kv_arange = lower_bound * kv_block_size + tl.arange(0, kv_block_size)
    qk_head_dim_pad_arange = tl.arange(0, qk_head_dim_pad)
    value_head_dim_arange = tl.arange(0, value_head_dim)

    k_offsets = (
        kv_arange[None, :] * stride_k_seq_len
        + qk_head_dim_pad_arange[:, None] * stride_k_dims
    ).to(tl.int64)
    v_offsets = (
        kv_arange[None, :] * stride_v_seq_len
        + value_head_dim_arange[:, None] * stride_v_dims
    ).to(tl.int64)

    key_transpose_ref = key_ref + k_offsets
    value_transpose_ref = value_ref + v_offsets

    for kv_block_id in range(lower_bound, upper_bound):
        if even_qk_head_dims:
            key_transpose = tl.load(key_transpose_ref)
        else:
            key_transpose = tl.load(
                key_transpose_ref,
                mask=qk_head_dim_pad_arange[:, None] < qk_head_dim,
                other=0.0,
            )
        value_transpose = tl.load(value_transpose_ref)
        attention_weights = tl.dot(query, key_transpose)
        p = tl.math.exp2(attention_weights - logsumexp)

        kv_leq_len_offset = kv_block_id * kv_block_size + tl.arange(0, kv_block_size)
        kv_span = kv_global_offset + kv_leq_len_offset

        if use_segment_mask:
            kv_segment_ids = tl.load(kv_segment_ids_ref + kv_leq_len_offset)
            mask = make_segment_mask(query_segment_ids, kv_segment_ids, False)
        if use_causal_mask:
            if assume_sequential_positions:
                causal_mask = query_span[:, None] >= kv_span[None, :]
            else:
                kv_positions_block = tl.load(kv_positions_ref + kv_leq_len_offset)
                causal_mask = make_causal_mask(
                    query_positions, kv_positions_block, False
                )
            if use_segment_mask:
                mask = causal_mask and mask
            else:
                mask = causal_mask

        if use_segment_mask or use_causal_mask:
            p = tl.where(mask, p, 0.0)

        dp = tl.dot(dout, value_transpose).to(tl.float32)
        ds = p * (dp - delta[:, None])

        ds = ds.to(key_transpose.type.element_ty)

        dquery += tl.dot(ds, tl.trans(key_transpose))

        # Increment pointers.
        key_transpose_ref += kv_block_size * stride_k_seq_len
        value_transpose_ref += kv_block_size * stride_v_seq_len
    return dquery


@triton.jit
def flash_attention_backward_kernel_dquery(
    query_ref,
    key_ref,
    value_ref,
    query_positions_ref,
    query_segment_ids_ref,
    kv_positions_ref,
    kv_segment_ids_ref,
    dout_ref,
    logsumexp_ref,
    delta_ref,
    lower_blocks_ref,
    upper_blocks_ref,
    lower_full_blocks_ref,
    upper_full_blocks_ref,
    query_global_offset_ref,
    scale,
    stride_q_batch,
    stride_q_heads,
    stride_q_seq_len,
    stride_q_dims,
    stride_k_batch,
    stride_k_heads,
    stride_k_seq_len,
    stride_k_dims,
    stride_v_batch,
    stride_v_heads,
    stride_v_seq_len,
    stride_v_dims,
    stride_o_batch,
    stride_o_heads,
    stride_o_seq_len,
    stride_o_dims,
    stride_lse_batch,
    stride_lse_heads,
    stride_lse_seq_len,
    dquery_ref,
    query_seq_len: tl.constexpr,
    kv_seq_len: tl.constexpr,
    qk_head_dim: tl.constexpr,
    qk_head_dim_pad: tl.constexpr,
    value_head_dim: tl.constexpr,
    query_block_size: tl.constexpr,
    kv_block_size: tl.constexpr,
    assume_sequential_positions: tl.constexpr,
    num_groups: tl.constexpr,
    is_context_parallelism: tl.constexpr,
):
    query_block_id = tl.program_id(0)
    batch_size_id = tl.program_id(1).to(tl.int64)
    num_heads_id = tl.program_id(2).to(tl.int64)
    num_query_block_programs = tl.num_programs(0)

    mask_offset = batch_size_id * num_query_block_programs + query_block_id
    lower_bound = tl.load(lower_blocks_ref + mask_offset)
    lower_full_bound = tl.load(lower_full_blocks_ref + mask_offset)
    upper_full_bound = tl.load(upper_full_blocks_ref + mask_offset)
    upper_bound = tl.load(upper_blocks_ref + mask_offset)

    even_qk_head_dims: tl.constexpr = qk_head_dim_pad == qk_head_dim

    query_init_offset = (
        batch_size_id * stride_q_batch + num_heads_id * stride_q_heads
    ).to(tl.int64)

    k_init_offset = (
        batch_size_id * stride_k_batch + (num_heads_id // num_groups) * stride_k_heads
    ).to(tl.int64)

    v_init_offset = (
        batch_size_id * stride_v_batch + (num_heads_id // num_groups) * stride_v_heads
    ).to(tl.int64)

    out_init_offset = (
        batch_size_id * stride_o_batch + num_heads_id * stride_o_heads
    ).to(tl.int64)

    lse_init_offset = (
        batch_size_id * stride_lse_batch + num_heads_id * stride_lse_heads
    ).to(tl.int64)

    qk_head_dim_pad_arange = tl.arange(0, qk_head_dim_pad)
    value_head_dim_arange = tl.arange(0, value_head_dim)

    if not even_qk_head_dims:
        qk_head_dim_mask = qk_head_dim_pad_arange[None, :] < qk_head_dim

    # Offset pointers for batch/head
    query_ref += query_init_offset
    key_ref += k_init_offset
    value_ref += v_init_offset
    dout_ref += out_init_offset
    dquery_ref += query_init_offset
    logsumexp_ref += lse_init_offset
    delta_ref += lse_init_offset

    # Offset segments ids pointers
    query_segment_ids_ref += batch_size_id * query_seq_len
    kv_segment_ids_ref += batch_size_id * kv_seq_len

    # Offset positions pointers
    if not assume_sequential_positions:
        query_positions_ref += batch_size_id * query_seq_len
        kv_positions_ref += batch_size_id * kv_seq_len

    if is_context_parallelism:
        query_global_offset = tl.load(query_global_offset_ref + query_block_id)
    else:
        query_global_offset = 0

    query_block_offset = query_block_id * query_block_size + tl.arange(
        0, query_block_size
    )

    query_offsets = (
        query_block_offset[:, None] * stride_q_seq_len
        + qk_head_dim_pad_arange[None, :] * stride_q_dims
    )

    out_offsets = (
        query_block_offset[:, None] * stride_o_seq_len
        + value_head_dim_arange[None, :] * stride_o_dims
    )

    dout = tl.load(dout_ref + out_offsets)
    dquery = tl.zeros([query_block_size, qk_head_dim_pad], dtype=tl.float32)

    qk_scale = (scale * LOG2_CONST).to(tl.float32)

    if even_qk_head_dims:
        query = tl.load(query_ref + query_offsets)
    else:
        query = tl.load(query_ref + query_offsets, mask=qk_head_dim_mask, other=0.0)

    lse_offset = query_block_offset * stride_lse_seq_len
    logsumexp = tl.load(logsumexp_ref + lse_offset)
    delta = tl.load(delta_ref + lse_offset)

    logsumexp = logsumexp[:, None]
    query = (query * qk_scale).to(key_ref.type.element_ty)

    query_segment_ids = tl.load(query_segment_ids_ref + query_block_offset)
    if assume_sequential_positions:
        # Isn't used when assume_sequential_positions == True
        query_positions_block = query_segment_ids
    else:
        query_positions_block = tl.load(query_positions_ref + query_block_offset)

    # Calculate partial blocks on the left side
    dquery = _flash_attention_backward_kernel_dquery_inner(
        dquery,
        query_segment_ids,
        query_positions_block,
        key_ref,
        value_ref,
        kv_segment_ids_ref,
        kv_positions_ref,
        query,
        dout,
        logsumexp,
        delta,
        stride_k_seq_len,
        stride_k_dims,
        stride_v_seq_len,
        stride_v_dims,
        lower_bound,
        lower_full_bound,
        kv_block_size=kv_block_size,
        qk_head_dim=qk_head_dim,
        qk_head_dim_pad=qk_head_dim_pad,
        value_head_dim=value_head_dim,
        even_qk_head_dims=even_qk_head_dims,
        kv_global_offset=0,  # Zero offset for FA
        query_span=query_global_offset + query_block_offset,
        use_causal_mask=False,
        use_segment_mask=True,
        assume_sequential_positions=assume_sequential_positions,
    )

    # Calculate full blocks
    dquery = _flash_attention_backward_kernel_dquery_inner(
        dquery,
        query_segment_ids,
        query_positions_block,
        key_ref,
        value_ref,
        kv_segment_ids_ref,
        kv_positions_ref,
        query,
        dout,
        logsumexp,
        delta,
        stride_k_seq_len,
        stride_k_dims,
        stride_v_seq_len,
        stride_v_dims,
        lower_full_bound,
        upper_full_bound,
        kv_block_size=kv_block_size,
        qk_head_dim=qk_head_dim,
        qk_head_dim_pad=qk_head_dim_pad,
        value_head_dim=value_head_dim,
        even_qk_head_dims=even_qk_head_dims,
        kv_global_offset=0,  # Zero offset for FA
        query_span=query_global_offset + query_block_offset,
        use_causal_mask=False,
        use_segment_mask=False,
        assume_sequential_positions=assume_sequential_positions,
    )

    # Calculate partial blocks on the right side.
    dquery = _flash_attention_backward_kernel_dquery_inner(
        dquery,
        query_segment_ids,
        query_positions_block,
        key_ref,
        value_ref,
        kv_segment_ids_ref,
        kv_positions_ref,
        query,
        dout,
        logsumexp,
        delta,
        stride_k_seq_len,
        stride_k_dims,
        stride_v_seq_len,
        stride_v_dims,
        upper_full_bound,
        upper_bound,
        kv_block_size=kv_block_size,
        qk_head_dim=qk_head_dim,
        qk_head_dim_pad=qk_head_dim_pad,
        value_head_dim=value_head_dim,
        even_qk_head_dims=even_qk_head_dims,
        kv_global_offset=0,  # Zero offset for FA
        query_span=query_global_offset + query_block_offset,
        use_causal_mask=True,
        use_segment_mask=True,
        assume_sequential_positions=assume_sequential_positions,
    )

    dquery *= scale.to(tl.float32)
    # Write back dQ.
    if even_qk_head_dims:
        tl.store(dquery_ref + query_offsets, dquery.to(dquery_ref.type.element_ty))
    else:
        tl.store(
            dquery_ref + query_offsets,
            dquery.to(dquery_ref.type.element_ty),
            mask=qk_head_dim_mask,
        )


@triton.jit
def flash_attention_backward_kernel_dkdv(
    query_ref,
    key_ref,
    value_ref,
    query_positions_ref,
    query_segment_ids_ref,
    kv_positions_ref,
    kv_segment_ids_ref,
    dout_ref,
    logsumexp_ref,
    delta_ref,
    lower_blocks_ref,
    upper_blocks_ref,
    lower_full_blocks_ref,
    upper_full_blocks_ref,
    query_global_offset_ref,
    scale,
    stride_q_batch,
    stride_q_heads,
    stride_q_seq_len,
    stride_q_dims,
    stride_k_batch,
    stride_k_heads,
    stride_k_seq_len,
    stride_k_dims,
    stride_v_batch,
    stride_v_heads,
    stride_v_seq_len,
    stride_v_dims,
    stride_o_batch,
    stride_o_heads,
    stride_o_seq_len,
    stride_o_dims,
    stride_lse_batch,
    stride_lse_heads,
    stride_lse_seq_len,
    stride_dk_batch,
    stride_dk_heads,
    stride_dk_seq_len,
    stride_dk_dims,
    stride_dv_batch,
    stride_dv_heads,
    stride_dv_seq_len,
    stride_dv_dims,
    dkey_ref,
    dvalue_ref,
    query_seq_len: tl.constexpr,
    kv_seq_len: tl.constexpr,
    qk_head_dim: tl.constexpr,
    qk_head_dim_pad: tl.constexpr,
    value_head_dim: tl.constexpr,
    query_block_size: tl.constexpr,
    kv_block_size: tl.constexpr,
    assume_sequential_positions: tl.constexpr,
    memory_optimized_gqa_backward: tl.constexpr,
    # To implement dkdv bwd it's needed to load new offsets when passing through the
    # middle of a token sequence because the sequence consists of two different blocks.
    # The load_second_global_offset flag indicates whether the second set of offsets
    # needs to be loaded.
    load_second_global_offset: tl.constexpr,
    num_groups: tl.constexpr,
    is_context_parallelism: tl.constexpr,
):
    kv_block_id = tl.program_id(0)
    batch_size_id = tl.program_id(1).to(tl.int64)
    num_heads_id = tl.program_id(2).to(tl.int64)
    num_kv_block_programs = tl.num_programs(0)

    if is_context_parallelism:
        query_global_offset = tl.load(query_global_offset_ref)
        if (
            load_second_global_offset
            and kv_block_id * kv_block_size - query_global_offset >= query_seq_len // 2
        ):
            query_global_offset = tl.load(query_global_offset_ref + 1)
    else:
        query_global_offset = 0

    mask_offset = batch_size_id * num_kv_block_programs + kv_block_id
    lower_bound = tl.load(lower_blocks_ref + mask_offset)
    lower_full_bound = tl.load(lower_full_blocks_ref + mask_offset)
    upper_full_bound = tl.load(upper_full_blocks_ref + mask_offset)
    upper_bound = tl.load(upper_blocks_ref + mask_offset)

    even_qk_head_dims: tl.constexpr = qk_head_dim_pad == qk_head_dim

    query_init_offset = (
        batch_size_id * stride_q_batch + num_heads_id * stride_q_heads
    ).to(tl.int64)

    k_init_offset = (
        batch_size_id * stride_k_batch + (num_heads_id // num_groups) * stride_k_heads
    ).to(tl.int64)

    v_init_offset = (
        batch_size_id * stride_v_batch + (num_heads_id // num_groups) * stride_v_heads
    ).to(tl.int64)

    out_init_offset = (
        batch_size_id * stride_o_batch + num_heads_id * stride_o_heads
    ).to(tl.int64)

    lse_init_offset = (
        batch_size_id * stride_lse_batch + num_heads_id * stride_lse_heads
    ).to(tl.int64)

    if memory_optimized_gqa_backward:
        dk_init_offset = (
            batch_size_id * stride_dk_batch
            + (num_heads_id // num_groups) * stride_dk_heads
        ).to(tl.int64)

        dv_init_offset = (
            batch_size_id * stride_dv_batch
            + (num_heads_id // num_groups) * stride_dv_heads
        ).to(tl.int64)
    else:
        dk_init_offset = (
            batch_size_id * stride_dk_batch + num_heads_id * stride_dk_heads
        ).to(tl.int64)

        dv_init_offset = (
            batch_size_id * stride_dv_batch + num_heads_id * stride_dv_heads
        ).to(tl.int64)

    # Offset pointers for batch/head
    query_ref += query_init_offset
    key_ref += k_init_offset
    value_ref += v_init_offset
    dout_ref += out_init_offset
    # dkey and dvalue reduce kv_num_heads in a separate kernel,
    # so they have the same num_heads as query
    dkey_ref += dk_init_offset
    dvalue_ref += dv_init_offset
    logsumexp_ref += lse_init_offset
    delta_ref += lse_init_offset

    # Offset segments ids pointers
    query_segment_ids_ref += batch_size_id * query_seq_len
    kv_segment_ids_ref += batch_size_id * kv_seq_len

    # offset positions pointers
    if not assume_sequential_positions:
        query_positions_ref += batch_size_id * query_seq_len
        kv_positions_ref += batch_size_id * kv_seq_len

    qk_head_dim_pad_arange = tl.arange(0, qk_head_dim_pad)
    value_head_dim_arange = tl.arange(0, value_head_dim)

    qk_scale = (scale * LOG2_CONST).to(tl.float32)

    kv_block_offset = kv_block_id * kv_block_size + tl.arange(0, kv_block_size)

    dvalue = tl.zeros([kv_block_size, value_head_dim], dtype=tl.float32)
    dkey = tl.zeros([kv_block_size, qk_head_dim_pad], dtype=tl.float32)

    # Load key and value
    k_offsets = (
        kv_block_offset[:, None] * stride_k_seq_len
        + qk_head_dim_pad_arange[None, :] * stride_k_dims
    )

    v_offsets = (
        kv_block_offset[:, None] * stride_v_seq_len
        + value_head_dim_arange[None, :] * stride_v_dims
    )

    dk_offsets = (
        kv_block_offset[:, None] * stride_dk_seq_len
        + qk_head_dim_pad_arange[None, :] * stride_dk_dims
    )

    dv_offsets = (
        kv_block_offset[:, None] * stride_dv_seq_len
        + value_head_dim_arange[None, :] * stride_dv_dims
    )

    if even_qk_head_dims:
        key = tl.load(key_ref + k_offsets)
    else:
        qk_head_dim_mask = qk_head_dim_pad_arange[None, :] < qk_head_dim
        key = tl.load(key_ref + k_offsets, mask=qk_head_dim_mask, other=0.0)

    value = tl.load(value_ref + v_offsets)

    kv_segment_ids = tl.load(kv_segment_ids_ref + kv_block_offset)
    if assume_sequential_positions:
        # Isn't used when assume_sequential_positions == True
        kv_positions_block = kv_segment_ids
    else:
        kv_positions_block = tl.load(kv_positions_ref + kv_block_offset)

    # Calculate partial blocks on the left side
    dkey, dvalue = _flash_attention_backward_kernel_dkdv_inner(
        dkey,
        dvalue,
        kv_segment_ids,
        kv_positions_block,
        query_ref,
        query_segment_ids_ref,
        query_positions_ref,
        dout_ref,
        logsumexp_ref,
        delta_ref,
        key,
        value,
        qk_scale,
        stride_q_seq_len,
        stride_q_dims,
        stride_o_seq_len,
        stride_o_dims,
        stride_lse_seq_len,
        lower_bound,
        lower_full_bound,
        query_block_size=query_block_size,
        qk_head_dim=qk_head_dim,
        qk_head_dim_pad=qk_head_dim_pad,
        value_head_dim=value_head_dim,
        even_qk_head_dims=even_qk_head_dims,
        query_global_offset=query_global_offset,
        kv_span=kv_block_offset,
        use_causal_mask=True,
        use_segment_mask=True,
        assume_sequential_positions=assume_sequential_positions,
    )

    # Calculate full blocks
    dkey, dvalue = _flash_attention_backward_kernel_dkdv_inner(
        dkey,
        dvalue,
        kv_segment_ids,
        kv_positions_block,
        query_ref,
        query_segment_ids_ref,
        query_positions_ref,
        dout_ref,
        logsumexp_ref,
        delta_ref,
        key,
        value,
        qk_scale,
        stride_q_seq_len,
        stride_q_dims,
        stride_o_seq_len,
        stride_o_dims,
        stride_lse_seq_len,
        lower_full_bound,
        upper_full_bound,
        query_block_size=query_block_size,
        qk_head_dim=qk_head_dim,
        qk_head_dim_pad=qk_head_dim_pad,
        value_head_dim=value_head_dim,
        even_qk_head_dims=even_qk_head_dims,
        query_global_offset=query_global_offset,
        kv_span=kv_block_offset,
        use_causal_mask=False,
        use_segment_mask=False,
        assume_sequential_positions=assume_sequential_positions,
    )

    # Calculate partial blocks on the right side
    dkey, dvalue = _flash_attention_backward_kernel_dkdv_inner(
        dkey,
        dvalue,
        kv_segment_ids,
        kv_positions_block,
        query_ref,
        query_segment_ids_ref,
        query_positions_ref,
        dout_ref,
        logsumexp_ref,
        delta_ref,
        key,
        value,
        qk_scale,
        stride_q_seq_len,
        stride_q_dims,
        stride_o_seq_len,
        stride_o_dims,
        stride_lse_seq_len,
        upper_full_bound,
        upper_bound,
        query_block_size=query_block_size,
        qk_head_dim=qk_head_dim,
        qk_head_dim_pad=qk_head_dim_pad,
        value_head_dim=value_head_dim,
        even_qk_head_dims=even_qk_head_dims,
        query_global_offset=query_global_offset,
        kv_span=kv_block_offset,
        use_causal_mask=False,
        use_segment_mask=True,
        assume_sequential_positions=assume_sequential_positions,
    )

    dkey *= scale.to(tl.float32)
    # Save dk and dv
    if memory_optimized_gqa_backward:
        tl.atomic_add(dvalue_ref + dv_offsets, dvalue.to(dvalue_ref.type.element_ty))
        if even_qk_head_dims:
            tl.atomic_add(dkey_ref + dk_offsets, dkey.to(dkey_ref.type.element_ty))
        else:
            tl.atomic_add(
                dkey_ref + dk_offsets,
                dkey.to(dkey_ref.type.element_ty),
                mask=qk_head_dim_mask,
            )
    else:
        tl.store(dvalue_ref + dv_offsets, dvalue.to(dvalue_ref.type.element_ty))

        if even_qk_head_dims:
            tl.store(dkey_ref + dk_offsets, dkey.to(dkey_ref.type.element_ty))
        else:
            tl.store(
                dkey_ref + dk_offsets,
                dkey.to(dkey_ref.type.element_ty),
                mask=qk_head_dim_mask,
            )


@triton.jit
def flash_attention_backward_kernel(
    query_ref,
    key_ref,
    value_ref,
    query_positions_ref,
    query_segment_ids_ref,
    kv_positions_ref,
    kv_segment_ids_ref,
    dout_ref,
    logsumexp_ref,
    delta_ref,
    lower_blocks_query_ref,
    upper_blocks_query_ref,
    lower_full_blocks_query_ref,
    upper_full_blocks_query_ref,
    lower_blocks_kv_ref,
    upper_blocks_kv_ref,
    lower_full_blocks_kv_ref,
    upper_full_blocks_kv_ref,
    query_global_offset_ref,
    scale,
    stride_q_batch,
    stride_q_heads,
    stride_q_seq_len,
    stride_q_dims,
    stride_k_batch,
    stride_k_heads,
    stride_k_seq_len,
    stride_k_dims,
    stride_v_batch,
    stride_v_heads,
    stride_v_seq_len,
    stride_v_dims,
    stride_o_batch,
    stride_o_heads,
    stride_o_seq_len,
    stride_o_dims,
    stride_lse_batch,
    stride_lse_heads,
    stride_lse_seq_len,
    stride_dk_batch,
    stride_dk_heads,
    stride_dk_seq_len,
    stride_dk_dims,
    stride_dv_batch,
    stride_dv_heads,
    stride_dv_seq_len,
    stride_dv_dims,
    dquery_ref,
    dkey_ref,
    dvalue_ref,
    query_seq_len: tl.constexpr,
    kv_seq_len: tl.constexpr,
    qk_head_dim: tl.constexpr,
    qk_head_dim_pad: tl.constexpr,
    value_head_dim: tl.constexpr,
    query_block_size_dkdv: tl.constexpr,
    kv_block_size_dkdv: tl.constexpr,
    query_block_size_dq: tl.constexpr,
    kv_block_size_dq: tl.constexpr,
    assume_sequential_positions: tl.constexpr,
    memory_optimized_gqa_backward: tl.constexpr,
    num_groups: tl.constexpr,
):
    # Calculate dkey and dvalue:
    flash_attention_backward_kernel_dkdv(
        query_ref,
        key_ref,
        value_ref,
        query_positions_ref,
        query_segment_ids_ref,
        kv_positions_ref,
        kv_segment_ids_ref,
        dout_ref,
        logsumexp_ref,
        delta_ref,
        lower_blocks_kv_ref,
        upper_blocks_kv_ref,
        lower_full_blocks_kv_ref,
        upper_full_blocks_kv_ref,
        query_global_offset_ref,
        scale,
        stride_q_batch,
        stride_q_heads,
        stride_q_seq_len,
        stride_q_dims,
        stride_k_batch,
        stride_k_heads,
        stride_k_seq_len,
        stride_k_dims,
        stride_v_batch,
        stride_v_heads,
        stride_v_seq_len,
        stride_v_dims,
        stride_o_batch,
        stride_o_heads,
        stride_o_seq_len,
        stride_o_dims,
        stride_lse_batch,
        stride_lse_heads,
        stride_lse_seq_len,
        stride_dk_batch,
        stride_dk_heads,
        stride_dk_seq_len,
        stride_dk_dims,
        stride_dv_batch,
        stride_dv_heads,
        stride_dv_seq_len,
        stride_dv_dims,
        dkey_ref,
        dvalue_ref,
        query_seq_len,
        kv_seq_len,
        qk_head_dim,
        qk_head_dim_pad,
        value_head_dim,
        query_block_size_dkdv,
        kv_block_size_dkdv,
        assume_sequential_positions,
        memory_optimized_gqa_backward,
        # The load_second_global_offset should always be disabled here because in this
        # scenario the sequence consists of a single block of tokens.
        False,
        num_groups,
        # This implementation is not used for the context parallelism case
        False,  # is_context_parallelism
    )

    # Calculate dquery block
    flash_attention_backward_kernel_dquery(
        query_ref,
        key_ref,
        value_ref,
        query_positions_ref,
        query_segment_ids_ref,
        kv_positions_ref,
        kv_segment_ids_ref,
        dout_ref,
        logsumexp_ref,
        delta_ref,
        lower_blocks_query_ref,
        upper_blocks_query_ref,
        lower_full_blocks_query_ref,
        upper_full_blocks_query_ref,
        query_global_offset_ref,
        scale,
        stride_q_batch,
        stride_q_heads,
        stride_q_seq_len,
        stride_q_dims,
        stride_k_batch,
        stride_k_heads,
        stride_k_seq_len,
        stride_k_dims,
        stride_v_batch,
        stride_v_heads,
        stride_v_seq_len,
        stride_v_dims,
        stride_o_batch,
        stride_o_heads,
        stride_o_seq_len,
        stride_o_dims,
        stride_lse_batch,
        stride_lse_heads,
        stride_lse_seq_len,
        dquery_ref,
        query_seq_len,
        kv_seq_len,
        qk_head_dim,
        qk_head_dim_pad,
        value_head_dim,
        query_block_size_dq,
        kv_block_size_dq,
        assume_sequential_positions,
        num_groups,
        # This implementation is not used for the context parallelism case
        False,  # is_context_parallelism
    )


def flash_attention_triton_forward(
    query: DeviceArray,
    key: DeviceArray,
    value: DeviceArray,
    query_positions: DeviceArray,
    query_segment_ids: DeviceArray,
    kv_positions: DeviceArray,
    kv_segment_ids: DeviceArray,
    mask_tensors: tuple[DeviceArray] | None,
    scale: float,
    bias: DeviceArray | None,
    fwd_params: FlashAttentionParamsConfig,
    bwd_params: FlashAttentionParamsConfig,
    assume_sequential_positions: bool = False,
    memory_optimized_gqa_backward: bool = False,
    permute_tokens_for_load_balance: bool = True,
    context_parallelism_mesh_axis_name: str | None = None,
    debug: bool = False,
) -> tuple[DeviceArray, Sequence[DeviceArray]]:
    if bias is not None:
        raise NotImplementedError("Bias is not supported in Flash multi head attention")

    possible_hid_dim_vals = [16, 32, 64, 128, 192, 256]
    chex.assert_axis_dimension_comparator(
        query,
        axis=-1,
        pass_fn=lambda x: x in possible_hid_dim_vals,
        error_string=f"Attention hid_dim can take values {possible_hid_dim_vals}",
    )

    chex.assert_rank([query, key, value], 4)

    batch_size, num_heads, query_seq_len, qk_head_dim = query.shape
    _, num_kv_heads, kv_seq_len, value_head_dim = value.shape

    chex.assert_shape([key, value], [batch_size, num_kv_heads, kv_seq_len, None])
    chex.assert_shape([query_positions, query_segment_ids], [batch_size, query_seq_len])
    chex.assert_shape([kv_positions, kv_segment_ids], [batch_size, kv_seq_len])

    chex.assert_is_divisible(num_heads, num_kv_heads)

    query_block_size = fwd_params.query_block_size
    kv_block_size = fwd_params.kv_block_size
    num_warps = fwd_params.num_warps
    num_stages = fwd_params.num_stages

    is_context_parallelism = context_parallelism_mesh_axis_name is not None
    if is_context_parallelism:
        query_chunk_idx = jax.lax.axis_index(context_parallelism_mesh_axis_name)
        chex.assert_is_divisible(query_seq_len, query_block_size)

        if permute_tokens_for_load_balance:
            axis_size = jax.lax.psum(1, context_parallelism_mesh_axis_name)
            query_global_offset_arange = jnp.zeros(
                (query_seq_len // 2 // query_block_size), dtype=jnp.int32
            )
            query_global_offset_first_part = (
                query_chunk_idx * query_seq_len // 2 + query_global_offset_arange
            )
            query_global_offset_second_part = (
                2 * axis_size - query_chunk_idx - 1
            ) * query_seq_len // 2 + query_global_offset_arange
            query_global_offset_second_part -= query_seq_len // 2
            query_global_offset = jnp.concatenate(
                [query_global_offset_first_part, query_global_offset_second_part],
                axis=0,
            )
        else:
            query_global_offset = (
                jnp.ones((query_seq_len // query_block_size), dtype=jnp.int32)
                * query_chunk_idx
                * query_seq_len
            )
    else:
        query_global_offset = jnp.zeros(
            (query_seq_len // query_block_size), dtype=jnp.int32
        )

    # Pad input tensors to block size. In practice this should be avoided by choosing
    # a block size that is a multiple of the sequence length.
    # This is however helpful during decoding when query length is 1,
    # and also to support arbitrarily-shaped inputs.
    (query,), query_positions, query_segment_ids = pad_to_block_size(
        embeddings=(query,),
        positions=query_positions,
        segment_ids=query_segment_ids,
        block_size=query_block_size,
        pos_fill_value=-1,  # Any filler should work
        transposed_inputs=True,
    )
    (key, value), kv_positions, kv_segment_ids = pad_to_block_size(
        embeddings=(key, value),
        positions=kv_positions,
        segment_ids=kv_segment_ids,
        block_size=kv_block_size,
        pos_fill_value=jnp.iinfo(jnp.int32).max,  # These should not be attended to
        transposed_inputs=True,
    )
    orig_query_seq_len = query_seq_len
    query_seq_len = query.shape[2]
    kv_seq_len = key.shape[2]

    num_query_blocks = jt.cdiv(orig_query_seq_len, query_block_size)

    grid = (num_query_blocks, batch_size * num_heads)

    if mask_tensors is None or len(mask_tensors) == 0:
        raise ValueError("Length of mask_tensors should be at least equal to one")
    mask_fwd = mask_tensors[0]
    lower_bounds = mask_fwd[0]
    upper_bounds = mask_fwd[1]
    lower_full_bounds = mask_fwd[2]
    upper_full_bounds = mask_fwd[3]

    num_groups = num_heads // num_kv_heads

    qk_head_dim_pad = jt.next_power_of_2(qk_head_dim)
    metaparams = dict(
        num_heads=num_heads,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        qk_head_dim=qk_head_dim,
        qk_head_dim_pad=qk_head_dim_pad,
        even_qk_head_dims=qk_head_dim_pad == qk_head_dim,
        value_head_dim=value_head_dim,
        query_block_size=query_block_size,
        kv_block_size=kv_block_size,
        assume_sequential_positions=assume_sequential_positions,
        is_context_parallelism=is_context_parallelism,
        num_groups=num_groups,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    out_shape = (batch_size, num_heads, query_seq_len, value_head_dim)
    out_shape_struct = [
        jax.ShapeDtypeStruct(shape=out_shape, dtype=query.dtype),  # out
        jax.ShapeDtypeStruct(
            shape=(batch_size, num_heads, query_seq_len), dtype=jnp.float32
        ),  # logsumexp
    ]

    out, logsumexp = jt.triton_call(
        query,
        key,
        value,
        query_positions,
        query_segment_ids,
        kv_positions,
        kv_segment_ids,
        lower_bounds,
        upper_bounds,
        lower_full_bounds,
        upper_full_bounds,
        query_global_offset,
        scale,
        *jt.strides_from_shape(query.shape),
        *jt.strides_from_shape(key.shape),
        *jt.strides_from_shape(value.shape),
        *jt.strides_from_shape(out_shape),
        kernel=flash_attention_forward_kernel,
        out_shape=out_shape_struct,
        grid=grid,
        debug=debug,
        **metaparams,
    )

    # Get rid of padding
    out = out[:, :, :orig_query_seq_len, :]
    logsumexp = logsumexp[:, :, :orig_query_seq_len]

    # Assign the checkpoint name to attention outputs to save the tensors for
    # the backward pass when remat_attn is False
    out = checkpoint_name(out, "attention_outputs")
    logsumexp = checkpoint_name(logsumexp, "attention_outputs")
    outputs_for_bwd_pass = [
        query,
        key,
        value,
        query_positions,
        query_segment_ids,
        kv_positions,
        kv_segment_ids,
        mask_tensors,
        out,
        logsumexp,
    ]

    return out, outputs_for_bwd_pass


def flash_attention_backward(
    scale: float,
    bias: DeviceArray | None,
    fwd_params: FlashAttentionParamsConfig,
    bwd_params: FlashAttentionParamsConfig,
    assume_sequential_positions: bool,
    memory_optimized_gqa_backward: bool,
    permute_tokens_for_load_balance: bool,
    context_parallelism_mesh_axis_name: str | None,
    debug: bool,
    res: DeviceArray,
    dout: DeviceArray,
):
    (
        query,
        key,
        value,
        query_positions,
        query_segment_ids,
        kv_positions,
        kv_segment_ids,
        mask_tensors,
        out,
        logsumexp,
    ) = res

    chex.assert_rank([query, key, value, out, dout], 4)

    batch_size, num_heads, query_seq_len, qk_head_dim = query.shape
    _, num_kv_heads, kv_seq_len, value_head_dim = value.shape
    qk_head_dim_pad = jt.next_power_of_2(qk_head_dim)

    num_groups = num_heads // num_kv_heads

    chex.assert_shape([key, value], [batch_size, num_kv_heads, kv_seq_len, None])
    chex.assert_shape([out, dout], [batch_size, None, query_seq_len, value_head_dim])
    chex.assert_shape([out, dout, query], [batch_size, num_heads, query_seq_len, None])
    chex.assert_shape([query_positions, query_segment_ids], [batch_size, query_seq_len])
    chex.assert_shape([kv_positions, kv_segment_ids], [batch_size, kv_seq_len])

    # Check if we don't need padding during training.
    # The block sizes are swapped for the dquery kernel (see below), so both
    # sequence lengths must be divisible by both block sizes.
    chex.assert_is_divisible(query_seq_len, bwd_params.query_block_size)
    chex.assert_is_divisible(kv_seq_len, bwd_params.kv_block_size)
    chex.assert_is_divisible(query_seq_len, bwd_params.kv_block_size)
    chex.assert_is_divisible(kv_seq_len, bwd_params.query_block_size)

    # We swap here query and kv block sizes for kv and query calculate
    # stagers due to performance
    query_block_size_dkdv = bwd_params.query_block_size
    kv_block_size_dkdv = bwd_params.kv_block_size
    query_block_size_dq = bwd_params.kv_block_size
    kv_block_size_dq = bwd_params.query_block_size

    num_query_blocks = query_seq_len // query_block_size_dq
    num_kv_blocks = kv_seq_len // kv_block_size_dkdv
    grid = (num_kv_blocks, batch_size, num_heads)

    is_context_parallelism = context_parallelism_mesh_axis_name is not None
    # The combined backward kernel shares the kv-block grid between the dkdv and
    # dquery computations, which requires matching numbers of query and kv
    # blocks. Fall back to separate kernel launches when they differ (e.g. when
    # query_seq_len != kv_seq_len).
    use_separate_kernel_impl = (
        is_context_parallelism or num_query_blocks != num_kv_blocks
    )

    if is_context_parallelism:
        query_chunk_idx = jax.lax.axis_index(context_parallelism_mesh_axis_name)
        chex.assert_is_divisible(query_seq_len, query_block_size_dkdv)
        chex.assert_is_divisible(query_seq_len, query_block_size_dq)

        if permute_tokens_for_load_balance:
            axis_size = jax.lax.psum(1, context_parallelism_mesh_axis_name)
            query_global_offset_dkdv = jnp.zeros((2), dtype=jnp.int32)
            query_global_offset_dq = jnp.zeros(
                (query_seq_len // query_block_size_dq), dtype=jnp.int32
            )

            query_global_offset_first_part = query_chunk_idx * query_seq_len // 2
            query_global_offset_second_part = (
                2 * axis_size - query_chunk_idx - 1
            ) * query_seq_len // 2 - query_seq_len // 2

            half_seq_len = 1
            query_global_offset_dkdv = query_global_offset_dkdv.at[:half_seq_len].set(
                query_global_offset_first_part
            )
            query_global_offset_dkdv = query_global_offset_dkdv.at[half_seq_len:].set(
                query_global_offset_second_part
            )

            half_seq_len = query_seq_len // query_block_size_dq // 2
            query_global_offset_dq = query_global_offset_dq.at[:half_seq_len].set(
                query_global_offset_first_part
            )
            query_global_offset_dq = query_global_offset_dq.at[half_seq_len:].set(
                query_global_offset_second_part
            )

        else:
            query_global_offset_dkdv = query_chunk_idx * query_seq_len

            query_global_offset_dq = (
                jnp.ones((query_seq_len // query_block_size_dq), dtype=jnp.int32)
                * query_chunk_idx
                * query_seq_len
            )
    else:
        query_global_offset_dkdv = jnp.zeros((1,), dtype=jnp.int32)
        query_global_offset_dq = jnp.zeros(
            (query_seq_len // query_block_size_dq), dtype=jnp.int32
        )

    if len(mask_tensors) != 3:
        raise ValueError("Length of mask_tensors should be equal to two")
    mask_dquery = mask_tensors[1]
    mask_dkdv = mask_tensors[2]

    lower_bounds_dquery = mask_dquery[0]
    upper_bounds_dquery = mask_dquery[1]
    lower_full_bounds_dquery = mask_dquery[2]
    upper_full_bounds_dquery = mask_dquery[3]

    lower_bounds_dkdv = mask_dkdv[0]
    upper_bounds_dkdv = mask_dkdv[1]
    lower_full_bounds_dkdv = mask_dkdv[2]
    upper_full_bounds_dkdv = mask_dkdv[3]

    preprocess_block_size = 128
    pre_grid = (query_seq_len // preprocess_block_size, batch_size, num_heads)

    out_shape = jax.ShapeDtypeStruct(shape=logsumexp.shape, dtype=jnp.float32)

    delta = jt.triton_call(
        out,
        dout,
        *jt.strides_from_shape(out.shape),
        num_heads=num_heads,
        seq_len=query_seq_len,
        kernel=attn_bwd_preprocess,
        out_shape=out_shape,
        grid=pre_grid,
        block_size=preprocess_block_size,
        head_dim=value_head_dim,
        zeroed_outputs=() if assume_sequential_positions else (0,),
    )

    metaparams = dict(
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        qk_head_dim=qk_head_dim,
        qk_head_dim_pad=qk_head_dim_pad,
        value_head_dim=value_head_dim,
        assume_sequential_positions=assume_sequential_positions,
        num_groups=num_groups,
        num_warps=bwd_params.num_warps,
        num_stages=bwd_params.num_stages,
        debug=debug,
    )

    if memory_optimized_gqa_backward:
        dk_shape = (batch_size, num_kv_heads, kv_seq_len, qk_head_dim)
        dv_shape = (batch_size, num_kv_heads, kv_seq_len, value_head_dim)
        zeroed_outputs = (0, 1) if use_separate_kernel_impl else (1, 2)
    else:
        dk_shape = (batch_size, num_heads, kv_seq_len, qk_head_dim)
        dv_shape = (batch_size, num_heads, kv_seq_len, value_head_dim)
        zeroed_outputs = ()

    dquery_shape_dtype = jax.ShapeDtypeStruct(shape=query.shape, dtype=query.dtype)
    dkey_shape_dtype = jax.ShapeDtypeStruct(shape=dk_shape, dtype=jnp.float32)
    dvalue_shape_dtype = jax.ShapeDtypeStruct(shape=dv_shape, dtype=jnp.float32)

    if use_separate_kernel_impl:
        out_shape = [dquery_shape_dtype]
        dq_metaparams = dict(
            **metaparams,
            query_block_size=query_block_size_dq,
            kv_block_size=kv_block_size_dq,
            is_context_parallelism=is_context_parallelism,
        )
        dquery = jt.triton_call(
            query,
            key,
            value,
            query_positions,
            query_segment_ids,
            kv_positions,
            kv_segment_ids,
            dout,
            logsumexp,
            delta,
            lower_bounds_dquery,
            upper_bounds_dquery,
            lower_full_bounds_dquery,
            upper_full_bounds_dquery,
            query_global_offset_dq,
            scale,
            *jt.strides_from_shape(query.shape),
            *jt.strides_from_shape(key.shape),
            *jt.strides_from_shape(value.shape),
            *jt.strides_from_shape(out.shape),
            *jt.strides_from_shape(logsumexp.shape),
            kernel=flash_attention_backward_kernel_dquery,
            out_shape=out_shape,
            grid=(num_query_blocks, batch_size, num_heads),
            **dq_metaparams,
        )[0]

        out_shape = [dkey_shape_dtype, dvalue_shape_dtype]
        dkdv_metaparams = dict(
            **metaparams,
            query_block_size=query_block_size_dkdv,
            kv_block_size=kv_block_size_dkdv,
            memory_optimized_gqa_backward=memory_optimized_gqa_backward,
            is_context_parallelism=is_context_parallelism,
            load_second_global_offset=is_context_parallelism
            and permute_tokens_for_load_balance,
        )
        dkey, dvalue = jt.triton_call(
            query,
            key,
            value,
            query_positions,
            query_segment_ids,
            kv_positions,
            kv_segment_ids,
            dout,
            logsumexp,
            delta,
            lower_bounds_dkdv,
            upper_bounds_dkdv,
            lower_full_bounds_dkdv,
            upper_full_bounds_dkdv,
            query_global_offset_dkdv,
            scale,
            *jt.strides_from_shape(query.shape),
            *jt.strides_from_shape(key.shape),
            *jt.strides_from_shape(value.shape),
            *jt.strides_from_shape(out.shape),
            *jt.strides_from_shape(logsumexp.shape),
            *jt.strides_from_shape(dk_shape),
            *jt.strides_from_shape(dv_shape),
            kernel=flash_attention_backward_kernel_dkdv,
            out_shape=out_shape,
            grid=grid,
            **dkdv_metaparams,
            zeroed_outputs=zeroed_outputs,
        )
    else:
        out_shape = [dquery_shape_dtype, dkey_shape_dtype, dvalue_shape_dtype]
        metaparams = dict(
            **metaparams,
            query_block_size_dkdv=query_block_size_dkdv,
            kv_block_size_dkdv=kv_block_size_dkdv,
            query_block_size_dq=query_block_size_dq,
            kv_block_size_dq=kv_block_size_dq,
            memory_optimized_gqa_backward=memory_optimized_gqa_backward,
        )
        query_global_offset = 0
        dquery, dkey, dvalue = jt.triton_call(
            query,
            key,
            value,
            query_positions,
            query_segment_ids,
            kv_positions,
            kv_segment_ids,
            dout,
            logsumexp,
            delta,
            lower_bounds_dquery,
            upper_bounds_dquery,
            lower_full_bounds_dquery,
            upper_full_bounds_dquery,
            lower_bounds_dkdv,
            upper_bounds_dkdv,
            lower_full_bounds_dkdv,
            upper_full_bounds_dkdv,
            query_global_offset,
            scale,
            *jt.strides_from_shape(query.shape),
            *jt.strides_from_shape(key.shape),
            *jt.strides_from_shape(value.shape),
            *jt.strides_from_shape(out.shape),
            *jt.strides_from_shape(logsumexp.shape),
            *jt.strides_from_shape(dk_shape),
            *jt.strides_from_shape(dv_shape),
            kernel=flash_attention_backward_kernel,
            out_shape=out_shape,
            grid=grid,
            **metaparams,
            zeroed_outputs=zeroed_outputs,
        )

    # group query attention
    if not memory_optimized_gqa_backward and num_groups > 1:
        # run sum across num_groups axis
        dkey = jnp.sum(
            dkey.reshape(batch_size, num_kv_heads, num_groups, kv_seq_len, qk_head_dim),
            axis=2,
            dtype=jnp.float32,
        )
        dvalue = jnp.sum(
            dvalue.reshape(
                batch_size, num_kv_heads, num_groups, kv_seq_len, value_head_dim
            ),
            axis=2,
            dtype=jnp.float32,
        )

    return (
        dquery,
        dkey,
        dvalue,
        None,
        None,
        None,
        None,
        None,
    )


@functools.partial(
    jax.custom_vjp,
    nondiff_argnums=[8, 9, 10, 11, 12, 13, 14, 15, 16],
)
@functools.partial(
    jax.jit,
    static_argnames=[
        "scale",
        "bias",
        "fwd_params",
        "bwd_params",
        "assume_sequential_positions",
        "memory_optimized_gqa_backward",
        "permute_tokens_for_load_balance",
        "context_parallelism_mesh_axis_name",
        "debug",
    ],
)
def flash_attention_triton_single_device(
    query: DeviceArray,
    key: DeviceArray,
    value: DeviceArray,
    query_positions: DeviceArray,
    query_segment_ids: DeviceArray,
    kv_positions: DeviceArray,
    kv_segment_ids: DeviceArray,
    mask_tensors: tuple[AttentionMask] | None = None,
    scale: float = 1.0,
    bias: DeviceArray | None = None,
    fwd_params: FlashAttentionParamsConfig = FlashAttentionParamsConfig(),
    bwd_params: FlashAttentionParamsConfig = FlashAttentionParamsConfig(),
    assume_sequential_positions: bool = False,
    memory_optimized_gqa_backward: bool = False,
    permute_tokens_for_load_balance: bool = True,
    context_parallelism_mesh_axis_name: str | None = None,
    debug: bool = False,
) -> DeviceArray:
    result = flash_attention_triton_forward(
        query=query,
        key=key,
        value=value,
        query_positions=query_positions,
        query_segment_ids=query_segment_ids,
        kv_positions=kv_positions,
        kv_segment_ids=kv_segment_ids,
        mask_tensors=mask_tensors,
        scale=scale,
        bias=bias,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
        assume_sequential_positions=assume_sequential_positions,
        memory_optimized_gqa_backward=memory_optimized_gqa_backward,
        permute_tokens_for_load_balance=permute_tokens_for_load_balance,
        context_parallelism_mesh_axis_name=context_parallelism_mesh_axis_name,
        debug=debug,
    )[0]

    return result


flash_attention_triton_single_device.defvjp(
    flash_attention_triton_forward, flash_attention_backward
)


def _make_flash_attention_partition_specs(
    transposed_inputs: bool = False,
    num_mask_tensors: int = 0,
) -> tuple[tuple[Specs, ...], tuple[Axes, ...]]:
    specs = get_attention_specs()
    kv_specs = specs.kv_specs
    query_specs = specs.query_specs

    if transposed_inputs:
        query_axes = (
            query_specs[Axes.batch],
            query_specs[Axes.heads],
            query_specs[Axes.query_sequence],
            query_specs[Axes.head_dim],
        )
        kv_axes = (
            kv_specs[Axes.batch],
            kv_specs[Axes.kv_heads],
            kv_specs[Axes.kv_sequence],
            kv_specs[Axes.head_dim],
        )
    else:
        query_axes = (
            query_specs[Axes.batch],
            query_specs[Axes.query_sequence],
            query_specs[Axes.heads],
            query_specs[Axes.head_dim],
        )
        kv_axes = (
            kv_specs[Axes.batch],
            kv_specs[Axes.kv_sequence],
            kv_specs[Axes.kv_heads],
            kv_specs[Axes.head_dim],
        )

    in_specs = [
        query_axes,  # query
        kv_axes,  # key
        kv_axes,  # value
        (query_specs[Axes.batch], query_specs[Axes.query_sequence]),  # query_pos
        (query_specs[Axes.batch], query_specs[Axes.query_sequence]),  # kv_pos
        (kv_specs[Axes.batch], kv_specs[Axes.kv_sequence]),  # query_sids
        (kv_specs[Axes.batch], kv_specs[Axes.kv_sequence]),  # kv_sids
    ]
    out_specs = query_axes  # out

    if num_mask_tensors == 1:
        in_specs.append((make_attention_mask_spec(dkdv_mask=False),))  # forward mask
    elif num_mask_tensors == 3:
        in_specs.append(
            (
                make_attention_mask_spec(dkdv_mask=False),  # forward mask
                make_attention_mask_spec(dkdv_mask=False),  # backward dq mask
                make_attention_mask_spec(dkdv_mask=True),  # backward dkdv mask
            ),
        )
    elif num_mask_tensors > 0:
        raise ValueError(
            f"The case with num_mask_tensors={num_mask_tensors}" " is not supported."
        )

    return tuple(in_specs), out_specs


def get_identity_with_bwd_dtype_convert(dtype: jnp.dtype):
    @jax.custom_vjp
    def identity_fn(x):
        return x

    def identity_fn_fwd(x):
        return x, None

    def identity_fn_bwd(res, g):
        return (g.astype(dtype),)

    identity_fn.defvjp(identity_fn_fwd, identity_fn_bwd)

    return identity_fn


def flash_attention_triton(
    query: DeviceArray,
    key: DeviceArray,
    value: DeviceArray,
    query_positions: DeviceArray,
    query_segment_ids: DeviceArray,
    kv_positions: DeviceArray,
    kv_segment_ids: DeviceArray,
    mask: tuple[AttentionMask, ...],
    scale: float = 1.0,
    bias: DeviceArray | None = None,
    fwd_params: FlashAttentionParamsConfig | None = None,
    bwd_params: FlashAttentionParamsConfig | None = None,
    assume_sequential_positions: bool = False,
    memory_optimized_gqa_backward: bool = False,
    permute_tokens_for_load_balance: bool = True,
    debug: bool = False,
    mesh: Mesh | None = None,
) -> DeviceArray:
    """
    Sharded version of flash attention to use in multi device configurations.

    Args:
        query (DeviceArray): Query tensor of shape:
            (batch_size, query_seq_length, num_heads, head_dim).
        key (DeviceArray): Key tensor of shape:
            (batch_size, kv_seq_length, num_kv_heads, head_dim).
        value (DeviceArray): Value tensor of shape:
            (batch_size, kv_seq_length, num_kv_heads, head_dim).
        query_positions (DeviceArray): Positions of query tokens with shape:
            (batch_size, query_seq_length).
        query_segment_ids (DeviceArray): Segment IDs for the query tokens with shape:
            (batch_size, query_seq_length).
        kv_positions (DeviceArray): Positions of key/value tokens with shape:
            (batch_size, kv_seq_length).
        kv_segment_ids (DeviceArray): Segment IDs for the key/value tokens with shape:
            (batch_size, kv_seq_length).
        mask (tuple[AttentionMask, ...]): Block-wise attention masks computed by
            create_attention_mask.
        scale (float, optional): Scaling factor for attention scores. Default is 1.0.
        bias (DeviceArray | None, optional): Optional bias tensor to add
            to attention scores. Default is None.
        fwd_params (FlashAttentionParamsConfig | None, optional): Configuration for
            forward pass parameters. Default is None.
        bwd_params (FlashAttentionParamsConfig | None, optional): Configuration for
            backward pass parameters. Default is None.
        assume_sequential_positions (bool, optional): If True, assumes sequential
            positions for tokens, optimising performance. Default is False.
        memory_optimized_gqa_backward (bool, optional): If True, enables
            memory-optimised gradient computation for grouped-query attention.
            Can affect to performance. Can be helpful in the case of small models with
            long sequence length. Default is False.
        permute_tokens_for_load_balance (bool, optional): If True, permutes tokens to
            achieve better load balance across GPUs. Default is True.
        debug (bool, optional): If True, prints low-level IR of the kernel.
            Default is False.
        mesh (Mesh | None, optional): Device mesh configuration for distributed
            execution. If None, it takes the mesh from the global context.
            Default is None.

    Returns:
        DeviceArray: A tensor containing the output tensor (attention-weighted values).

    Notes:
        - Defaults for `fwd_params` and `bwd_params` are set using
            `get_default_flash_attention_params`.
        - If mesh is None and mesh via global context is not provided, raise a
            value error.
    """
    # Check if context parallelism is enabled by determining the mesh axis name.
    context_parallelism_mesh_axis_name = get_query_context_mesh_axis_name(mesh)
    is_context_parallelism = context_parallelism_mesh_axis_name is not None

    # Define default parameters for flash attention if not provided.
    if fwd_params is None:
        fwd_params = get_default_flash_attention_params(backward=False)

    if bwd_params is None:
        bwd_params = get_default_flash_attention_params(backward=True)

    # When using context parallelism, dv and dk returned from
    # flash attention triton kernel in fp32
    # to make more precise AllReduce. This leads to a very weird
    # behavior when using jax.lax.cond in backward pass
    # where one of the branches return fp32 tensor (gradient of the k/v)
    # So we need to convert them back to the original dtype.
    identity_fn = get_identity_with_bwd_dtype_convert(query.dtype)
    key = identity_fn(key)
    value = identity_fn(value)

    # Predefine static parameters for flash attention function.
    flash_attention_fn = functools.partial(
        flash_attention_triton_single_device,
        scale=scale,
        bias=bias,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
        assume_sequential_positions=assume_sequential_positions,
        memory_optimized_gqa_backward=memory_optimized_gqa_backward,
        permute_tokens_for_load_balance=permute_tokens_for_load_balance,
        context_parallelism_mesh_axis_name=context_parallelism_mesh_axis_name,
        debug=debug,
    )

    # Generate input and output sharding specifications for partitioning computations.
    in_specs, out_specs = _make_flash_attention_partition_specs(
        transposed_inputs=True,
        num_mask_tensors=len(mask),
    )

    # Apply the sharding specifications to distribute the flash attention function
    # across devices.
    flash_attention_fn = shard_map(
        flash_attention_fn,
        in_specs=in_specs,
        out_specs=out_specs,
        mesh=mesh,
    )

    # Unpermute key and value tensors if permuted for load balancing during
    # context parallelism. If permute_tokens_for_load_balance is set, input tensors
    # are permuted and key and value tensors are needed to be unpermuted to
    # work correctly.
    if is_context_parallelism and permute_tokens_for_load_balance:
        key, value = unpermute_tokens_context_parallelism(
            (key, value),
            mesh=mesh,
        )

    # Transform query, key, and value tensors for optimised processing of long contexts.
    # (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim).
    query = query.transpose((0, 2, 1, 3))
    key = key.transpose((0, 2, 1, 3))
    value = value.transpose((0, 2, 1, 3))

    # Perform flash attention
    output = flash_attention_fn(
        query,
        key,
        value,
        query_positions,
        query_segment_ids,
        kv_positions,
        kv_segment_ids,
        tuple(mask.flatten() for mask in mask),
    )

    # Revert the output tensor transposition to match the original input format.
    # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim).
    output = output.transpose((0, 2, 1, 3))

    return output
