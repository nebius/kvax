import functools
from functools import reduce
from typing import Callable

import jax
import jax.numpy as jnp
import pytest

from kvax.ops.flash_attention_cudnn import (
    create_cudnn_attn_mask,
    flash_attention_cudnn,
    make_segment_mask,
)
from kvax.ops.flash_attention_triton import flash_attention_triton
from kvax.ops.mask_creator import AttentionMask, create_attention_mask
from kvax.utils.benchmarking import (
    check_outputs_bwd,
    check_outputs_fwd,
    generate_random_segments_lengths,
    make_random_attention_inputs_with_sharding,
    shard_input_data,
)
from kvax.utils.common import PADDING_SEGMENT_ID, FlashAttentionParamsConfig
from kvax.utils.permutation import (
    permute_tokens_context_parallelism,
    unpermute_tokens_context_parallelism,
)
from kvax.utils.sharding import get_query_context_mesh_axis_size
from kvax.utils.specs import attention_specs
from kvax.utils.typing import DeviceArray


def _make_attention_mask(
    query_positions: DeviceArray,
    query_segment_ids: DeviceArray,
    kv_positions: DeviceArray,
    kv_segment_ids: DeviceArray,
) -> DeviceArray:
    def _make_causal_mask(
        query_positions: DeviceArray,
        kv_positions: DeviceArray,
    ) -> DeviceArray:
        return query_positions[..., :, None] >= kv_positions[..., None, :]

    masks = [
        make_segment_mask(query_segment_ids, kv_segment_ids),
        _make_causal_mask(query_positions, kv_positions),
    ]
    return reduce(jnp.logical_and, masks)


def _flash_attention_forward_func(
    attention_ref_func: Callable,
    attention_test_func: Callable,
    batch_size: int = 2,
    query_seq_len: int = 256,
    kv_seq_len: int = 256,
    num_heads: int = 16,
    num_kv_heads: int = 16,
    qk_head_dim: int = 16,
    value_head_dim: int = 16,
    assume_sequential_positions: bool = False,
    random_num_pad_tokens_in_batch: bool = False,
    num_pad_tokens: int = 0,
    shard_kv: bool = True,
    num_segments: int = 1,
    list_of_segment_lengths: list[int] | None = None,
    start_query_position: int | list[int] = 0,
    triton_config: FlashAttentionParamsConfig = FlashAttentionParamsConfig(),
):
    rng = jax.random.PRNGKey(0)
    (
        inputs_dict,
        mesh,
        query_specs,
        kv_specs,
    ) = make_random_attention_inputs_with_sharding(
        batch_size=batch_size,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        num_query_heads=num_heads,
        num_kv_heads=num_kv_heads,
        qk_head_dim=qk_head_dim,
        value_head_dim=value_head_dim,
        shard_kv=shard_kv,
        random_num_pad_tokens_in_batch=random_num_pad_tokens_in_batch,
        num_pad_tokens=num_pad_tokens,
        list_of_segment_lengths=list_of_segment_lengths,
        start_query_position=start_query_position,
        rng=rng,
    )

    query, key, value, q_pos, q_sids, kv_pos, kv_sids = inputs_dict["data"]
    scale = inputs_dict["scale"]

    with attention_specs(query_specs, kv_specs):
        triton_mask = create_attention_mask(
            q_pos,
            q_sids,
            kv_pos,
            kv_sids,
            fwd_params=triton_config,
            mesh=mesh,
        )

        flash_attn_fn = functools.partial(
            attention_test_func,
            mask=triton_mask,
            fwd_params=triton_config,
            scale=scale,
            mesh=mesh,
            assume_sequential_positions=assume_sequential_positions,
        )

        result_triton = jax.jit(flash_attn_fn)(
            query,
            key,
            value,
            q_pos,
            q_sids,
            kv_pos,
            kv_sids,
        ).block_until_ready()

        if num_segments > 1:
            cudnn_mask = create_cudnn_attn_mask(
                q_sids,
                kv_sids,
            )
        else:
            cudnn_mask = None

        result_ref = attention_ref_func(
            query,
            key,
            value,
            mask=cudnn_mask,
            query_segment_ids=q_sids,
            kv_segment_ids=kv_sids,
            scale=scale,
            mesh=mesh,
        )

    result_triton = jnp.where(
        q_sids[:, :, None, None] == PADDING_SEGMENT_ID, 0, result_triton
    )
    result_ref = jnp.where(
        q_sids[:, :, None, None] == PADDING_SEGMENT_ID, 0, result_ref
    )

    return {
        "reference": result_ref,
        "triton": result_triton,
    }


def _flash_attention_backward_func(
    attention_ref_func: Callable,
    attention_func_to_test: Callable,
    batch_size: int = 2,
    query_seq_len: int = 256,
    kv_seq_len: int = 256,
    num_heads: int = 16,
    num_kv_heads: int = 16,
    qk_head_dim: int = 16,
    value_head_dim: int = 16,
    random_num_pad_tokens_in_batch: bool = False,
    num_pad_tokens: int = 0,
    assume_sequential_positions: bool = False,
    memory_optimized_gqa_backward: bool = False,
    shard_kv: bool = True,
    num_segments: int = 1,
    triton_config_fwd: FlashAttentionParamsConfig = FlashAttentionParamsConfig(),
    triton_config_bwd: FlashAttentionParamsConfig = FlashAttentionParamsConfig(),
):
    rng = jax.random.PRNGKey(0)
    (
        inputs_dict,
        mesh,
        query_specs,
        kv_specs,
    ) = make_random_attention_inputs_with_sharding(
        batch_size=batch_size,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        num_query_heads=num_heads,
        num_kv_heads=num_kv_heads,
        qk_head_dim=qk_head_dim,
        value_head_dim=value_head_dim,
        shard_kv=shard_kv,
        random_num_pad_tokens_in_batch=random_num_pad_tokens_in_batch,
        num_pad_tokens=num_pad_tokens,
        rng=rng,
    )

    query, key, value, q_pos, q_sids, kv_pos, kv_sids = inputs_dict["data"]
    scale = inputs_dict["scale"]

    with attention_specs(query_specs, kv_specs):
        # We sum the output tensor at the end to avoid the error:
        # TypeError: Gradient only defined for scalar-output functions.
        def mha_triton(*args, **kwargs):
            return attention_func_to_test(
                *args,
                **kwargs,
                scale=scale,
                fwd_params=triton_config_fwd,
                bwd_params=triton_config_bwd,
                assume_sequential_positions=assume_sequential_positions,
                memory_optimized_gqa_backward=memory_optimized_gqa_backward,
                mesh=mesh,
            ).sum()

        def mha_ref(*args, **kwargs):
            return attention_ref_func(
                *args,
                **kwargs,
                scale=scale,
            ).sum()

        mask = create_attention_mask(
            q_pos,
            q_sids,
            kv_pos,
            kv_sids,
            fwd_params=triton_config_fwd,
            bwd_params=triton_config_bwd,
            calc_bwd_mask=True,
            mesh=mesh,
        )

        mha_grad_triton = jax.grad(mha_triton, argnums=(0, 1, 2))
        mha_grad_ref = jax.grad(mha_ref, argnums=(0, 1, 2))

        result_triton = mha_grad_triton(
            query,
            key,
            value,
            q_pos,
            q_sids,
            kv_pos,
            kv_sids,
            mask=mask,
        )

        if num_segments > 1:
            cudnn_mask = create_cudnn_attn_mask(
                q_sids,
                kv_sids,
            )
        else:
            cudnn_mask = None

        result_ref = mha_grad_ref(
            query,
            key,
            value,
            q_sids,
            kv_sids,
            mask=cudnn_mask,
            mesh=mesh,
        )

    dquery_triton, dkey_triton, dvalue_triton = result_triton
    dquery_ref, dkey_ref, dvalue_ref = result_ref

    dquery_triton = jnp.where(
        q_sids[:, :, None, None] == PADDING_SEGMENT_ID, 0, dquery_triton
    )
    dkey_triton = jnp.where(
        kv_sids[:, :, None, None] == PADDING_SEGMENT_ID, 0, dkey_triton
    )
    dvalue_triton = jnp.where(
        kv_sids[:, :, None, None] == PADDING_SEGMENT_ID, 0, dvalue_triton
    )

    dquery_ref = jnp.where(
        q_sids[:, :, None, None] == PADDING_SEGMENT_ID, 0, dquery_ref
    )
    dkey_ref = jnp.where(kv_sids[:, :, None, None] == PADDING_SEGMENT_ID, 0, dkey_ref)
    dvalue_ref = jnp.where(
        kv_sids[:, :, None, None] == PADDING_SEGMENT_ID, 0, dvalue_ref
    )

    result_ref = dquery_ref, dkey_ref, dvalue_ref
    result_triton = dquery_triton, dkey_triton, dvalue_triton

    return {
        "reference": result_ref,
        "triton": result_triton,
    }


def _masks_equal(
    attn_mask1: AttentionMask,
    attn_mask2: AttentionMask,
):
    assert jnp.all(attn_mask1.lower_bounds == attn_mask2.lower_bounds)
    assert jnp.all(attn_mask1.lower_full_bounds == attn_mask2.lower_full_bounds)
    assert jnp.all(attn_mask1.upper_full_bounds == attn_mask2.upper_full_bounds)
    assert jnp.all(attn_mask1.upper_bounds == attn_mask2.upper_bounds)


@pytest.mark.parametrize(
    "query_seq_len, kv_seq_len, num_heads, num_kv_heads, qk_head_dim,"
    "value_head_dim, num_pad_tokens, random_num_pad_tokens_in_batch,"
    "assume_sequential_positions",
    [
        (384, 384, 8, 8, 16, 16, 0, False, False),
        (400, 400, 8, 8, 16, 16, 0, False, False),
        (384, 256, 8, 8, 16, 16, 0, False, False),
        (384, 300, 8, 8, 16, 16, 0, False, False),
        (400, 256, 8, 8, 16, 16, 0, False, False),
        (400, 300, 8, 8, 16, 16, 0, False, False),
        (1, 256, 8, 8, 16, 16, 0, False, False),
        (1, 300, 8, 8, 16, 16, 0, False, False),
        (256, 1, 8, 8, 16, 16, 0, False, False),
        (300, 1, 8, 8, 16, 16, 0, False, False),
        (1, 3, 8, 8, 16, 16, 0, False, False),
        (384, 384, 8, 8, 16, 16, 100, False, False),
        (384, 384, 8, 8, 16, 16, 383, False, False),
        (384, 384, 8, 8, 16, 16, 383, True, False),
        (384, 384, 8, 8, 16, 16, 100, False, True),
        (384, 384, 8, 8, 16, 16, 383, False, True),
        (384, 384, 8, 8, 16, 16, 383, True, True),
        (384, 384, 8, 8, 16, 16, 0, False, True),
        (256, 256, 8, 8, 16, 16, 0, False, True),
        (384, 384, 16, 8, 16, 16, 0, False, False),
        (1, 256, 16, 8, 16, 16, 0, False, False),
        (256, 1, 16, 8, 16, 16, 0, False, False),
        (384, 384, 16, 1, 16, 16, 0, False, False),
        (1, 256, 16, 1, 16, 16, 0, False, False),
        (384, 384, 16, 8, 16, 16, 100, False, False),
        (384, 384, 16, 8, 16, 16, 383, False, False),
        (384, 384, 16, 8, 16, 16, 383, True, False),
        (384, 384, 16, 8, 16, 16, 100, False, True),
        (384, 384, 16, 8, 16, 16, 383, False, True),
        (384, 384, 16, 8, 16, 16, 383, True, True),
        (384, 384, 16, 8, 16, 16, 0, False, True),
        (384, 384, 16, 1, 16, 16, 0, False, True),
    ],
)
def test_flash_multi_head_attention(
    query_seq_len: int,
    kv_seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    qk_head_dim: int,
    value_head_dim: int,
    num_pad_tokens: int,
    random_num_pad_tokens_in_batch: bool,
    assume_sequential_positions: bool,
):
    attention_ref_func = functools.partial(
        flash_attention_cudnn,
        is_causal=True,
    )
    num_devices = len(jax.devices())
    shard_kv = num_kv_heads >= num_devices

    results = _flash_attention_forward_func(
        attention_ref_func,
        flash_attention_triton,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        random_num_pad_tokens_in_batch=random_num_pad_tokens_in_batch,
        assume_sequential_positions=assume_sequential_positions,
        qk_head_dim=qk_head_dim,
        value_head_dim=value_head_dim,
        num_pad_tokens=num_pad_tokens,
        shard_kv=shard_kv,
    )

    check_outputs_fwd(results["reference"], results["triton"])


@pytest.mark.parametrize(
    "query_seq_len, kv_seq_len, num_heads, num_kv_heads, qk_head_dim, value_head_dim,"
    "num_pad_tokens, random_num_pad_tokens_in_batch, memory_optimized,"
    "assume_sequential_positions",
    [
        (384, 384, 8, 8, 64, 64, 0, False, False, False),
        (384, 384, 8, 8, 128, 128, 0, False, False, False),
        (256, 256, 8, 8, 128, 128, 0, False, False, False),
        (256, 256, 8, 8, 128, 128, 200, False, False, False),
        (256, 256, 8, 8, 128, 128, 200, False, False, True),
        (384, 384, 8, 8, 64, 64, 0, False, False, True),
        (384, 384, 8, 8, 128, 128, 0, False, False, True),
        (256, 256, 8, 8, 128, 128, 0, False, False, True),
        (256, 256, 16, 8, 128, 128, 0, False, False, False),
        (256, 256, 16, 8, 128, 128, 0, False, False, True),
        (256, 256, 16, 1, 128, 128, 0, False, False, False),
        (256, 256, 16, 1, 128, 128, 0, False, False, True),
        (256, 256, 16, 8, 128, 128, 0, False, True, False),
        (256, 256, 16, 8, 128, 128, 0, False, True, True),
        # query_seq_len != kv_seq_len exercises the separate backward kernel
        # launches (num_query_blocks != num_kv_blocks).
        (384, 256, 8, 8, 64, 64, 0, False, False, False),
        (256, 384, 8, 8, 64, 64, 0, False, False, False),
    ],
)
def test_flash_multi_head_attention_backward(
    query_seq_len: int,
    kv_seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    qk_head_dim: int,
    value_head_dim: int,
    num_pad_tokens: int,
    random_num_pad_tokens_in_batch: bool,
    memory_optimized: bool,
    assume_sequential_positions: bool,
):
    attention_ref_func = functools.partial(
        flash_attention_cudnn,
        is_causal=True,
    )
    num_devices = len(jax.devices())
    shard_kv = num_kv_heads >= num_devices

    results = _flash_attention_backward_func(
        attention_ref_func,
        flash_attention_triton,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        qk_head_dim=qk_head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        value_head_dim=value_head_dim,
        num_pad_tokens=num_pad_tokens,
        random_num_pad_tokens_in_batch=random_num_pad_tokens_in_batch,
        assume_sequential_positions=assume_sequential_positions,
        memory_optimized_gqa_backward=memory_optimized,
        shard_kv=shard_kv,
    )

    dq_triton, dk_triton, dv_triton = results["triton"]
    dq_ref, dk_ref, dv_ref = results["reference"]

    dq_ref = dq_ref[:, : query_seq_len - num_pad_tokens, :, :]
    dk_ref = dk_ref[:, : kv_seq_len - num_pad_tokens, :, :]
    dv_ref = dv_ref[:, : kv_seq_len - num_pad_tokens, :, :]

    dq_triton = dq_triton[:, : query_seq_len - num_pad_tokens, :, :]
    dk_triton = dk_triton[:, : kv_seq_len - num_pad_tokens, :, :]
    dv_triton = dv_triton[:, : kv_seq_len - num_pad_tokens, :, :]

    check_kwargs = {}

    # Relax constraints for the MQA case
    if num_kv_heads == 1:
        check_kwargs = {
            "kv_atol": 5e-2,
        }

    check_outputs_bwd(results["reference"], results["triton"], **check_kwargs)


@pytest.mark.parametrize(
    "query_block_size, kv_block_size, num_warps,"
    "num_segments, assume_sequential_positions",
    [
        (128, 64, 8, 1, False),
        (128, 128, 8, 1, False),
        (128, 64, 8, 1, True),
        (128, 128, 8, 1, True),
        (128, 128, 8, 3, False),
        (128, 128, 8, 12, False),
        (128, 128, 8, 3, True),
        (128, 128, 8, 12, True),
    ],
)
def test_flash_multi_head_attention_params_on_long_context(
    query_block_size: int,
    kv_block_size: int,
    num_warps: int,
    num_segments: int,
    assume_sequential_positions: bool,
):
    batch_size = 1
    query_seq_len = 32768
    kv_seq_len = 32768
    value_head_dim = 128
    qk_head_dim = 128
    triton_config = FlashAttentionParamsConfig(
        **{
            "query_block_size": query_block_size,
            "kv_block_size": kv_block_size,
            "num_warps": num_warps,
            "num_stages": 3,
        }
    )
    num_devices = len(jax.devices())

    attention_ref_func = functools.partial(
        flash_attention_cudnn,
        is_causal=True,
    )

    results = _flash_attention_forward_func(
        attention_ref_func,
        flash_attention_triton,
        batch_size=batch_size,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        assume_sequential_positions=assume_sequential_positions,
        qk_head_dim=qk_head_dim,
        value_head_dim=value_head_dim,
        triton_config=triton_config,
        list_of_segment_lengths=generate_random_segments_lengths(
            query_seq_len,
            num_segments,
        ),
        num_segments=num_segments,
        num_heads=num_devices,
        num_kv_heads=num_devices,
    )

    check_outputs_fwd(results["reference"], results["triton"])


@pytest.mark.parametrize(
    "num_pad_tokens, num_segments, assume_sequential_positions,"
    "permute_tokens_for_load_balance",
    [
        (0, 1, True, False),
        (10000, 1, True, False),
        (0, 1, False, False),
        (10000, 1, False, False),
        (0, 1, True, True),
        (10000, 1, True, True),
        (0, 3, True, True),
        (10000, 12, True, True),
    ],
)
def test_flash_multi_head_attention_params_context_parallelism_fwd(
    num_pad_tokens: int,
    num_segments: int,
    assume_sequential_positions: bool,
    permute_tokens_for_load_balance: bool,
):
    batch_size = 2
    query_seq_len = 32768
    kv_seq_len = 32768
    value_head_dim = 128
    num_heads = 8
    qk_head_dim = value_head_dim

    triton_config = FlashAttentionParamsConfig(
        **{
            "query_block_size": 128,
            "kv_block_size": 128,
            "num_warps": 8,
            "num_stages": 3,
        }
    )

    rng = jax.random.PRNGKey(0)
    (
        inputs_dict,
        mesh,
        query_specs,
        kv_specs,
    ) = make_random_attention_inputs_with_sharding(
        batch_size=batch_size,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        num_query_heads=num_heads,
        num_kv_heads=num_heads,
        qk_head_dim=qk_head_dim,
        value_head_dim=value_head_dim,
        shard_kv=False,
        num_pad_tokens=num_pad_tokens,
        parallelism_type="context",
        list_of_segment_lengths=generate_random_segments_lengths(
            query_seq_len,
            num_segments,
        ),
        rng=rng,
    )

    fn_args = inputs_dict["data"]
    scale = inputs_dict["scale"]
    _, _, _, _, q_sids, _, _ = fn_args

    flash_attn_fn_common = functools.partial(
        flash_attention_triton,
        fwd_params=triton_config,
        scale=scale,
        assume_sequential_positions=assume_sequential_positions,
        permute_tokens_for_load_balance=permute_tokens_for_load_balance,
    )

    with attention_specs(query_specs, kv_specs):
        if permute_tokens_for_load_balance:
            query, key, value, q_pos, q_sids, kv_pos, kv_sids = fn_args
            fn_args_cp = permute_tokens_context_parallelism(
                (query, key, value, q_pos, q_sids),
                mesh=mesh,
            )
            fn_args_cp = (*fn_args_cp, kv_pos, kv_sids)
        else:
            fn_args_cp = fn_args

        _, _, _, q_pos, q_sids, kv_pos, kv_sids = fn_args_cp
        mask_cp = create_attention_mask(
            q_pos,
            q_sids,
            kv_pos,
            kv_sids,
            fwd_params=triton_config,
            mesh=mesh,
        )

        flash_attn_fn_cp = functools.partial(
            flash_attn_fn_common,
            mask=mask_cp,
            mesh=mesh,
        )

        result_cp = jax.jit(flash_attn_fn_cp)(*fn_args_cp).block_until_ready()

        if permute_tokens_for_load_balance:
            result_cp = unpermute_tokens_context_parallelism(
                result_cp,
                mesh=mesh,
            )

    inputs_dict, mesh, query_specs, kv_specs = shard_input_data(
        inputs_dict,
        parallelism_type="tensor",
    )
    fn_args = inputs_dict["data"]

    with attention_specs(query_specs, kv_specs):
        _, _, _, q_pos, q_sids, kv_pos, kv_sids = fn_args
        mask_tp = create_attention_mask(
            q_pos,
            q_sids,
            kv_pos,
            kv_sids,
            fwd_params=triton_config,
            mesh=mesh,
        )

        flash_attn_fn_tp = functools.partial(
            flash_attn_fn_common,
            mask=mask_tp,
            mesh=mesh,
        )

        result_tp = jax.jit(flash_attn_fn_tp)(*fn_args).block_until_ready()

    result_cp = jnp.where(q_sids[:, :, None, None] == PADDING_SEGMENT_ID, 0, result_cp)
    result_tp = jnp.where(q_sids[:, :, None, None] == PADDING_SEGMENT_ID, 0, result_tp)

    check_outputs_fwd(result_cp, result_tp)


@pytest.mark.parametrize(
    "num_pad_tokens, num_segments, assume_sequential_positions,"
    "permute_tokens_for_load_balance",
    [
        (0, 1, True, False),
        (10000, 1, True, False),
        (0, 1, False, False),
        (10000, 1, False, False),
        (0, 1, True, True),
        (10000, 1, True, True),
        (0, 3, True, False),
        (10000, 12, True, False),
        (0, 3, False, False),
        (10000, 12, False, False),
        (0, 3, False, True),
        (10000, 12, False, True),
        (0, 3, True, True),
        (10000, 12, True, True),
    ],
)
def test_flash_multi_head_attention_params_context_parallelism_bwd(
    num_pad_tokens: int,
    num_segments: int,
    assume_sequential_positions: bool,
    permute_tokens_for_load_balance: bool,
):
    batch_size = 2
    query_seq_len = 32768
    kv_seq_len = 32768
    value_head_dim = 128
    num_heads = 8
    qk_head_dim = value_head_dim

    fwd_params = FlashAttentionParamsConfig(
        **{
            "query_block_size": 128,
            "kv_block_size": 128,
            "num_warps": 8,
            "num_stages": 3,
        }
    )

    bwd_params = FlashAttentionParamsConfig(
        **{
            "query_block_size": 64,
            "kv_block_size": 128,
            "num_warps": 8,
            "num_stages": 3,
        }
    )

    rng = jax.random.PRNGKey(0)
    (
        inputs_dict,
        mesh,
        query_specs,
        kv_specs,
    ) = make_random_attention_inputs_with_sharding(
        batch_size=batch_size,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        num_query_heads=num_heads,
        num_kv_heads=num_heads,
        qk_head_dim=qk_head_dim,
        value_head_dim=value_head_dim,
        shard_kv=False,
        num_pad_tokens=num_pad_tokens,
        parallelism_type="context",
        list_of_segment_lengths=generate_random_segments_lengths(
            query_seq_len,
            num_segments,
        ),
        rng=rng,
    )

    fn_args = inputs_dict["data"]
    scale = inputs_dict["scale"]

    def flash_attention_fwd_bwd(*args, **kwargs):
        return flash_attention_triton(
            *args,
            **kwargs,
            fwd_params=fwd_params,
            bwd_params=bwd_params,
            scale=scale,
            assume_sequential_positions=assume_sequential_positions,
            permute_tokens_for_load_balance=permute_tokens_for_load_balance,
        ).sum()

    with attention_specs(query_specs, kv_specs):
        if permute_tokens_for_load_balance:
            query, key, value, q_pos, q_sids, kv_pos, kv_sids = fn_args
            fn_args_cp = permute_tokens_context_parallelism(
                (query, key, value, q_pos, q_sids),
                mesh=mesh,
            )
            fn_args_cp = (*fn_args_cp, kv_pos, kv_sids)
        else:
            fn_args_cp = fn_args

        _, _, _, q_pos, q_sids, kv_pos, kv_sids = fn_args_cp
        mask_cp = create_attention_mask(
            q_pos,
            q_sids,
            kv_pos,
            kv_sids,
            fwd_params=fwd_params,
            bwd_params=bwd_params,
            calc_bwd_mask=True,
            mesh=mesh,
        )

        flash_attn_fn_cp = functools.partial(
            flash_attention_fwd_bwd,
            mask=mask_cp,
            mesh=mesh,
        )

        result_cp = jax.jit(jax.grad(flash_attn_fn_cp, argnums=(0, 1, 2)))(*fn_args_cp)
        result_cp[0].block_until_ready()

        if permute_tokens_for_load_balance:
            result_cp = unpermute_tokens_context_parallelism(
                result_cp,
                mesh=mesh,
            )

    inputs_dict, mesh, query_specs, kv_specs = shard_input_data(
        inputs_dict,
        parallelism_type="tensor",
    )

    fn_args = inputs_dict["data"]

    with attention_specs(query_specs, kv_specs):
        _, _, _, q_pos, q_sids, kv_pos, kv_sids = fn_args
        mask_tp = create_attention_mask(
            q_pos,
            q_sids,
            kv_pos,
            kv_sids,
            fwd_params=fwd_params,
            bwd_params=bwd_params,
            calc_bwd_mask=True,
            mesh=mesh,
        )

        flash_attn_fn_tp = functools.partial(
            flash_attention_fwd_bwd,
            mask=mask_tp,
            mesh=mesh,
        )

        result_tp = jax.jit(jax.grad(flash_attn_fn_tp, argnums=(0, 1, 2)))(*fn_args)
        result_tp[0].block_until_ready()

    dq_cp, dk_cp, dv_cp = result_cp
    dq_tp, dk_tp, dv_tp = result_tp

    dq_cp = dq_cp[:, : query_seq_len - num_pad_tokens, :, :]
    dk_cp = dk_cp[:, : kv_seq_len - num_pad_tokens, :, :]
    dv_cp = dv_cp[:, : kv_seq_len - num_pad_tokens, :, :]

    dq_tp = dq_tp[:, : query_seq_len - num_pad_tokens, :, :]
    dk_tp = dk_tp[:, : kv_seq_len - num_pad_tokens, :, :]
    dv_tp = dv_tp[:, : kv_seq_len - num_pad_tokens, :, :]

    check_outputs_bwd(
        (dq_cp, dk_cp, dv_cp),
        (dq_tp, dk_tp, dv_tp),
    )


@pytest.mark.parametrize(
    "num_pad_tokens, num_segments, permute_tokens_for_load_balance," "skip_pad_tokens",
    [
        (0, 1, True, False),
        (10000, 1, True, False),
        (20645, 1, False, False),
        (5632, 3, True, True),
        (6759, 5, True, False),
        (1024, 12, True, False),
    ],
)
def test_attention_mask(
    num_pad_tokens: int,
    num_segments: int,
    permute_tokens_for_load_balance: bool,
    skip_pad_tokens: bool,
):
    batch_size = 2
    query_seq_len = 32768
    kv_seq_len = 32768
    num_heads = 8
    qk_head_dim = 1
    value_head_dim = 1

    def _create_attention_mask_ref(
        q_pos: DeviceArray,
        q_sids: DeviceArray,
        kv_pos: DeviceArray,
        kv_sids: DeviceArray,
        fwd_params: FlashAttentionParamsConfig,
        bwd_params: FlashAttentionParamsConfig,
        cp_shard: int,
        skip_pad_tokens: bool,
    ):
        def _attn_mask_to_block_mask(
            attn_mask: DeviceArray,
            max_blocks_in_row: int,
            kv_mask: bool = False,
        ):
            block_mask = attn_mask.any(axis=[3, 4])
            full_block_mask = attn_mask.all(axis=[3, 4])

            lower = jnp.argmax(block_mask, axis=-1)
            upper = max_blocks_in_row - jnp.argmax(
                jnp.flip(block_mask, axis=-1), axis=-1
            )

            no_true_rows = ~jnp.any(block_mask, axis=-1)
            upper = jnp.where(no_true_rows, 0, upper)
            lower = jnp.where(no_true_rows, 0, lower)

            lower_full = jnp.argmax(full_block_mask, axis=-1)
            upper_full = max_blocks_in_row - jnp.argmax(
                jnp.flip(full_block_mask, axis=-1), axis=-1
            )
            no_true_rows = ~jnp.any(full_block_mask, axis=-1)
            if kv_mask:
                lower_full = jnp.where(no_true_rows, upper, lower_full)
                upper_full = jnp.where(no_true_rows, upper, upper_full)
            else:
                lower_full = jnp.where(no_true_rows, lower, lower_full)
                upper_full = jnp.where(no_true_rows, lower, upper_full)

            # add shard axis
            lower = lower.reshape(batch_size, 1, -1)
            lower_full = lower_full.reshape(batch_size, 1, -1)
            upper_full = upper_full.reshape(batch_size, 1, -1)
            upper = upper.reshape(batch_size, 1, -1)

            return AttentionMask(
                lower_bounds=lower,
                lower_full_bounds=lower_full,
                upper_full_bounds=upper_full,
                upper_bounds=upper,
            )

        attn_mask = _make_attention_mask(
            q_pos,
            q_sids,
            kv_pos,
            kv_sids,
        )

        if skip_pad_tokens:
            pad_mask1 = jnp.broadcast_to(
                q_sids[..., :, None] != PADDING_SEGMENT_ID,
                (batch_size, query_seq_len, kv_seq_len),
            )
            attn_mask &= pad_mask1
            pad_mask2 = jnp.broadcast_to(
                kv_sids[..., None, :] != PADDING_SEGMENT_ID,
                (batch_size, query_seq_len, kv_seq_len),
            )
            attn_mask &= pad_mask2

        # fwd_mask
        attn_mask_fwd = attn_mask.reshape(
            batch_size,
            query_seq_len // fwd_params.query_block_size,
            fwd_params.query_block_size,
            kv_seq_len // fwd_params.kv_block_size,
            fwd_params.kv_block_size,
        )
        attn_mask_fwd = attn_mask_fwd.transpose(0, 1, 3, 2, 4)
        fwd_mask = _attn_mask_to_block_mask(
            attn_mask_fwd,
            kv_seq_len // fwd_params.kv_block_size,
        )

        # q_bwd_mask
        attn_mask_q_bwd = attn_mask.reshape(
            batch_size,
            query_seq_len // bwd_params.kv_block_size,
            bwd_params.kv_block_size,
            kv_seq_len // bwd_params.query_block_size,
            bwd_params.query_block_size,
        )
        attn_mask_q_bwd = attn_mask_q_bwd.transpose(0, 1, 3, 2, 4)
        q_mask = _attn_mask_to_block_mask(
            attn_mask_q_bwd,
            kv_seq_len // bwd_params.query_block_size,
        )

        # kv_bwd_mask
        attn_mask = attn_mask.transpose(0, 2, 1)
        q_block = bwd_params.query_block_size
        attn_mask_kv_bwd = attn_mask.reshape(
            batch_size,
            kv_seq_len // bwd_params.kv_block_size,
            bwd_params.kv_block_size,
            query_seq_len // q_block,
            q_block,
        )
        attn_mask_kv_bwd = attn_mask_kv_bwd.transpose(0, 1, 3, 2, 4)
        lower = jnp.zeros(
            shape=(batch_size, cp_shard, kv_seq_len // bwd_params.kv_block_size),
            dtype=jnp.int32,
        )
        lower_full = jnp.zeros(
            shape=(batch_size, cp_shard, kv_seq_len // bwd_params.kv_block_size),
            dtype=jnp.int32,
        )
        upper_full = jnp.zeros(
            shape=(batch_size, cp_shard, kv_seq_len // bwd_params.kv_block_size),
            dtype=jnp.int32,
        )
        upper = jnp.zeros(
            shape=(batch_size, cp_shard, kv_seq_len // bwd_params.kv_block_size),
            dtype=jnp.int32,
        )

        query_shard = query_seq_len // bwd_params.query_block_size // cp_shard
        for i in range(cp_shard):
            shard_kv_mask = _attn_mask_to_block_mask(
                attn_mask_kv_bwd[:, :, i * query_shard : (i + 1) * query_shard, :, :],
                query_shard,
                kv_mask=True,
            )
            lower = lower.at[:, i, :].set(
                shard_kv_mask.lower_bounds.reshape(batch_size, -1)
            )
            lower_full = lower_full.at[:, i, :].set(
                shard_kv_mask.lower_full_bounds.reshape(batch_size, -1)
            )
            upper_full = upper_full.at[:, i, :].set(
                shard_kv_mask.upper_full_bounds.reshape(batch_size, -1)
            )
            upper = upper.at[:, i, :].set(
                shard_kv_mask.upper_bounds.reshape(batch_size, -1)
            )

        kv_mask = AttentionMask(
            lower_bounds=lower,
            lower_full_bounds=lower_full,
            upper_full_bounds=upper_full,
            upper_bounds=upper,
        )

        return (fwd_mask, q_mask, kv_mask)

    rng = jax.random.PRNGKey(0)

    (
        inputs_dict,
        mesh,
        query_specs,
        kv_specs,
    ) = make_random_attention_inputs_with_sharding(
        batch_size=batch_size,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        num_query_heads=num_heads,
        num_kv_heads=num_heads,
        qk_head_dim=qk_head_dim,
        value_head_dim=value_head_dim,
        num_pad_tokens=num_pad_tokens,
        parallelism_type="context",
        list_of_segment_lengths=generate_random_segments_lengths(
            query_seq_len,
            num_segments,
        ),
        shard_kv=False,
        rng=rng,
    )

    fwd_params_dict = {
        "query_block_size": 128,
        "kv_block_size": 128,
        "num_warps": 8,
        "num_stages": 3,
    }

    bwd_params_dict = {
        "query_block_size": 64,
        "kv_block_size": 128,
        "num_warps": 8,
        "num_stages": 3,
    }
    fwd_params_config = FlashAttentionParamsConfig(**fwd_params_dict)
    bwd_params_config = FlashAttentionParamsConfig(**bwd_params_dict)

    _, _, _, q_pos, q_sids, kv_pos, kv_sids = inputs_dict["data"]

    with attention_specs(query_specs, kv_specs):
        if permute_tokens_for_load_balance:
            q_pos, q_sids = permute_tokens_context_parallelism(
                (q_pos, q_sids),
                mesh=mesh,
            )

        attn_mask_triton = create_attention_mask(
            q_pos,
            q_sids,
            kv_pos,
            kv_sids,
            fwd_params=fwd_params_config,
            bwd_params=bwd_params_config,
            calc_bwd_mask=True,
            skip_pad_tokens=skip_pad_tokens,
            mesh=mesh,
        )

        attn_mask_ref = _create_attention_mask_ref(
            q_pos,
            q_sids,
            kv_pos,
            kv_sids,
            fwd_params_config,
            bwd_params_config,
            cp_shard=get_query_context_mesh_axis_size(mesh),
            skip_pad_tokens=skip_pad_tokens,
        )

    _masks_equal(attn_mask_triton[0], attn_mask_ref[0])
    _masks_equal(attn_mask_triton[1], attn_mask_ref[1])
    _masks_equal(attn_mask_triton[2], attn_mask_ref[2])


@pytest.mark.parametrize(
    "kv_seq_len, num_kv_heads, start_query_position",
    [
        (384, 16, 300),
        (512, 16, 500),
        (500, 16, 300),
        (384, 8, 300),
        (512, 8, 500),
        (500, 8, 300),
        (4096, 16, (1200, 200, 1000, 3500)),
        (4096, 16, 5),
        (32768, 16, (12000, 20000, 1000, 3500)),
    ],
)
def test_flash_multi_head_attention_inference(
    kv_seq_len: int,
    num_kv_heads: int,
    start_query_position: list[int] | int,
):
    query_seq_len = 1
    batch_size = 4
    num_heads = 16

    attention_ref_func = functools.partial(
        flash_attention_cudnn,
        is_causal=False,
    )

    results = _flash_attention_forward_func(
        attention_ref_func,
        flash_attention_triton,
        batch_size=batch_size,
        query_seq_len=query_seq_len,
        kv_seq_len=kv_seq_len,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        start_query_position=start_query_position,
    )

    check_outputs_fwd(results["reference"], results["triton"])
