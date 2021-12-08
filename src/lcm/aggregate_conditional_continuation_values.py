import jax
import jax.numpy as jnp


def aggregate_conditional_continuation_values(
    values,
    shock_type,
    agg_axes=None,
    segment_info=None,
    shock_params=None,
):
    out = values
    if shock_type is None:
        if agg_axes is not None:
            out = out.max(axis=agg_axes)
        if segment_info is not None:
            out = _segment_max_over_last_axis(out, segment_info)

    elif shock_type == "extreme_value":
        scale = shock_params
        if agg_axes is not None:
            out = scale * jax.scipy.special.logsumexp(out / scale, axis=agg_axes)
        if segment_info is not None:
            out = _segment_extreme_value_emax_over_last_axis(out, scale, segment_info)
    else:
        raise ValueError("Invalid shock_type: {shock_type}.")

    return out


def _segment_max_over_last_axis(a, segment_info):
    a_t = _put_last_axis_first(a)
    segmax_t = jax.ops.segment_max(
        data=a_t,
        indices_are_sorted=True,
        **segment_info,
    )
    segmax = _put_first_axis_last(segmax_t)
    return segmax


def _segment_extreme_value_emax_over_last_axis(a, scale, segment_info):
    a_t = _put_last_axis_first(a)

    a_t_scaled = a_t / scale

    lse = _segment_logsumexp(a_t_scaled, segment_info)

    emax_t = scale * lse
    emax = _put_first_axis_last(emax_t)
    return emax


def _segment_logsumexp(a, segment_info):
    segmax = jax.ops.segment_max(
        data=a,
        indices_are_sorted=True,
        **segment_info,
    )

    exp = jnp.exp(a - segmax[segment_info["segment_ids"]])

    summed = jax.ops.segment_sum(
        data=exp,
        indices_are_sorted=True,
        **segment_info,
    )
    out = segmax + jnp.log(summed)
    return out


def _put_last_axis_first(a):
    last_axis = a.ndim - 1
    transpose_info = tuple([last_axis] + list(range(last_axis)))
    out = jnp.transpose(a, transpose_info)
    return out


def _put_first_axis_last(a):
    transpose_info = tuple(list(range(1, a.ndim)) + [0])
    out = jnp.transpose(a, transpose_info)
    return out
