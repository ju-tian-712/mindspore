# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
Function operator.

A collection of function to build neural networks or to compute functions.
"""

from . import (
    array_func,
    parameter_func,
    math_func,
    nn_func,
    linalg_func,
    clip_func,
)
from .array_func import (
    unique,
    unique_with_pad,
    eye,
    matrix_band_part,
    padding,
    fill,
    full,
    full_like,
    chunk,
    tile,
    size,
    ones,
    ones_like,
    zeros,
    zeros_like,
    shape,
    shape_,
    ger,
    dyn_shape,
    hamming_window,
    ravel,
    reshape,
    reshape_,
    reverse,
    reverse_sequence,
    flatten,
    cat,
    concat,
    stack,
    unbind,
    unstack,
    tensor_slice,
    strided_slice,
    slice,
    slice_scatter,
    select_scatter,
    scalar_to_array,
    scalar_to_tensor,
    tuple_to_array,
    expand_dims,
    squeeze,
    unsqueeze,
    transpose,
    scatter_nd,
    scatter_nd_add,
    scatter_nd_sub,
    scatter_nd_mul,
    scatter_nd_div,
    scatter_nd_max,
    scatter_nd_min,
    tril,
    triu,
    gather,
    gather_d,
    gather_elements,
    gather_nd,
    is_tensor,
    scalar_cast,
    masked_fill,
    narrow,
    tensor_scatter_add,
    tensor_scatter_sub,
    tensor_scatter_mul,
    unique_consecutive,
    tensor_scatter_div,
    tensor_scatter_max,
    tensor_scatter_min,
    tensor_scatter_elements,
    scatter,
    scatter_add,
    scatter_mul,
    scatter_max,
    scatter_min,
    scatter_div,
    scatter_update,
    unsorted_segment_min,
    unsorted_segment_max,
    unsorted_segment_prod,
    nonzero,
    is_nonzero,
    space_to_batch_nd,
    batch_to_space_nd,
    arange,
    select,
    one_hot,
    matrix_diag,
    matrix_diag_part,
    matrix_set_diag,
    diag,
    diagflat,
    masked_select,
    where,
    meshgrid,
    affine_grid,
    fills,
    broadcast_to,
    unsorted_segment_sum,
    col2im,
    split,
    tensor_split,
    vsplit,
    hsplit,
    dsplit,
    index_fill,
    index_select,
    max,
    argmax,
    min,
    population_count,
    topk,
    expand,
    fold,
    unfold,
    diagonal,
    diagonal_scatter,
    lstsq,
    mvlgamma,
    argsort,
    swapaxes,
    swapdims,
    sequence_mask,
    repeat_elements,
    repeat_interleave,
    argwhere,
    column_stack,
    hstack,
    movedim,
    moveaxis,
    searchsorted,
    aminmax,
    sort,
    top_k,
    deepcopy
)
from .parameter_func import (
    assign_add,
    assign_sub,
    index_add,
)
from .math_func import (
    accumulate_n,
    addn,
    absolute,
    abs,
    argmin,
    angle,
    bincount,
    bucketize,
    tensor_add,
    cosine_similarity,
    cov,
    add,
    addcdiv,
    addcmul,
    neg_tensor,
    neg,
    negative,
    tensor_lt,
    less,
    lt,
    tensor_le,
    le,
    lerp,
    norm,
    vector_norm,
    matrix_norm,
    round,
    tensor_gt,
    gt,
    tensor_ge,
    ge,
    tensor_sub,
    rsqrt,
    reciprocal,
    real,
    sub,
    subtract,
    sqrt,
    square,
    tensor_mul,
    mul,
    multiply,
    digamma,
    lgamma,
    tensor_div,
    div,
    divide,
    true_divide,
    tensor_floordiv,
    floor_div,
    floor_divide,
    floordiv,
    float_power,
    fmod,
    xdivy,
    tensor_pow,
    pow,
    pows,
    tensor_mod,
    floor_mod,
    floormod,
    lcm,
    tensor_exp,
    einsum,
    view_as_real,
    var,
    var_mean,
    std_mean,
    exp,
    tensor_expm1,
    expm1,
    eq,
    equal,
    not_equal,
    ne,
    isneginf,
    isposinf,
    isfinite,
    isnan,
    isclose,
    hypot,
    heaviside,
    gcd,
    log,
    logcumsumexp,
    logdet,
    log_matrix_determinant,
    slogdet,
    matrix_determinant,
    det,
    linspace,
    lu_solve,
    matrix_solve,
    maximum,
    median,
    nan_to_num,
    nansum,
    nanmean,
    nanmedian,
    logaddexp,
    logaddexp2,
    logit,
    std,
    ldexp,
    mv,
    addbmm,
    addmv,
    addmm,
    addr,
    adjoint,
    inplace_add,
    inplace_sub,
    inplace_update,
    inv,
    inverse,
    invert,
    minimum,
    renorm,
    floor,
    logical_not,
    logical_or,
    logical_and,
    logsumexp,
    outer,
    sign,
    signbit,
    sgn,
    cos,
    t,
    tan,
    asin,
    acos,
    arccos,
    atan,
    arcsin,
    arccosh,
    arctan,
    arctan2,
    cosh,
    tanh,
    tanhshrink,
    asinh,
    arcsinh,
    acosh,
    atanh,
    arctanh,
    atan2,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    bitwise_left_shift,
    bitwise_right_shift,
    erf,
    erfc,
    cdist,
    ceil,
    positive,
    numel,
    permute,
    i0,
    bernoulli,
    bessel_i0,
    bessel_i0e,
    bessel_j0,
    bessel_j1,
    bessel_k0,
    bessel_k0e,
    bessel_y0,
    bessel_y1,
    bessel_i1,
    bessel_i1e,
    bessel_k1,
    bessel_k1e,
    stft,
    exp2,
    deg2rad,
    isreal,
    is_complex,
    rad2deg,
    truncate_div,
    truncate_mod,
    trunc,
    gumbel_softmax,
    kaiser_window,
    matmul,
    inner,
    baddbmm,
    cummin,
    cummax,
    cumsum,
    amin,
    amax,
    mean,
    prod,
    all,
    any,
    sparse_segment_mean,
    block_diag,
    atleast_1d,
    dstack,
    diff,
    atleast_2d,
    cartesian_prod,
    atleast_3d,
    vander,
    vstack,
    row_stack,
    combinations,
    dist,
    copysign,
    hann_window,
    log2,
    xlogy,
    log10,
    log1p,
    approximate_equal,
    frac,
    kron,
    rot90,
    remainder,
    iou,
    bmm,
    trapz,
    cholesky,
    cholesky_inverse,
    cholesky_solve,
    conj,
    cross,
    erfinv,
    less_equal,
    cumprod,
    greater,
    greater_equal,
    igamma,
    igammac,
    isinf,
    logical_xor,
    imag,
    roll,
    orgqr,
    ormqr,
    sum,
    matrix_power,
    matrix_exp,
    logspace,
    diag_embed,
    fmax,
    fmin,
    inplace_index_add,
    lu_unpack,
    nanquantile,
    polar,
    polygamma,
    quantile,
    tril_indices,
    triu_indices,
    nextafter,
    trace,
    zeta,
    histc,
    fft,
    fft2,
    fftn,
    ifft,
    ifft2,
    ifftn,
    count_nonzero,
    tensor_dot,
    vecdot,
    dot,
    batch_dot,
    eps,
)
from .nn_func import (
    adaptive_avg_pool1d,
    adaptive_avg_pool2d,
    adaptive_avg_pool3d,
    adaptive_max_pool1d,
    adaptive_max_pool2d,
    adaptive_max_pool3d,
    avg_pool1d,
    avg_pool2d,
    avg_pool3d,
    max_pool2d,
    max_pool3d,
    batch_norm,
    bidense,
    celu,
    bias_add,
    binary_cross_entropy,
    binary_cross_entropy_with_logits,
    cosine_embedding_loss,
    dropout1d,
    dropout2d,
    dropout3d,
    dense,
    deformable_conv2d,
    fast_gelu,
    flip,
    fliplr,
    flipud,
    fractional_max_pool2d,
    fractional_max_pool3d,
    pixel_shuffle,
    pixel_unshuffle,
    hardshrink,
    soft_shrink,
    is_floating_point,
    intopk,
    interpolate,
    upsample,
    kl_div,
    log_softmax,
    lrn,
    mish,
    margin_ranking_loss,
    max_unpool1d,
    max_unpool2d,
    max_unpool3d,
    hardswish,
    hardtanh,
    huber_loss,
    softsign,
    silu,
    selu,
    soft_margin_loss,
    softmax,
    softmin,
    softshrink,
    softplus,
    pdist,
    pad,
    prelu,
    mirror_pad,
    nll_loss,
    smooth_l1_loss,
    l1_loss,
    threshold,
    leaky_relu,
    cross_entropy,
    grid_sample,
    ctc_greedy_decoder,
    ctc_loss,
    dropout,
    conv3d_transpose,
    conv1d,
    conv2d,
    sigmoid,
    logsigmoid,
    relu,
    relu6,
    rrelu,
    conv3d,
    glu,
    multi_margin_loss,
    multilabel_margin_loss,
    multilabel_soft_margin_loss,
    elu,
    gelu,
    hinge_embedding_loss,
    gaussian_nll_loss,
    lp_pool1d,
    lp_pool2d,
    mse_loss,
    triplet_margin_loss,
    msort,
    channel_shuffle,
    hardsigmoid,
)
from .linalg_func import (
    cond,
    eig,
    eigvals,
    geqrf,
    svd,
    pinv,
    qr
)
from .sparse_func import (
    coalesce,
    coo2csr,
    coo_tensor_get_dense_shape,
    coo_tensor_get_indices,
    coo_tensor_get_values,
    csr_div,
    csr_gather,
    csr_mul,
    csr_mv,
    csr_mm,
    csr_reduce_sum,
    csr_to_coo,
    csr2coo,
    csr_tensor_get_dense_shape,
    csr_tensor_get_indices,
    csr_tensor_get_indptr,
    csr_tensor_get_values,
    dense_to_sparse_coo,
    dense_to_sparse_csr,
    make_sparse_tensor,
    make_coo_tensor,
    make_csr_tensor,
    make_row_tensor,
    make_row_tensor_inner,
    make_map_parameter,
    row_tensor_get_values,
    row_tensor_get_indices,
    row_tensor_get_dense_shape,
    row_tensor_add,
    coo_add,
    coo_concat,
    csr_add,
    csr_softmax,
    csr_to_dense,
)
from .random_func import (
    standard_laplace,
    random_categorical,
    uniform,
    standard_normal,
    random_gamma,
    uniform_candidate_sampler,
    random_poisson,
    log_uniform_candidate_sampler,
    shuffle,
    choice_with_mask,
    normal,
    laplace,
    gamma,
    poisson,
    multinomial,
    rand,
    rand_like,
    randn,
    randn_like,
    randint,
    randint_like,
    multinomial_with_replacement,
    randperm,
)
from .grad import (
    grad_func,
    grad,
    value_and_grad,
    jacfwd,
    jacrev,
    jet,
    derivative,
    jvp,
    vjp,
    linearize,
    stop_gradient,
    get_grad
)
from .debug_func import (
    print_,
)
from .image_func import (
    bounding_box_decode,
    bounding_box_encode,
    check_valid,
    crop_and_resize
)
from .spectral_func import (
    blackman_window,
    bartlett_window,
)
from .vmap_func import (
    vmap,
    _VmapGeneralPreprocess,
    _VmapGeneralRule,
)
from .sparse_unary_func import (
    csr_cos,
    csr_tan,
    csr_inv,
    csr_exp,
    csr_relu,
    csr_expm1,
    csr_isfinite,
    csr_asin,
    csr_sqrt,
    csr_log,
    csr_isnan,
    csr_acos,
    csr_floor,
    csr_atan,
    csr_square,
    csr_relu6,
    csr_sinh,
    csr_ceil,
    csr_cosh,
    csr_softsign,
    csr_log1p,
    csr_round,
    csr_tanh,
    csr_neg,
    csr_asinh,
    csr_acosh,
    csr_abs,
    csr_isinf,
    csr_atanh,
    csr_sigmoid,
    csr_sin,
    coo_cos,
    coo_tan,
    coo_inv,
    coo_exp,
    coo_relu,
    coo_expm1,
    coo_isfinite,
    coo_asin,
    coo_sqrt,
    coo_log,
    coo_isnan,
    coo_acos,
    coo_floor,
    coo_atan,
    coo_square,
    coo_relu6,
    coo_sinh,
    coo_ceil,
    coo_cosh,
    coo_softsign,
    coo_log1p,
    coo_round,
    coo_tanh,
    coo_neg,
    coo_asinh,
    coo_acosh,
    coo_abs,
    coo_isinf,
    coo_atanh,
    coo_sigmoid,
    coo_sin
)
from .clip_func import (
    clip_by_value,
    clip_by_norm,
    clamp,
    clip,
    clip_by_global_norm,
)
from .other_func import (
    depend,
    partial,
)
from ..operations.manually_defined import (rank,)
from ..auto_generate import (assign, sin, sinc, sinh)

__all__ = [
    'assign',
    'rank',
    'sin',
    'sinc',
    'sinh',
]
__all__.extend(array_func.__all__)
__all__.extend(parameter_func.__all__)
__all__.extend(math_func.__all__)
__all__.extend(nn_func.__all__)
__all__.extend(linalg_func.__all__)
__all__.extend(sparse_func.__all__)
__all__.extend(random_func.__all__)
__all__.extend(grad_func.__all__)
__all__.extend(debug_func.__all__)
__all__.extend(image_func.__all__)
__all__.extend(spectral_func.__all__)
__all__.extend(vmap_func.__all__)
__all__.extend(sparse_unary_func.__all__)
__all__.extend(clip_func.__all__)
__all__.extend(other_func.__all__)
__all__.sort()
