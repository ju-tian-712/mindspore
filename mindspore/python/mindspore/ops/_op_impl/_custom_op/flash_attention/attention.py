# Copyright 2023 Huawei Technologies Co., Ltd
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
"""the base class of flash attention"""
from abc import ABCMeta
from abc import abstractmethod
from functools import partial

import te.platform as tbe_platform
from tbe import tik
from tbe.common.platform import get_soc_spec
from tbe.common.platform import set_current_compile_soc_info

from mindspore.ops._op_impl._custom_op.flash_attention.constants import FP16
from mindspore.ops._op_impl._custom_op.flash_attention.constants import GM
from mindspore.ops._op_impl._custom_op.flash_attention.constants import INT8
from mindspore.ops._op_impl._custom_op.flash_attention.constants import MASK_FILL_VALUE
from mindspore.ops._op_impl._custom_op.flash_attention.constants import UB
from mindspore.ops._op_impl._custom_op.flash_attention.tik_ops_utils import TikOpsUtils
from mindspore.ops._op_impl._custom_op.flash_attention.tiling_strategy.strategy import TilingPara
from mindspore.ops._op_impl._custom_op.flash_attention.tiling_strategy.strategy import TilingStrategy
from mindspore.ops._op_impl._custom_op.flash_attention.tiling_strategy.xunfei_tiling import XunfeiTiling

set_current_compile_soc_info("Ascend910")  # 默认310


class FlashAttention(metaclass=ABCMeta):
    """The base class of FlashAttention"""

    def __init__(self, q, k, v, dim_mask, attn_mask, dropout_mask, alibi_mask, kernel_name,
                 tiling_stgy_cls,
                 prev_block_num=65536,
                 next_block_num=65536,
                 disable_debug=True):
        """
        Init parameter shape
        :param q: with shape: (B, h, N, d)
        :param k: with shape: (B, h, N, d)
        :param v: with shape: (B, h, N, d)
        :param dim_mask: with shape: (x, ) x equals length last dim that not padded to 16.
        :param attn_mask: with shape: (1, N, N) or (B, N, N)
        :param dropout_mask: with shape: (B, h, N, N)
        :param alibi_mask: with shape: (B, h, 1, N)
        :param kernel_name:
        :param tiling_stgy_cls:
        :param prev_block_num:
        :param next_block_num:
        :param disable_debug:
        """
        self.tik_instance = tik.Tik(disable_debug=disable_debug)
        self.core_num = get_soc_spec(tbe_platform.CORE_NUM)
        self.M = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)
        self.kernel_name = kernel_name
        self.cont_data_mv_1_bust = partial(self.tik_instance.data_move, sid=0, nburst=1,
                                           src_stride=0,
                                           dst_stride=0)
        self.tik_ops_utils = TikOpsUtils(self.tik_instance)
        self.parser_input_shape(alibi_mask, attn_mask, dim_mask, dropout_mask, k, q, v)

        batch_size, h, Nq, d = self.q_shape
        self.head_num = h
        self.B, self.Nq, self.d = batch_size * h, Nq, d
        self.N = self.k_shape[2]

        self.l_shape = [batch_size, h, self.Nq]
        self.m_shape = [batch_size, h, self.Nq]
        self.O_shape = [batch_size, h, self.Nq, self.d]
        self.actual_d = self.dim_mask_shape[0]

        self.K0 = 16
        self.prev_block_num = prev_block_num
        self.next_block_num = next_block_num
        if tiling_stgy_cls is None:
            self.tiling_stgy = XunfeiTiling(self.Nq, self.N, self.d)
        else:
            self.tiling_stgy: TilingStrategy = tiling_stgy_cls(self.Nq, self.N, self.d)
        self.Br = None
        self.last_Br = None
        self.Bc = None
        self.last_Bc = None
        self.Tr = None
        self.Tc = None
        self.Q_gm = None
        self.K_gm = None
        self.V_gm = None
        self.dim_mask_gm = None
        self.att_mask_gm = None
        self.drop_mask_gm = None
        self.alibi_mask_gm = None

    @staticmethod
    def get_gm_offset(batch_start, batch_idx, h, w, block_h, block_idx):
        gm_offset = (batch_start + batch_idx) * h * w + block_idx * block_h * w
        return gm_offset

    @staticmethod
    def get_alibi_gm_offset(batch_start, batch_idx, w, block_w, block_idx):
        gm_offset = (batch_start + batch_idx) * w + block_idx * block_w
        return gm_offset

    @staticmethod
    def get_drop_mask_gm_offset(batch_start, batch_idx, h, w, block_h, block_h_idx, block_w, block_w_idx):
        gm_offset = (batch_start + batch_idx) * h * w + block_h_idx * (w * block_h) + block_w_idx * block_w
        return gm_offset

    @abstractmethod
    def define_custom_inputs(self):
        raise NotImplementedError

    @abstractmethod
    def define_outputs(self):
        raise NotImplementedError

    @abstractmethod
    def collect_inputs(self):
        raise NotImplementedError

    @abstractmethod
    def collect_outputs(self):
        raise NotImplementedError

    def get_attn_mask_gm_offset(self, batch_start, batch_idx, h, w, block_h, block_h_idx, block_w, block_w_idx):
        if self.att_mask_shape[0] == 1:
            gm_offset = block_h_idx * (w * block_h) + block_w_idx * block_w
        else:
            gm_offset = ((batch_start + batch_idx) // self.head_num) * h * w \
                        + block_h_idx * (w * block_h) + block_w_idx * block_w
        return gm_offset

    def parser_input_shape(self, alibi_mask, attn_mask, dim_mask, dropout_mask, k, q, v):
        """parser input shape"""
        self.has_attn_mask = False
        self.has_drop_mask = False
        self.has_alibi_mask = False
        if isinstance(q, dict):
            self.q_shape = q["shape"]
            self.k_shape = k["shape"]
            self.v_shape = v["shape"]
            self.dim_mask_shape = dim_mask["shape"]
            if attn_mask is not None:
                self.has_attn_mask = True
                self.att_mask_shape = attn_mask["shape"]
            if dropout_mask is not None:
                self.has_drop_mask = True
                self.drop_mask_shape = dropout_mask["shape"]
            if alibi_mask is not None:
                self.has_alibi_mask = True
                self.alibi_mask_shape = alibi_mask["shape"]
        else:
            self.q_shape = q.shape
            self.k_shape = k.shape
            self.v_shape = v.shape
            self.dim_mask_shape = dim_mask.shape
            if attn_mask is not None:
                self.has_attn_mask = True
                self.att_mask_shape = attn_mask.shape
            if dropout_mask is not None:
                self.has_drop_mask = True
                self.drop_mask_shape = dropout_mask.shape
            if alibi_mask is not None:
                self.has_alibi_mask = True
                self.alibi_mask_shape = alibi_mask.shape

    def define_inputs_outputs(self):
        self.define_common_inputs()

        self.define_custom_inputs()

        self.define_outputs()

    def init(self):
        """init parameters"""
        tiling_para: TilingPara = self.tiling_stgy.tiling()

        self.Br = tiling_para.Br
        self.last_Br = tiling_para.last_Br
        self.Bc = tiling_para.Bc
        self.last_Bc = tiling_para.last_Bc
        self.Tr = tiling_para.Tr
        self.Tc = tiling_para.Tc

        self.define_inputs_outputs()

    def define_common_inputs(self):
        """define common input gm tensors"""
        self.Q_gm = self.tik_instance.Tensor(FP16, self.q_shape, name="Q_gm", scope=GM)
        self.K_gm = self.tik_instance.Tensor(FP16, self.k_shape, name="K_gm", scope=GM)
        self.V_gm = self.tik_instance.Tensor(FP16, self.v_shape, name="V_gm", scope=GM)
        self.dim_mask_gm = self.tik_instance.Tensor(INT8, self.dim_mask_shape, name="mask_gm",
                                                    scope=GM)
        if self.has_attn_mask:
            self.att_mask_gm = self.tik_instance.Tensor(FP16, self.att_mask_shape,
                                                        name="att_mask_gm", scope=GM)
        if self.has_drop_mask:
            self.drop_mask_gm = self.tik_instance.Tensor(FP16, self.drop_mask_shape,
                                                         name="drop_mask_gm", scope=GM)
        if self.has_alibi_mask:
            self.alibi_mask_gm = self.tik_instance.Tensor(FP16, self.alibi_mask_shape,
                                                          name="alibi_mask_gm", scope=GM)

    def do_alibi_mask(self, Sij_ub, alibi_mask_gm_offset, m_aligned, n_aligned):
        """load alibi mask from gm to ub, then add Sij"""
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            alibi_mask_ub = self.tik_instance.Tensor(FP16, (1, n_aligned),
                                                     scope=UB, name="alibi_mask_ub")
            self.tik_instance.data_move(alibi_mask_ub, self.alibi_mask_gm[alibi_mask_gm_offset], 0, 1,
                                        n_aligned // 16, 0, 0)
            alibi_mask_ub_broadcast = self.tik_ops_utils.broadcast_row(alibi_mask_ub, (m_aligned, n_aligned))
            self.tik_instance.h_add(Sij_ub, Sij_ub, alibi_mask_ub_broadcast)

    def do_att_mask(self, Sij_ub, attn_mask_gm_offset, q_blk_height, kv_blk_height,
                    q_blk_h_aligned, kv_blk_h_aligned):
        """load attn mask from gm to ub, then mul it by MASK_FILL_VALUE and add Sij"""
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            att_mask_ub = self.tik_instance.Tensor(FP16, (q_blk_h_aligned, kv_blk_h_aligned),
                                                   scope=UB, name="att_mask_ub")
            self.tik_instance.data_move(att_mask_ub, self.att_mask_gm[attn_mask_gm_offset], 0,
                                        q_blk_height, kv_blk_height // 16, (self.N - kv_blk_height) // 16, 0)
            self.tik_instance.h_mul(att_mask_ub, att_mask_ub, MASK_FILL_VALUE)
            self.tik_instance.h_add(Sij_ub, Sij_ub, att_mask_ub)

    def do_dropout_mask(self, Pij_ub, dropout_mask_gm_offset, kv_blk_h_aligned, kv_blk_height,
                        q_blk_h_aligned, q_blk_height):
        """load drop mask from gm to ub, then mul it by Pij"""
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            dropout_mask_ub = self.tik_instance.Tensor(FP16, (q_blk_h_aligned, kv_blk_h_aligned),
                                                       scope=UB, name="drop_mask_ub")
            self.tik_instance.data_move(dropout_mask_ub, self.drop_mask_gm[dropout_mask_gm_offset], 0,
                                        q_blk_height, kv_blk_height // 16, (self.N - kv_blk_height) // 16, 0)
            self.tik_instance.h_mul(Pij_ub, Pij_ub, dropout_mask_ub)
