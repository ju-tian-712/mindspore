# Copyright 2024 Huawei Technologies Co., Ltd
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
import os
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='allcards',
          essential_mark='essential')
def test_pynative_hccl_allreduce_8p():
    '''
    Feature: run allreduce op in pynative mode using msrun.
    Description: Test case entry allreduce op in pynative mode.
    Expectation: Run success.
    '''
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10969 --join=True "\
        "pytest -s test_pynative_hccl_allreduce.py::test_msrun_pynative_hccl_allreduce_8p"
    )
    assert return_code == 0
