# Copyright 2021 Huawei Technologies Co., Ltd
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
# ==============================================================================
import numpy as np
import pytest
from mindspore import nn, context, Tensor
from mindspore import ops as P
from mindspore.train import DatasetHelper, connect_network_with_dataset
import mindspore.dataset as ds
import mindspore as ms
from mindspore.common.initializer import One
from tests.mark_utils import arg_mark


def _exec_preprocess(network, is_train, dataset, dataset_sink_mode, sink_size=1, epoch_num=1, dataset_helper=None):

    if dataset_helper is None:
        dataset_helper = DatasetHelper(
            dataset, dataset_sink_mode, sink_size, epoch_num)

    if dataset_sink_mode:
        network = connect_network_with_dataset(network, dataset_helper)

    network.set_train(is_train)

    return dataset_helper, network


def _eval_dataset_sink_process(network, valid_dataset):
    dataset_helper, eval_network = _exec_preprocess(network, is_train=False, dataset=valid_dataset,
                                                    dataset_sink_mode=True)
    for inputs1, inputs2 in zip(dataset_helper, valid_dataset.create_dict_iterator()):
        outputs = eval_network(*inputs1)
        for elem1, (_, elem2) in zip(outputs, inputs2.items()):
            assert elem1.shape == elem2.shape


def dataset_generator():
    for i in range(1, 10):
        yield (
            np.ones((32, i), dtype=np.float32), np.zeros(
                (32, i, i, 3), dtype=np.int32),
            np.ones((32,), dtype=np.float32),
            np.ones((32, i, 8), dtype=np.float32), np.ones((32, 8, 8), dtype=np.float32))


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = P.ReLU()

    def construct(self, x1, x2, x3, x4, x5):
        x1 = self.relu(x1)
        x1 = self.relu(x1)

        x2 = self.relu(x2)

        x3 = self.relu(x3)
        x3 = self.relu(x3)

        x4 = self.relu(x4)

        x5 = self.relu(x5)
        return x1, x2, x3, x4, x5


def test_getnext_dynamic_pipeline_ascend():
    """
    Feature: sink one step of dynamic data sink.
    Description: datasets with dynamic shape as input.
    Expectation: success without assert exception.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    network = Net()
    dataset = ds.GeneratorDataset(
        dataset_generator, ["data1", "data2", "data3", "data4", "data5"])

    t0 = Tensor(dtype=ms.float32, shape=[32, None])
    t1 = Tensor(dtype=ms.int32, shape=[32, None, None, 3])
    t2 = Tensor(dtype=ms.float32, shape=[32], init=One())
    t3 = Tensor(dtype=ms.float32, shape=[32, None, 8])
    t4 = Tensor(dtype=ms.float32, shape=[32, 8, 8], init=One())
    network.set_inputs(t0, t1, t2, t3, t4)
    _eval_dataset_sink_process(network, dataset)


def test_getnext_sink_size_dynamic_pipeline():
    """
    Feature: arbitrary sink size of dynamic data sink.
    Description: datasets with dynamic shape as input.
    Expectation: success without assert exception.
    """
    network = Net()
    dataset = ds.GeneratorDataset(
        dataset_generator, ["data1", "data2", "data3", "data4", "data5"])

    t0 = Tensor(dtype=ms.float32, shape=[32, None])
    t1 = Tensor(dtype=ms.int32, shape=[32, None, None, 3])
    t2 = Tensor(dtype=ms.float32, shape=[32], init=One())
    t3 = Tensor(dtype=ms.float32, shape=[32, None, 8])
    t4 = Tensor(dtype=ms.float32, shape=[32, 8, 8], init=One())
    network.set_inputs(t0, t1, t2, t3, t4)

    dataset_helper, eval_network = _exec_preprocess(
        network, is_train=False, dataset=dataset, dataset_sink_mode=True, sink_size=-1)
    for inputs in dataset_helper:
        outputs = eval_network(*inputs)
        for data_item in dataset.create_dict_iterator():
            last_inputs = data_item.items()
        for output, (_, last_input) in zip(outputs, last_inputs):
            assert output.shape == last_input.shape

@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_getnext_sink_size_dynamic_pipeline_ascend():
    """
    Feature: arbitrary sink size of dynamic data sink.
    Description: datasets with dynamic shape as input.
    Expectation: success without assert exception.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_getnext_sink_size_dynamic_pipeline()

@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_getnext_sink_size_dynamic_pipeline_gpu():
    """
    Feature: arbitrary sink size of dynamic data sink.
    Description: datasets with dynamic shape as input.
    Expectation: success without assert exception.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_getnext_sink_size_dynamic_pipeline()
    