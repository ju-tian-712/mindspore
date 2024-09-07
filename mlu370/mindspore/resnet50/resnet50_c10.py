# Copyright 2020 Huawei Technologies Co., Ltd
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
"""train resnet."""
import os
import argparse


from mindspore import context
from mindspore.common import set_seed
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net


parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--net', type=str, default='resnet50',
                    help='Resnet Model, either resnet50 or resnet101. Default: resnet50')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='Dataset, either cifar10 or imagenet2012. Default: imagenet2012')
parser.add_argument('--checkpoint_path', type=str, default='./resnet_1-90_937.ckpt', help='Checkpoint file path')
parser.add_argument('--dataset_path',  type=str, default="./cifar-10-batches-bin/", help='Dataset path')
parser.add_argument('--device_target', type=str, default='CPU', help='Device target. Default: CPU')
args_opt = parser.parse_args()

set_seed(1)

if args_opt.net == "resnet50":
    from src.resnet import resnet50 as resnet
    if args_opt.dataset == "cifar10":
        from src.config import config1 as config
        from src.dataset import create_dataset1 as create_dataset
    else:
       print("data arg must be cifar10")
       exit() 
else:
    print("net arg must be resnet50")
    exit()

if __name__ == '__main__':
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#              record_shapes=True,profile_memory=True,
#              on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/gpu')) as prof:
    target = args_opt.device_target

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False,enable_profiling=True)
    # if target != "GPU":
    #     device_id = int(os.getenv('DEVICE_ID'))
    #     context.set_context(device_id=device_id)
    
    # create dataset
    dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=False, batch_size=config.batch_size,
                             target=target)
    step_size = dataset.get_dataset_size()

    # define net
    net = resnet(class_num=config.class_num)

    # load checkpoint
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss, model
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # define model
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})
    # eval model
    res = model.eval(dataset)
    print("result:", res, "ckpt=", args_opt.checkpoint_path)
    # prof.step()

