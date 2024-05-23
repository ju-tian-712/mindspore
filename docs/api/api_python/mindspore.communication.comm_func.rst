mindspore.communication.comm_func
=================================
集合通信函数式接口。

注意，集合通信函数式接口需要先配置好通信环境变量。

针对Ascend/GPU/CPU设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/msrun_launcher.html>`_ 。

.. py:function:: mindspore.communication.comm_func.all_reduce(tensor, op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)

    使用指定方式对通信组内的所有设备的Tensor数据进行规约操作，所有设备都得到相同的结果，返回规约操作后的张量。

    .. note::
        集合中的所有进程的Tensor必须具有相同的shape和格式。

    参数：
        - **tensor** (Tensor) - 输入待规约操作的Tensor，Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **op** (str，可选) - 规约的具体操作。如 ``"sum"`` 、 ``"prod"`` 、 ``"max"`` 、和 ``"min"`` 。默认值： ``ReduceOp.SUM`` 。
        - **group** (str，可选) - 工作的通信组。默认值：``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"`` ，GPU平台为 ``"nccl_world_group"`` ）。

    返回：
        Tensor，shape与输入相同，即 :math:`(x_1, x_2, ..., x_R)` 。其内容取决于操作。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor，`op` 或 `group` 不是str。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: ops/mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。


.. py:function:: mindspore.communication.comm_func.all_gather_into_tensor(tensor, group=GlobalComm.WORLD_COMM_GROUP)

    汇聚指定的通信组中的Tensor，并返回汇聚后的张量。

    .. note::
        - 集合中所有进程的Tensor必须具有相同的shape和格式。

    参数：
        - **tensor** (Tensor) - 输入待汇聚操作的Tensor，Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **group** (str) - 工作的通信组，默认值： ``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"`` ，GPU平台为 ``"nccl_world_group"`` ）。

    返回：
        Tensor，如果组中的device数量为N，则输出的shape为 :math:`(N, x_1, x_2, ..., x_R)` 。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor，`group` 不是str。
        - **ValueError** - 调用进程的rank id大于本通信组的rank大小。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: ops/mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。


.. py:function:: mindspore.communication.comm_func.reduce_scatter_tensor(tensor, op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)

    规约并且分发指定通信组中的张量，返回分发后的张量。

    .. note::
        在集合的所有过程中，Tensor必须具有相同的shape和格式。

    参数：
        - **tensor** (Tensor) - 输入待规约且分发的Tensor，假设其形状为 :math:`(N, *)` ，其中 `*` 为任意数量的额外维度。N必须能够被rank_size整除，rank_size为当前通讯组里面的计算卡数量。
        - **op** (str, 可选) - 规约的具体操作。如 ``"sum"`` 、 ``"prod"`` 、 ``"max"`` 、和 ``"min"`` 。默认值： ``ReduceOp.SUM`` 。
        - **group** (str，可选) - 工作的通信组，默认值： ``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"`` ，GPU平台为 ``"nccl_world_group"`` ）。

    返回：
        Tensor，数据类型与 `input_x` 一致，shape为 :math:`(N/rank\_size, *)` 。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor，`op` 和 `group` 不是字符串。
        - **ValueError** - 如果输入的第一个维度不能被rank size整除。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: ops/mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。


.. py:function:: mindspore.communication.comm_func.reduce(tensor, dst, op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)

    规约指定通信组中的张量，并将规约结果发送到目标为dst的进程(全局的进程编号)中，返回发送到目标进程的张量。

    .. note::
        只有目标为dst的进程(全局的进程编号)才会收到规约操作后的输出。
        当前支持pynative模式，不支持graph模式。
        其他进程只得到一个形状为[1]的张量，且该张量没有数学意义。

    参数：
        - **tensor** (Tensor) - 输入待规约的Tensor，Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **dst** (int) - 指定接收输出的目标进程编号，只有该进程会接收规约操作后的输出结果。
        - **op** (str, 可选) - 规约的具体操作。如 ``"sum"`` 、 ``"prod"`` 、 ``"max"`` 、和 ``"min"`` 。默认值： ``ReduceOp.SUM`` 。
        - **group** (str，可选) - 工作的通信组，默认值： ``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"`` ，GPU平台为 ``"nccl_world_group"`` ）。

    返回：
        Tensor，数据类型与输入的 `tensor` 一致，shape为 :math:`(x_1, x_2, ..., x_R)`。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor，`op` 和 `group` 不是字符串。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: ops/mindspore.ops.comm_note.rst

        该样例需要在4卡环境下运行。

