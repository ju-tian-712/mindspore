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
"""Hal stream class"""
from mindspore._c_expression import Stream as Stream_
from mindspore._c_expression import set_cur_stream as set_cur_stream_
from mindspore._c_expression import synchronize as synchronize_
from mindspore._c_expression import current_stream as current_stream_
from mindspore._c_expression import default_stream as default_stream_
from .event import Event


class Stream(Stream_):
    r"""
    Wrapper around a device stream.

    A device stream is a linear sequence of execution that belongs to a specific device,
    independent from other streams.

    Args:
        priority (int, optional) – priority of the stream, lower numbers
        represent higher priorities. By default, streams have priority 0.

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> s1 = ms.hal.Stream()
        >>> s2 = ms.hal.Stream()
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([2, 2]), ms.float32)
        >>> with ms.hal.StreamCtx(s1):
        >>>     c = ops.matmul(a, b)
        >>> with ms.hal.StreamCtx(s2):
        >>>     s2.wait_stream(s1)
        >>>     d = ops.matmul(c, b)
        >>> ms.hal.synchronize()
        >>> print(d)
        [[4. 4.]]
    """
    def __init__(self, priority=0, **kwargs):
        if 'stream_id' in kwargs:
            super().__init__(priority, kwargs['stream_id'])
        else:
            super().__init__(priority)

    def record_event(self, event=None):
        r"""
        Records an event.

        Inputs:
            event (Event, optional): event to record. If not given, a new one
                will be allocated.

        Returns:
            Recorded event.

        Examples:
            >>> import mindspore as ms
            >>> import numpy as np
            >>> from mindspore import Tensor, ops
            >>> a = Tensor(np.ones([3, 3]), ms.float32)
            >>> b = Tensor(np.ones([3, 3]), ms.float32)
            >>> s1 = ms.hal.Stream()
            >>> with ms.hal.StreamCtx(s1):
            >>>     c = a + b
            >>>     event = s1.record_event()
            >>>     d = a * b
            >>> cur_stream = ms.hal.current_stream()
            >>> cur_stream.wait_event(event)
            >>> e = c + 3
            >>> print(e)
            [[5. 5. 5.]
            [5. 5. 5.]
            [5. 5. 5.]]
        """
        if event is None:
            event = Event()
        if not isinstance(event, Event):
            raise TypeError(f"For 'record_event', the argument 'event' should be Event,"
                            f" but got {type(event)}.")
        event.record(self)
        return event

    def wait_event(self, event):
        r"""
        Makes all future work submitted to the stream wait for an event.

        Inputs:
            event (Event): an event to wait for.
        """
        if not isinstance(event, Event):
            raise TypeError(f"For 'wait_event', the argument 'event' should be Event,"
                            f" but got {type(event)}.")
        event.wait(self)

    def wait_stream(self, stream):
        r"""
        Synchronizes with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Inputs:
            stream (Stream): a stream to synchronize.
        """
        if not isinstance(stream, Stream):
            raise TypeError(f"For 'wait_stream', the argument 'stream' should be Stream,"
                            f" but got {type(stream)}.")
        self.wait_event(stream.record_event())

    def synchronize(self):
        r"""
        Wait for all the kernels in this stream to complete.
        """
        # pylint: disable=useless-super-delegation
        super().synchronize()

    def query(self):
        r"""
        Checks if all the work submitted has been completed.

        Outputs:
            A boolean indicating if all kernels in this stream are completed.

        Examples:
            >>> import mindspore as ms
            >>> import numpy as np
            >>> from mindspore import Tensor, ops
            >>> a = Tensor(np.ones([1024, 2048]), ms.float32)
            >>> b = Tensor(np.ones([2048, 4096]), ms.float32)
            >>> s1 = ms.hal.Stream()
            >>> with ms.hal.StreamCtx(s1):
            >>>     ops.matmul(a, b)
            >>>     assert not s1.query()
            >>> s1.synchronize()
            >>> assert s1.query()
        """
        # pylint: disable=useless-super-delegation
        return super().query()

    def __eq__(self, o):
        if not isinstance(o, Stream):
            raise TypeError(f"For '__eq__', the argument 'other' should be Stream,"
                            f" but got {type(o)}.")
        return super().__eq__(o)

    def __hash__(self):
        return hash((self.id, self.device_id))


def synchronize():
    r"""
    Synchronize all streams on current device.(Each MindSpore process only occupies one device)
    """
    synchronize_()

def set_cur_stream(stream):
    r"""
    Sets the current stream.This is a wrapper API to set the stream.
    Usage of this function is discouraged in favor of the ``stream`` context manager.
    Inputs:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.

    Examples:
        >>> import mindspore as ms
        >>> cur_stream = ms.hal.current_stream()
        >>> assert cur_stream == ms.hal.default_stream()
        >>> s1 = ms.hal.Stream()
        >>> ms.hal.set_cur_stream(s1)
        >>> assert ms.hal.current_stream() == s1
    """
    if stream is None:
        return
    if not isinstance(stream, Stream):
        raise TypeError(f"For 'set_cur_stream', the argument 'stream' should be Stream,"
                        f" but got {type(stream)}.")
    set_cur_stream_(stream)

def current_stream():
    r"""
    Return current stream used on this device.

    Outputs:
        stream (Stream): current stream.
    """
    return Stream(current_stream_())

def default_stream():
    r"""
    Return default stream on this device.

    Outputs:
        stream (Stream): default stream.
    """
    return Stream(default_stream_())

class StreamCtx():
    r"""
    Context-manager that selects a given stream.

    All kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        ctx_stream (Stream): selected stream. This manager is a no-op if it's ``None``.
    """
    def __init__(self, ctx_stream):
        if ctx_stream is not None and not isinstance(ctx_stream, Stream):
            raise TypeError(f"For 'StreamCtx', the argument 'ctx_stream' should be Stream,"
                            f" but got {type(ctx_stream)}.")
        self.stream = ctx_stream
        self.prev_stream = None

    def __enter__(self):
        if self.stream is None:
            return
        self.prev_stream = current_stream()
        set_cur_stream(self.stream)
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stream is None:
            return
        set_cur_stream(self.prev_stream)
        return