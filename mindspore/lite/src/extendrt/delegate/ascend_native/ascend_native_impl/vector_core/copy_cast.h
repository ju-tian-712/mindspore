// /**
//  * Copyright 2023 Huawei Technologies Co., Ltd
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  * http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_COPY_CAST_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_COPY_CAST_H_

namespace mindspore::ascend_native {
void CopyHostFp32ToDeviceFp16(void *src, void **dst_ptr, size_t elem_num, void *stream);
void CopyHostFp32ToDeviceFp32(void *src, void **dst_ptr, size_t elem_num, void *stream);
void CopyHostFp16ToDeviceFp16(void *src, void **dst_ptr, size_t elem_num, void *stream);
void CopyHostFp16ToDeviceFp32(void *src, void **dst_ptr, size_t elem_num, void *stream);
void CopyDeviceFp16ToHostFp32(void *src, void *dst, size_t elem_num, void *stream);
void CopyDeviceFp32ToHostFp32(void *src, void *dst, size_t elem_num, void *stream);
void CopyDeviceFp16ToHostFp16(void *src, void *dst, size_t elem_num, void *stream);
void CopyDeviceFp32ToHostFp16(void *src, void *dst, size_t elem_num, void *stream);
void CopyDeviceInt32oHostInt64(void *src, void *dst, size_t elem_num, void *stream);
}  // namespace mindspore::ascend_native
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_COPY_CAST_H_
