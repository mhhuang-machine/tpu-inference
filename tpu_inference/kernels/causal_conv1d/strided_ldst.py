# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

# NOTE: When performing strided ldst, using pltpu.bitcast on vreg data should
# be avoided as it can trigger unintended relayout.


def load_large_to_compact(vmem_ref,
                          dst_dtype: jnp.dtype | None = None) -> jax.Array:
    assert vmem_ref.ndim == 2

    row_size = vmem_ref.shape[0]
    src_dtype = vmem_ref.dtype
    should_unpack = dst_dtype is not None and dst_dtype != src_dtype
    packing = 4 // src_dtype.itemsize

    unpacked_list = []
    for row_start in range(0, row_size, packing):
        row_end = row_start + packing
        if should_unpack:
            packed_row = row_start // packing
            u32_vmem_ref = vmem_ref.bitcast(jnp.uint32)
            packed = u32_vmem_ref[packed_row:packed_row + 1]

            for p in range(packing):
                unpacked = pltpu.unpack_elementwise(
                    packed,
                    index=p,
                    packed_dtype=src_dtype,
                    unpacked_dtype=dst_dtype,
                )
                unpacked_list.append(unpacked)
        else:
            unpacked_list.append(vmem_ref[row_start:row_end])

    return jnp.stack(unpacked_list, axis=0)


def store_compact_to_large(vmem_ref, vreg: jax.Array):
    dst_row_size = vmem_ref.shape[0]
    src_row_size = vreg.shape[0]

    src_dtype = vreg.dtype
    dst_dtype = vmem_ref.dtype
    should_pack = src_dtype != dst_dtype
    src_packing = 4 // src_dtype.itemsize
    dst_packing = 4 // dst_dtype.itemsize

    assert vreg.ndim == 3
    assert vreg.shape[-2] == src_packing
    assert vmem_ref.ndim == 2
    assert vmem_ref.shape[-1] == vreg.shape[-1]
    assert src_row_size == dst_row_size

    for row_start in range(0, dst_row_size, dst_packing):
        row_end = row_start + dst_packing
        packed_row = row_start // dst_packing
        if should_pack:
            assert src_dtype.itemsize == 4
            assert dst_dtype.itemsize == 2

            unpacked_list = [vreg[i] for i in range(row_start, row_end)]
            packed = pltpu.pack_elementwise(unpacked_list,
                                            packed_dtype=dst_dtype)
            u32_vmem_ref = vmem_ref.bitcast(jnp.uint32)
            u32_vmem_ref[packed_row:packed_row + 1] = packed
        else:
            vmem_ref[row_start:row_end] = vreg[packed_row]


def load_large_to_compact_for_conv_state(
    vmem_ref,  # (kernel_size - 1, packing, packed_dim_size)
    dst_dtype: jnp.dtype | None = None,
) -> jax.Array:  # (kernel_size - 1, 1, packing * packed_dim_size)
    assert vmem_ref.ndim == 3

    src_row_size = vmem_ref.shape[0]
    src_dtype = vmem_ref.dtype
    should_unpack = dst_dtype is not None and dst_dtype != src_dtype
    packing = 4 // src_dtype.itemsize

    unpacked_list = []
    for row_idx in range(src_row_size):
        if should_unpack:
            u32_vmem_ref = vmem_ref.bitcast(jnp.uint32)
            concat_list = []
            packed = u32_vmem_ref[row_idx]
            for p in range(packing):
                unpacked = pltpu.unpack_elementwise(
                    packed, index=p, packed_dtype=src_dtype, unpacked_dtype=dst_dtype
                )
                # Flattening to 1D to hint mosaic to use [1, 128] tiling for concat.
                concat_list.append(unpacked.reshape(-1))
            result = jnp.concat(concat_list, axis=0)
            unpacked_list.append(result.reshape(1, -1))
        else:
            unpacked_list.append(vmem_ref[row_idx])

    return jnp.stack(unpacked_list, axis=0)


def store_compact_to_large_for_conv_state(
    vmem_ref,  # (kernel_size - 1, packing, packed_dim_size)
    vreg: jax.Array,  # (kernel_size - 1, 1, packing * packed_dim_size)
):
    dst_row_size = vmem_ref.shape[0]
    src_row_size = vreg.shape[0]
    assert src_row_size == dst_row_size

    src_dtype = vreg.dtype
    dst_dtype = vmem_ref.dtype
    should_pack = src_dtype != dst_dtype
    src_packing = 4 // src_dtype.itemsize
    dst_packing = 4 // dst_dtype.itemsize

    assert vreg.ndim == 3
    assert vreg.shape[-2] == src_packing
    assert vmem_ref.ndim == 3
    assert vmem_ref.shape[-2] == dst_packing

    u32_vmem_ref = vmem_ref.bitcast(jnp.uint32)
    for row_idx in range(dst_row_size):
        if should_pack:
            assert src_dtype.itemsize == 4
            assert dst_dtype.itemsize == 2
            unpacked_list = jnp.split(vreg[row_idx], dst_packing, axis=-1)
            packed = pltpu.pack_elementwise(unpacked_list, packed_dtype=dst_dtype)
            u32_vmem_ref[row_idx] = packed
        else:
            vmem_ref[row_idx] = vreg[row_idx]
