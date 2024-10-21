/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <InvertedLists.h>
#include <InvertedListsIOHook.h>

namespace faiss {


/** Inverted Lists that are organized by blocks.
 *
 * Different from the regular inverted lists, the codes are organized by blocks
 * of size block_size bytes that reprsent a set of n_per_block. Therefore, code
 * allocations are always rounded up to block_size bytes. The codes are also
 * aligned on 32-byte boundaries for use with SIMD.
 *
 * To avoid misinterpretations, the code_size is set to (size_t)(-1), even if
 * arguably the amount of memory consumed by code is block_size / n_per_block.
 *
 * The writing functions add_entries and update_entries operate on block-aligned
 * data.
 */
struct BlockInvertedLists : InvertedLists {
    size_t n_per_block = 0; // nb of vectors stored per block
    size_t block_size = 0;  // nb bytes per block

    std::vector<std::vector<idx_t>> ids;

    BlockInvertedLists(size_t nlist, size_t vec_per_block, size_t block_size);

    BlockInvertedLists();

    ~BlockInvertedLists() override;
};

struct BlockInvertedListsIOHook : InvertedListsIOHook {
    BlockInvertedListsIOHook();
};

} // namespace faiss
