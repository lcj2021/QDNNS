/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <BlockInvertedLists.h>
#include <cstring>
#include <cassert>
#include <FaissAssert.h>
#include <iostream>

namespace faiss {

BlockInvertedLists::BlockInvertedLists(
        size_t nlist,
        size_t n_per_block,
        size_t block_size)
        : InvertedLists(nlist, InvertedLists::INVALID_CODE_SIZE),
          n_per_block(n_per_block),
          block_size(block_size) {
}

BlockInvertedLists::BlockInvertedLists()
        : InvertedLists(0, InvertedLists::INVALID_CODE_SIZE) {}

BlockInvertedLists::~BlockInvertedLists() {
}

/**************************************************
 * IO hook implementation
 **************************************************/

BlockInvertedListsIOHook::BlockInvertedListsIOHook()
        : InvertedListsIOHook("ilbl", typeid(BlockInvertedLists).name()) {
            std::cerr << "BlockInvertedLists: " << typeid(BlockInvertedLists).name() << std::endl;
        }

} // namespace faiss
