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
#include <IDSelector.h>

#include <io.h>
#include <io_macros.h>

namespace faiss {

BlockInvertedLists::BlockInvertedLists(
        size_t nlist,
        size_t n_per_block,
        size_t block_size)
        : InvertedLists(nlist, InvertedLists::INVALID_CODE_SIZE),
          n_per_block(n_per_block),
          block_size(block_size) {
    ids.resize(nlist);
}

BlockInvertedLists::BlockInvertedLists()
        : InvertedLists(0, InvertedLists::INVALID_CODE_SIZE) {}

size_t BlockInvertedLists::add_entries(
        size_t list_no,
        size_t n_entry,
        const idx_t* ids_in,
        const uint8_t* code) {
    if (n_entry == 0) {
        return 0;
    }
    FAISS_THROW_IF_NOT(list_no < nlist);
    size_t o = ids[list_no].size();
    ids[list_no].resize(o + n_entry);
    memcpy(&ids[list_no][o], ids_in, sizeof(ids_in[0]) * n_entry);
    size_t n_block = (o + n_entry + n_per_block - 1) / n_per_block;
    return o;
}

size_t BlockInvertedLists::list_size(size_t list_no) const {
    assert(list_no < nlist);
    return ids[list_no].size();
}

const uint8_t* BlockInvertedLists::get_codes(size_t list_no) const {
    // assert(list_no < nlist);
    // return codes[list_no].get();
    return 0;
}

size_t BlockInvertedLists::remove_ids(const IDSelector& sel) {
    idx_t nremove = 0;
#pragma omp parallel for
    for (idx_t i = 0; i < nlist; i++) {
        // std::vector<uint8_t> buffer(packer->code_size);
        idx_t l = ids[i].size(), j = 0;
        while (j < l) {
            if (sel.is_member(ids[i][j])) {
                l--;
                ids[i][j] = ids[i][l];
                // packer->unpack_1(codes[i].data(), l, buffer.data());
                // packer->pack_1(buffer.data(), j, codes[i].data());
            } else {
                j++;
            }
        }
        resize(i, l);
        nremove += ids[i].size() - l;
    }

    return nremove;
}

const idx_t* BlockInvertedLists::get_ids(size_t list_no) const {
    assert(list_no < nlist);
    return ids[list_no].data();
}

void BlockInvertedLists::resize(size_t list_no, size_t new_size) {
    ids[list_no].resize(new_size);
    size_t n_block = (new_size + n_per_block - 1) / n_per_block;
    size_t new_nbytes = n_block * block_size;
}

void BlockInvertedLists::update_entries(
        size_t,
        size_t,
        size_t,
        const idx_t*,
        const uint8_t*) {
    FAISS_THROW_MSG("not implemented");
}

BlockInvertedLists::~BlockInvertedLists() {
    // delete packer;
}

/**************************************************
 * IO hook implementation
 **************************************************/

BlockInvertedListsIOHook::BlockInvertedListsIOHook()
        : InvertedListsIOHook("ilbl", typeid(BlockInvertedLists).name()) {}

void BlockInvertedListsIOHook::write(const InvertedLists* ils_in, IOWriter* f)
        const {
    uint32_t h = fourcc("ilbl");
    WRITE1(h);
    const BlockInvertedLists* il =
            dynamic_cast<const BlockInvertedLists*>(ils_in);
    WRITE1(il->nlist);
    WRITE1(il->code_size);
    WRITE1(il->n_per_block);
    WRITE1(il->block_size);

    for (size_t i = 0; i < il->nlist; i++) {
        WRITEVECTOR(il->ids[i]);
        // WRITEVECTOR(il->codes[i]);
    }
}

InvertedLists* BlockInvertedListsIOHook::read(IOReader* f, int /* io_flags */)
        const {
    BlockInvertedLists* il = new BlockInvertedLists();
    READ1(il->nlist);
    READ1(il->code_size);
    READ1(il->n_per_block);
    READ1(il->block_size);

    il->ids.resize(il->nlist);

    for (size_t i = 0; i < il->nlist; i++) {
        READVECTOR(il->ids[i]);
    }

    return il;
}

} // namespace faiss
