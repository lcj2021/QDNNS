/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <InvertedListsIOHook.h>

#include <FaissAssert.h>

#include <BlockInvertedLists.h>
#include <iostream>
namespace faiss {

/**********************************************************
 * InvertedListIOHook's
 **********************************************************/

InvertedListsIOHook::InvertedListsIOHook(
        const std::string& key,
        const std::string& classname)
        : key(key), classname(classname) {}

namespace {

/// std::vector that deletes its contents
struct IOHookTable : std::vector<InvertedListsIOHook*> {
    IOHookTable() {
        std::cerr << "Registering InvertedListsIOHook" << std::endl;
        push_back(new BlockInvertedListsIOHook());
    }

    ~IOHookTable() {
        for (auto x : *this) {
            delete x;
        }
    }
};

static IOHookTable InvertedListsIOHook_table;

} // namespace

} // namespace faiss
