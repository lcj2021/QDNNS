/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <InvertedLists.h>
#include <string>

namespace faiss {

/** Callbacks to handle other types of InvertedList objects.
 *
 * The callbacks should be registered with add_callback before calling
 * read_index or read_InvertedLists. The callbacks for
 * OnDiskInvertedLists are registrered by default. The invlist type is
 * identified by:
 *
 * - the key (a fourcc) at read time
 * - the class name (as given by typeid.name) at write time
 */
struct InvertedListsIOHook {
    const std::string key;       ///< string version of the fourcc
    const std::string classname; ///< typeid.name

    InvertedListsIOHook(const std::string& key, const std::string& classname);

    virtual ~InvertedListsIOHook() {}

    /**************************** Manage the set of callbacks ******/

    // transfers ownership
    static void add_callback(InvertedListsIOHook*);
    static InvertedListsIOHook* lookup_classname(const std::string& classname);
};

} // namespace faiss
