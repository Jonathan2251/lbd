//===- lib/ReaderWriter/ELF/Cpu0/Cpu0RelocationPass.h -----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Declares the relocation processing pass for cpu0. This includes
///   GOT and PLT entries, TLS, COPY, and ifunc.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_CPU0_RELOCATION_PASS_H
#define LLD_READER_WRITER_ELF_CPU0_RELOCATION_PASS_H

#include <memory>

namespace lld {
class Pass;
namespace elf {
class Cpu0LinkingContext;

/// \brief Create cpu0 relocation pass for the given linking context.
std::unique_ptr<Pass>
createCpu0RelocationPass(const Cpu0LinkingContext &);
}
}

#endif
