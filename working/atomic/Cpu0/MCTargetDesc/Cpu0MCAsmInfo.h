//===-- Cpu0MCAsmInfo.h - Cpu0 Asm Info ------------------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the Cpu0MCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef CPU0TARGETASMINFO_H
#define CPU0TARGETASMINFO_H

#include "Cpu0Config.h"
#if CH >= CH3_2

#include "llvm/MC/MCAsmInfo.h"

namespace llvm {
  class StringRef;
  class Target;

  class Cpu0MCAsmInfo : public MCAsmInfo {
    virtual void anchor();
  public:
    explicit Cpu0MCAsmInfo(StringRef TT);
  };

} // namespace llvm

#endif // #if CH >= CH3_2

#endif
