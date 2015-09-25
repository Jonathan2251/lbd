//===-- Cpu0.h - Top-level interface for Cpu0 representation ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in
// the LLVM Cpu0 back-end.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_CPU0_H
#define TARGET_CPU0_H

#include "MCTargetDesc/Cpu0MCTargetDesc.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class Cpu0TargetMachine;
  class FunctionPass;

  FunctionPass *createCpu0ReplaceSelectPass(Cpu0TargetMachine &TM);
  FunctionPass *createCpu0ISelDag(Cpu0TargetMachine &TM);
  FunctionPass *createCpu0EmitGPRestorePass(Cpu0TargetMachine &TM);
  FunctionPass *createCpu0DelJmpPass(Cpu0TargetMachine &TM);

} // end namespace llvm;

#endif
