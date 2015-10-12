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

#ifndef LLVM_LIB_TARGET_CPU0_CPU0_H
#define LLVM_LIB_TARGET_CPU0_CPU0_H

#include "Cpu0Config.h"
#include "MCTargetDesc/Cpu0MCTargetDesc.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class Cpu0TargetMachine;
  class FunctionPass;

#if CH >= CH9_3
#ifdef ENABLE_GPRESTORE
  FunctionPass *createCpu0EmitGPRestorePass(Cpu0TargetMachine &TM);
#endif
#endif //#if CH >= CH9_3
#if CH >= CH8_2 //1
  FunctionPass *createCpu0DelaySlotFillerPass(Cpu0TargetMachine &TM);
#endif
#if CH >= CH8_2 //2
  FunctionPass *createCpu0DelJmpPass(Cpu0TargetMachine &TM);
#endif
#if CH >= CH8_2 //3
  FunctionPass *createCpu0LongBranchPass(Cpu0TargetMachine &TM);
#endif

} // end namespace llvm;

#endif
