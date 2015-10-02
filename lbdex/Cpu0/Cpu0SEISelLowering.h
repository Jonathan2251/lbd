//===-- Cpu0ISEISelLowering.h - Cpu0ISE DAG Lowering Interface ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Subclass of Cpu0ITargetLowering specialized for cpu032/64.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_CPU0_CPU0SEISELLOWERING_H
#define LLVM_LIB_TARGET_CPU0_CPU0SEISELLOWERING_H

#include "Cpu0Config.h"
#if CH >= CH3_1

#include "Cpu0ISelLowering.h"
#include "Cpu0RegisterInfo.h"

namespace llvm {
  class Cpu0SETargetLowering : public Cpu0TargetLowering  {
  public:
    explicit Cpu0SETargetLowering(const Cpu0TargetMachine &TM,
                                  const Cpu0Subtarget &STI);

    SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;
  private:
#if CH >= CH9_1
    bool isEligibleForTailCallOptimization(const Cpu0CC &Cpu0CCInfo,
                                     unsigned NextStackOffset,
                                     const Cpu0FunctionInfo& FI) const override;
#endif
  };
}

#endif // #if CH >= CH3_1

#endif // Cpu0ISEISELLOWERING_H
