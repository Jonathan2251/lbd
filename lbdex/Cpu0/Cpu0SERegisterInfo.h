//===-- Cpu0SERegisterInfo.h - Cpu032 Register Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Cpu032/64 implementation of the TargetRegisterInfo
// class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_CPU0_CPU0SEREGISTERINFO_H
#define LLVM_LIB_TARGET_CPU0_CPU0SEREGISTERINFO_H

#include "Cpu0Config.h"
#if CH >= CH3_1

#include "Cpu0RegisterInfo.h"

namespace llvm {
class Cpu0SEInstrInfo;

class Cpu0SERegisterInfo : public Cpu0RegisterInfo {
public:
  Cpu0SERegisterInfo(const Cpu0Subtarget &Subtarget);

  const TargetRegisterClass *intRegClass(unsigned Size) const override;

#if CH >= CH9_3 //4
private:
  void eliminateFI(MachineBasicBlock::iterator II, unsigned OpNo,
                   int FrameIndex, int FrameReg, uint64_t StackSize,
                   int64_t SPOffset, bool &IsKill) const override;
#endif //#if CH >= CH9_3 //4
};

} // end namespace llvm

#endif // #if CH >= CH3_1

#endif
