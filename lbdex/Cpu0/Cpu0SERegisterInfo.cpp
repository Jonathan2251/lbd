//===-- Cpu0SERegisterInfo.cpp - CPU0 Register Information ------== -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the CPU0 implementation of the TargetRegisterInfo
// class.
//
//===----------------------------------------------------------------------===//

#include "Cpu0SERegisterInfo.h"
#include "Cpu0MachineFunction.h"
#include "Cpu0SEInstrInfo.h"
#include "Cpu0Subtarget.h"
#if CH >= CH3_1

using namespace llvm;

#define DEBUG_TYPE "cpu0-reg-info"

Cpu0SERegisterInfo::Cpu0SERegisterInfo(const Cpu0Subtarget &ST)
  : Cpu0RegisterInfo(ST) {}

const TargetRegisterClass *
Cpu0SERegisterInfo::intRegClass(unsigned Size) const {
  return &Cpu0::CPURegsRegClass;
}

void Cpu0SERegisterInfo::eliminateFI(MachineBasicBlock::iterator II,
                                     unsigned OpNo, int FrameIndex,
                                     int FrameReg, uint64_t StackSize,
                                     int64_t Offset, bool &IsKill) const {
  MachineInstr &MI = *II;
  if (!MI.isDebugValue()) {
    // Make sure Offset fits within the field available.
    if (isInt<16>(Offset)) {
      assert("FIXME!");
    } else if (!isInt<16>(Offset)) {
      // Otherwise split the offset into 16-bit pieces and add it in multiple
      // instructions.
      MachineBasicBlock &MBB = *MI.getParent();
      DebugLoc DL = II->getDebugLoc();
      unsigned NewImm = 0;
      const Cpu0SEInstrInfo &TII =
          *static_cast<const Cpu0SEInstrInfo *>(
              MBB.getParent()->getSubtarget().getInstrInfo());
      unsigned Reg = TII.loadImmediate(Offset, MBB, II, DL, &NewImm);
      BuildMI(MBB, II, DL, TII.get(Cpu0::ADDiu), Reg).addReg(FrameReg)
        .addReg(Reg, RegState::Kill);

      FrameReg = Reg;
      Offset = SignExtend64<16>(NewImm);
      IsKill = true;
    }
  }
}
#endif
