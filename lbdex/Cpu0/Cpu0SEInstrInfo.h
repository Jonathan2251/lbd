//===-- Cpu0SEInstrInfo.h - Cpu032/64 Instruction Information ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Cpu032/64 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_CPU0_CPU0SEINSTRINFO_H
#define LLVM_LIB_TARGET_CPU0_CPU0SEINSTRINFO_H

#include "Cpu0Config.h"
#if CH >= CH3_1

#include "Cpu0InstrInfo.h"
#include "Cpu0SERegisterInfo.h"
#include "Cpu0MachineFunction.h"

namespace llvm {

class Cpu0SEInstrInfo : public Cpu0InstrInfo {
  const Cpu0SERegisterInfo RI;

public:
  explicit Cpu0SEInstrInfo(const Cpu0Subtarget &STI);

  const Cpu0RegisterInfo &getRegisterInfo() const override;

#if CH >= CH4_1
  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                   const DebugLoc &DL, MCRegister DestReg, MCRegister SrcReg,
                   bool KillSrc) const override;
#endif

#if CH >= CH3_5 //1
  void storeRegToStack(MachineBasicBlock &MBB,
                       MachineBasicBlock::iterator MI,
                       Register SrcReg, bool isKill, int FrameIndex,
                       const TargetRegisterClass *RC,
                       const TargetRegisterInfo *TRI,
                       int64_t Offset) const override;

  void loadRegFromStack(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator MI,
                        Register DestReg, int FrameIndex,
                        const TargetRegisterClass *RC,
                        const TargetRegisterInfo *TRI,
                        int64_t Offset) const override;
#endif //#if CH >= CH3_5 //1

#if CH >= CH3_4 //1
//@expandPostRAPseudo
  bool expandPostRAPseudo(MachineInstr &MI) const override;
#endif //#if CH >= CH3_4 //1

#if CH >= CH3_5 //2
  /// Adjust SP by Amount bytes.
  void adjustStackPtr(unsigned SP, int64_t Amount, MachineBasicBlock &MBB,
                      MachineBasicBlock::iterator I) const override;

  /// Emit a series of instructions to load an immediate. If NewImm is a
  /// non-NULL parameter, the last instruction is not emitted, but instead
  /// its immediate operand is returned in NewImm.
  unsigned loadImmediate(int64_t Imm, MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator II, const DebugLoc &DL,
                         unsigned *NewImm) const;
#endif //#if CH >= CH3_5 //2
#if CH >= CH3_4 //2
private:
  void expandRetLR(MachineBasicBlock &MBB, MachineBasicBlock::iterator I) const;
#endif //#if CH >= CH3_4 //2

#if CH >= CH8_2 //1
  unsigned getOppositeBranchOpc(unsigned Opc) const override;
#endif
  
#if CH >= CH9_3
  void expandEhReturn(MachineBasicBlock &MBB,
                      MachineBasicBlock::iterator I) const;
#endif
};

}

#endif // #if CH >= CH3_1

#endif
