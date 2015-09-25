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

#ifndef CPU0SEINSTRUCTIONINFO_H
#define CPU0SEINSTRUCTIONINFO_H

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
  void copyPhysReg(MachineBasicBlock &MBB,
                   MachineBasicBlock::iterator MI, DebugLoc DL,
                   unsigned DestReg, unsigned SrcReg,
                   bool KillSrc) const override;
#endif

#if CH >= CH3_4
  void storeRegToStack(MachineBasicBlock &MBB,
                       MachineBasicBlock::iterator MI,
                       unsigned SrcReg, bool isKill, int FrameIndex,
                       const TargetRegisterClass *RC,
                       const TargetRegisterInfo *TRI,
                       int64_t Offset) const override;

  void loadRegFromStack(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator MI,
                        unsigned DestReg, int FrameIndex,
                        const TargetRegisterClass *RC,
                        const TargetRegisterInfo *TRI,
                        int64_t Offset) const override;

  bool expandPostRAPseudo(MachineBasicBlock::iterator MI) const override;

  /// Adjust SP by Amount bytes.
  void adjustStackPtr(Cpu0FunctionInfo *Cpu0FI, unsigned SP, int64_t Amount,
                      MachineBasicBlock &MBB, MachineBasicBlock::iterator I) 
                      const;

  /// Emit a series of instructions to load an immediate. If NewImm is a
  /// non-NULL parameter, the last instruction is not emitted, but instead
  /// its immediate operand is returned in NewImm.
  unsigned loadImmediate(int64_t Imm, MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator II, DebugLoc DL,
                         unsigned *NewImm) const;
private:
  void ExpandRetLR(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                   unsigned Opc) const;
#endif
};

}

#endif // #if CH >= CH3_1

#endif
