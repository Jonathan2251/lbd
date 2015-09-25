//===- Cpu0Disassembler.cpp - Disassembler for Cpu0 -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of the Cpu0 Disassembler.
//
//===----------------------------------------------------------------------===//

#include "Cpu0.h"
#include "Cpu0Subtarget.h"
#include "Cpu0RegisterInfo.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCFixedLenDisassembler.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

typedef MCDisassembler::DecodeStatus DecodeStatus;

/// Cpu0Disassembler - a disasembler class for Cpu032.
class Cpu0Disassembler : public MCDisassembler {
public:
  /// Constructor     - Initializes the disassembler.
  ///
  Cpu0Disassembler(const MCSubtargetInfo &STI, bool bigEndian) :
    MCDisassembler(STI), isBigEndian(bigEndian) {
  }

  ~Cpu0Disassembler() {
  }

  /// getInstruction - See MCDisassembler.
  DecodeStatus getInstruction(MCInst &instr,
                              uint64_t &size,
                              const MemoryObject &region,
                              uint64_t address,
                              raw_ostream &vStream,
                              raw_ostream &cStream) const;

private:
  bool isBigEndian;
};

// Decoder tables for Cpu0 register
static const unsigned CPURegsTable[] = {
  Cpu0::ZERO, Cpu0::AT, Cpu0::V0, Cpu0::V1,
  Cpu0::A0, Cpu0::A1, Cpu0::T9, Cpu0::S0, 
  Cpu0::S1, Cpu0::S2, Cpu0::GP, Cpu0::FP, 
  Cpu0::SW, Cpu0::SP, Cpu0::LR, Cpu0::PC
};

static DecodeStatus DecodeCPURegsRegisterClass(MCInst &Inst,
                                               unsigned RegNo,
                                               uint64_t Address,
                                               const void *Decoder);
static DecodeStatus DecodeCMPInstruction(MCInst &Inst,
                                       unsigned Insn,
                                       uint64_t Address,
                                       const void *Decoder);
static DecodeStatus DecodeBranchTarget(MCInst &Inst,
                                       unsigned Insn,
                                       uint64_t Address,
                                       const void *Decoder);
static DecodeStatus DecodeJumpRelativeTarget(MCInst &Inst,
                                       unsigned Insn,
                                       uint64_t Address,
                                       const void *Decoder);
static DecodeStatus DecodeJumpAbsoluteTarget(MCInst &Inst,
                                     unsigned Insn,
                                     uint64_t Address,
                                     const void *Decoder);

static DecodeStatus DecodeMem(MCInst &Inst,
                              unsigned Insn,
                              uint64_t Address,
                              const void *Decoder);
static DecodeStatus DecodeSimm16(MCInst &Inst,
                                 unsigned Insn,
                                 uint64_t Address,
                                 const void *Decoder);

namespace llvm {
extern Target TheCpu0elTarget, TheCpu0Target, TheCpu064Target,
              TheCpu064elTarget;
}

static MCDisassembler *createCpu0Disassembler(
                       const Target &T,
                       const MCSubtargetInfo &STI) {
  return new Cpu0Disassembler(STI,true);
}

static MCDisassembler *createCpu0elDisassembler(
                       const Target &T,
                       const MCSubtargetInfo &STI) {
  return new Cpu0Disassembler(STI,false);
}

extern "C" void LLVMInitializeCpu0Disassembler() {
  // Register the disassembler.
  TargetRegistry::RegisterMCDisassembler(TheCpu0Target,
                                         createCpu0Disassembler);
  TargetRegistry::RegisterMCDisassembler(TheCpu0elTarget,
                                         createCpu0elDisassembler);
}


#include "Cpu0GenDisassemblerTables.inc"

  /// readInstruction - read four bytes from the MemoryObject
  /// and return 32 bit word sorted according to the given endianess
static DecodeStatus readInstruction32(const MemoryObject &region,
                                      uint64_t address,
                                      uint64_t &size,
                                      uint32_t &insn,
                                      bool isBigEndian) {
  uint8_t Bytes[4];

  // We want to read exactly 4 Bytes of data.
  if (region.readBytes(address, 4, (uint8_t*)Bytes, NULL) == -1) {
    size = 0;
    return MCDisassembler::Fail;
  }

  if (isBigEndian) {
    // Encoded as a big-endian 32-bit word in the stream.
    insn = (Bytes[3] <<  0) |
           (Bytes[2] <<  8) |
           (Bytes[1] << 16) |
           (Bytes[0] << 24);
  }
  else {
    // Encoded as a small-endian 32-bit word in the stream.
    insn = (Bytes[0] <<  0) |
           (Bytes[1] <<  8) |
           (Bytes[2] << 16) |
           (Bytes[3] << 24);
  }

  return MCDisassembler::Success;
}

DecodeStatus
Cpu0Disassembler::getInstruction(MCInst &instr,
                                 uint64_t &Size,
                                 const MemoryObject &Region,
                                 uint64_t Address,
                                 raw_ostream &vStream,
                                 raw_ostream &cStream) const {
  uint32_t Insn;

  DecodeStatus Result = readInstruction32(Region, Address, Size,
                                          Insn, isBigEndian);
  if (Result == MCDisassembler::Fail)
    return MCDisassembler::Fail;

  // Calling the auto-generated decoder function.
  Result = decodeInstruction(DecoderTableCpu032, instr, Insn, Address,
                             this, STI);
  if (Result != MCDisassembler::Fail) {
    Size = 4;
    return Result;
  }

  return MCDisassembler::Fail;
}

static DecodeStatus DecodeCPURegsRegisterClass(MCInst &Inst,
                                               unsigned RegNo,
                                               uint64_t Address,
                                               const void *Decoder) {
  if (RegNo > 16)
    return MCDisassembler::Fail;

  Inst.addOperand(MCOperand::CreateReg(CPURegsTable[RegNo]));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeMem(MCInst &Inst,
                              unsigned Insn,
                              uint64_t Address,
                              const void *Decoder) {
  int Offset = SignExtend32<16>(Insn & 0xffff);
  int Reg = (int)fieldFromInstruction(Insn, 20, 4);
  int Base = (int)fieldFromInstruction(Insn, 16, 4);

  Inst.addOperand(MCOperand::CreateReg(CPURegsTable[Reg]));
  Inst.addOperand(MCOperand::CreateReg(CPURegsTable[Base]));
  Inst.addOperand(MCOperand::CreateImm(Offset));

  return MCDisassembler::Success;
}

/* CMP instruction define $rc and then $ra, $rb; The printOperand() print 
operand 1 and operand 2 (operand 0 is $rc and operand 1 is $ra), so we Create 
register $rc first and create $ra next, as follows,

// Cpu0InstrInfo.td
class CmpInstr<bits<8> op, string instr_asm, 
                    InstrItinClass itin, RegisterClass RC, RegisterClass RD, bit isComm = 0>:
  FA<op, (outs RD:$rc), (ins RC:$ra, RC:$rb),
     !strconcat(instr_asm, "\t$ra, $rb"), [], itin> {

// Cpu0AsmWriter.inc
void Cpu0InstPrinter::printInstruction(const MCInst *MI, raw_ostream &O) {
...
  case 3:
    // CMP, JEQ, JGE, JGT, JLE, JLT, JNE
    printOperand(MI, 1, O); 
    break;
...
  case 1:
    // CMP
    printOperand(MI, 2, O); 
    return;
    break;
*/
static DecodeStatus DecodeCMPInstruction(MCInst &Inst,
                                       unsigned Insn,
                                       uint64_t Address,
                                       const void *Decoder) {
  int Reg_a = (int)fieldFromInstruction(Insn, 20, 4);
  int Reg_b = (int)fieldFromInstruction(Insn, 16, 4);
  int Reg_c = (int)fieldFromInstruction(Insn, 12, 4);

  Inst.addOperand(MCOperand::CreateReg(CPURegsTable[Reg_c]));
  Inst.addOperand(MCOperand::CreateReg(CPURegsTable[Reg_a]));
  Inst.addOperand(MCOperand::CreateReg(CPURegsTable[Reg_b]));
  return MCDisassembler::Success;
}

/* CBranch instruction define $ra and then imm24; The printOperand() print 
operand 1 (operand 0 is $ra and operand 1 is imm24), so we Create register 
operand first and create imm24 next, as follows,

// Cpu0InstrInfo.td
class CBranch<bits<8> op, string instr_asm, RegisterClass RC,
                   list<Register> UseRegs>:
  FJ<op, (outs), (ins RC:$ra, brtarget:$addr),
             !strconcat(instr_asm, "\t$addr"),
             [(brcond RC:$ra, bb:$addr)], IIBranch> {

// Cpu0AsmWriter.inc
void Cpu0InstPrinter::printInstruction(const MCInst *MI, raw_ostream &O) {
...
  case 3:
    // CMP, JEQ, JGE, JGT, JLE, JLT, JNE
    printOperand(MI, 1, O); 
    break;
*/
static DecodeStatus DecodeBranchTarget(MCInst &Inst,
                                       unsigned Insn,
                                       uint64_t Address,
                                       const void *Decoder) {
  int BranchOffset = fieldFromInstruction(Insn, 0, 24);
  if (BranchOffset > 0x8fffff)
  	BranchOffset = -1*(0x1000000 - BranchOffset);
  Inst.addOperand(MCOperand::CreateReg(CPURegsTable[0]));
  Inst.addOperand(MCOperand::CreateImm(BranchOffset));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeJumpRelativeTarget(MCInst &Inst,
                                     unsigned Insn,
                                     uint64_t Address,
                                     const void *Decoder) {

  int JumpOffset = fieldFromInstruction(Insn, 0, 24);
  if (JumpOffset > 0x8fffff)
  	JumpOffset = -1*(0x1000000 - JumpOffset);
  Inst.addOperand(MCOperand::CreateImm(JumpOffset));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeJumpAbsoluteTarget(MCInst &Inst,
                                     unsigned Insn,
                                     uint64_t Address,
                                     const void *Decoder) {

  unsigned JumpOffset = fieldFromInstruction(Insn, 0, 24);
  Inst.addOperand(MCOperand::CreateImm(JumpOffset));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeSimm16(MCInst &Inst,
                                 unsigned Insn,
                                 uint64_t Address,
                                 const void *Decoder) {
  Inst.addOperand(MCOperand::CreateImm(SignExtend32<16>(Insn)));
  return MCDisassembler::Success;
}

