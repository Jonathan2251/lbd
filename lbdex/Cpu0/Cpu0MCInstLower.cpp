//===-- Cpu0MCInstLower.cpp - Convert Cpu0 MachineInstr to MCInst ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains code to lower Cpu0 MachineInstrs to their corresponding
// MCInst records.
//
//===----------------------------------------------------------------------===//

#include "Cpu0MCInstLower.h"
#if CH >= CH3_2

#include "Cpu0AsmPrinter.h"
#include "Cpu0InstrInfo.h"
#include "MCTargetDesc/Cpu0BaseInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"

using namespace llvm;

Cpu0MCInstLower::Cpu0MCInstLower(Cpu0AsmPrinter &asmprinter)
  : AsmPrinter(asmprinter) {}

void Cpu0MCInstLower::Initialize(MCContext* C) {
  Ctx = C;
}

#if CH >= CH6_1 //1
//@LowerSymbolOperand {
MCOperand Cpu0MCInstLower::LowerSymbolOperand(const MachineOperand &MO,
                                              MachineOperandType MOTy,
                                              unsigned Offset) const {
  MCSymbolRefExpr::VariantKind Kind = MCSymbolRefExpr::VK_None;
  Cpu0MCExpr::Cpu0ExprKind TargetKind = Cpu0MCExpr::CEK_None;
  const MCSymbol *Symbol;

  switch(MO.getTargetFlags()) {
  default:                   llvm_unreachable("Invalid target flag!");
  case Cpu0II::MO_NO_FLAG:
    break;

// Cpu0_GPREL is for llc -march=cpu0 -relocation-model=static -cpu0-islinux-
//  format=false (global var in .sdata).
  case Cpu0II::MO_GPREL:
    TargetKind = Cpu0MCExpr::CEK_GPREL;
    break;

#if CH >= CH9_1 //1
  case Cpu0II::MO_GOT_CALL:
    TargetKind = Cpu0MCExpr::CEK_GOT_CALL;
    break;
#endif
  case Cpu0II::MO_GOT:
    TargetKind = Cpu0MCExpr::CEK_GOT;
    break;
// ABS_HI and ABS_LO is for llc -march=cpu0 -relocation-model=static (global 
//  var in .data).
  case Cpu0II::MO_ABS_HI:
    TargetKind = Cpu0MCExpr::CEK_ABS_HI;
    break;
  case Cpu0II::MO_ABS_LO:
    TargetKind = Cpu0MCExpr::CEK_ABS_LO;
    break;
#if CH >= CH12_1
  case Cpu0II::MO_TLSGD:
    TargetKind = Cpu0MCExpr::CEK_TLSGD;
    break;
  case Cpu0II::MO_TLSLDM:
    TargetKind = Cpu0MCExpr::CEK_TLSLDM;
    break;
  case Cpu0II::MO_DTP_HI:
    TargetKind = Cpu0MCExpr::CEK_DTP_HI;
    break;
  case Cpu0II::MO_DTP_LO:
    TargetKind = Cpu0MCExpr::CEK_DTP_LO;
    break;
  case Cpu0II::MO_GOTTPREL:
    TargetKind = Cpu0MCExpr::CEK_GOTTPREL;
    break;
  case Cpu0II::MO_TP_HI:
    TargetKind = Cpu0MCExpr::CEK_TP_HI;
    break;
  case Cpu0II::MO_TP_LO:
    TargetKind = Cpu0MCExpr::CEK_TP_LO;
    break;
#endif
  case Cpu0II::MO_GOT_HI16:
    TargetKind = Cpu0MCExpr::CEK_GOT_HI16;
    break;
  case Cpu0II::MO_GOT_LO16:
    TargetKind = Cpu0MCExpr::CEK_GOT_LO16;
    break;
  }

  switch (MOTy) {
  case MachineOperand::MO_GlobalAddress:
    Symbol = AsmPrinter.getSymbol(MO.getGlobal());
    Offset += MO.getOffset();
    break;

#if CH >= CH8_1
  case MachineOperand::MO_MachineBasicBlock:
    Symbol = MO.getMBB()->getSymbol();
    break;

  case MachineOperand::MO_BlockAddress:
    Symbol = AsmPrinter.GetBlockAddressSymbol(MO.getBlockAddress());
    Offset += MO.getOffset();
    break;
#endif

#if CH >= CH9_1 //2
  case MachineOperand::MO_ExternalSymbol:
    Symbol = AsmPrinter.GetExternalSymbolSymbol(MO.getSymbolName());
    Offset += MO.getOffset();
    break;
#endif

#if CH >= CH8_1
  case MachineOperand::MO_JumpTableIndex:
    Symbol = AsmPrinter.GetJTISymbol(MO.getIndex());
    break;
#endif

  default:
    llvm_unreachable("<unknown operand type>");
  }

  const MCExpr *Expr = MCSymbolRefExpr::create(Symbol, Kind, *Ctx);

  if (Offset) {
    // Assume offset is never negative.
    assert(Offset > 0);
    Expr = MCBinaryExpr::createAdd(Expr, MCConstantExpr::create(Offset, *Ctx),
                                   *Ctx);
  }

  if (TargetKind != Cpu0MCExpr::CEK_None)
    Expr = Cpu0MCExpr::create(TargetKind, Expr, *Ctx);

  return MCOperand::createExpr(Expr);

}
//@LowerSymbolOperand }
#endif // if CH >= CH6_1 //1

static void CreateMCInst(MCInst& Inst, unsigned Opc, const MCOperand& Opnd0,
                         const MCOperand& Opnd1,
                         const MCOperand& Opnd2 = MCOperand()) {
  Inst.setOpcode(Opc);
  Inst.addOperand(Opnd0);
  Inst.addOperand(Opnd1);
  if (Opnd2.isValid())
    Inst.addOperand(Opnd2);
}

#if CH >= CH6_1 //2
// Lower ".cpload $reg" to
//  "lui   $gp, %hi(_gp_disp)"
//  "addiu $gp, $gp, %lo(_gp_disp)"
//  "addu  $gp, $gp, $t9"
void Cpu0MCInstLower::LowerCPLOAD(SmallVector<MCInst, 4>& MCInsts) {
  MCOperand GPReg = MCOperand::createReg(Cpu0::GP);
  MCOperand T9Reg = MCOperand::createReg(Cpu0::T9);
  StringRef SymName("_gp_disp");
  const MCSymbol *Sym = Ctx->getOrCreateSymbol(SymName);
  const Cpu0MCExpr *MCSym;

  MCSym = Cpu0MCExpr::create(Sym, Cpu0MCExpr::CEK_ABS_HI, *Ctx);
  MCOperand SymHi = MCOperand::createExpr(MCSym);
  MCSym = Cpu0MCExpr::create(Sym, Cpu0MCExpr::CEK_ABS_LO, *Ctx);
  MCOperand SymLo = MCOperand::createExpr(MCSym);

  MCInsts.resize(3);

  CreateMCInst(MCInsts[0], Cpu0::LUi, GPReg, SymHi);
  CreateMCInst(MCInsts[1], Cpu0::ORi, GPReg, GPReg, SymLo);
  CreateMCInst(MCInsts[2], Cpu0::ADD, GPReg, GPReg, T9Reg);
}
#endif

#if CH >= CH9_3
#ifdef ENABLE_GPRESTORE
// Lower ".cprestore offset" to "st $gp, offset($sp)".
void Cpu0MCInstLower::LowerCPRESTORE(int64_t Offset,
                                     SmallVector<MCInst, 4>& MCInsts) {
  assert(isInt<32>(Offset) && (Offset >= 0) &&
         "Imm operand of .cprestore must be a non-negative 32-bit value.");

  MCOperand SPReg = MCOperand::createReg(Cpu0::SP), BaseReg = SPReg;
  MCOperand GPReg = MCOperand::createReg(Cpu0::GP);
  MCOperand ZEROReg = MCOperand::createReg(Cpu0::ZERO);

  if (!isInt<16>(Offset)) {
    unsigned Hi = ((Offset + 0x8000) >> 16) & 0xffff;
    Offset &= 0xffff;
    MCOperand ATReg = MCOperand::createReg(Cpu0::AT);
    BaseReg = ATReg;

    // lui   at,hi
    // add   at,at,sp
    MCInsts.resize(2);
    CreateMCInst(MCInsts[0], Cpu0::LUi, ATReg, ZEROReg, MCOperand::createImm(Hi));
    CreateMCInst(MCInsts[1], Cpu0::ADD, ATReg, ATReg, SPReg);
  }

  MCInst St;
  CreateMCInst(St, Cpu0::ST, GPReg, BaseReg, MCOperand::createImm(Offset));
  MCInsts.push_back(St);
}
#endif
#endif //#if CH >= CH9_3

//@LowerOperand {
MCOperand Cpu0MCInstLower::LowerOperand(const MachineOperand& MO,
                                        unsigned offset) const {
  MachineOperandType MOTy = MO.getType();

  switch (MOTy) {
  //@2
  default: llvm_unreachable("unknown operand type");
  case MachineOperand::MO_Register:
    // Ignore all implicit register operands.
    if (MO.isImplicit()) break;
    return MCOperand::createReg(MO.getReg());
  case MachineOperand::MO_Immediate:
    return MCOperand::createImm(MO.getImm() + offset);
#if CH >= CH8_1
  case MachineOperand::MO_MachineBasicBlock:
#endif
#if CH >= CH9_1 //3
  case MachineOperand::MO_ExternalSymbol:
#endif
#if CH >= CH8_1
  case MachineOperand::MO_JumpTableIndex:
  case MachineOperand::MO_BlockAddress:
#endif
#if CH >= CH6_1 //3
  case MachineOperand::MO_GlobalAddress:
//@1
    return LowerSymbolOperand(MO, MOTy, offset);
#endif
  case MachineOperand::MO_RegisterMask:
    break;
 }

  return MCOperand();
}

#if CH >= CH8_2 //1
MCOperand Cpu0MCInstLower::createSub(MachineBasicBlock *BB1,
                                     MachineBasicBlock *BB2,
                                     Cpu0MCExpr::Cpu0ExprKind Kind) const {
  const MCSymbolRefExpr *Sym1 = MCSymbolRefExpr::create(BB1->getSymbol(), *Ctx);
  const MCSymbolRefExpr *Sym2 = MCSymbolRefExpr::create(BB2->getSymbol(), *Ctx);
  const MCBinaryExpr *Sub = MCBinaryExpr::createSub(Sym1, Sym2, *Ctx);

  return MCOperand::createExpr(Cpu0MCExpr::create(Kind, Sub, *Ctx));
}

void Cpu0MCInstLower::
lowerLongBranchLUi(const MachineInstr *MI, MCInst &OutMI) const {
  OutMI.setOpcode(Cpu0::LUi);

  // Lower register operand.
  OutMI.addOperand(LowerOperand(MI->getOperand(0)));

  // Create %hi($tgt-$baltgt).
  OutMI.addOperand(createSub(MI->getOperand(1).getMBB(),
                             MI->getOperand(2).getMBB(),
                             Cpu0MCExpr::CEK_ABS_HI));
}

void Cpu0MCInstLower::
lowerLongBranchADDiu(const MachineInstr *MI, MCInst &OutMI, int Opcode,
                     Cpu0MCExpr::Cpu0ExprKind Kind) const {
  OutMI.setOpcode(Opcode);

  // Lower two register operands.
  for (unsigned I = 0, E = 2; I != E; ++I) {
    const MachineOperand &MO = MI->getOperand(I);
    OutMI.addOperand(LowerOperand(MO));
  }

  // Create %lo($tgt-$baltgt) or %hi($tgt-$baltgt).
  OutMI.addOperand(createSub(MI->getOperand(2).getMBB(),
                             MI->getOperand(3).getMBB(), Kind));
}

bool Cpu0MCInstLower::lowerLongBranch(const MachineInstr *MI,
                                      MCInst &OutMI) const {
  switch (MI->getOpcode()) {
  default:
    return false;
  case Cpu0::LONG_BRANCH_LUi:
    lowerLongBranchLUi(MI, OutMI);
    return true;
  case Cpu0::LONG_BRANCH_ADDiu:
    lowerLongBranchADDiu(MI, OutMI, Cpu0::ADDiu,
                         Cpu0MCExpr::CEK_ABS_LO);
    return true;
  }
}
#endif

void Cpu0MCInstLower::Lower(const MachineInstr *MI, MCInst &OutMI) const {
#if CH >= CH8_2 //2
  if (lowerLongBranch(MI, OutMI))
    return;
#endif
  OutMI.setOpcode(MI->getOpcode());

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    MCOperand MCOp = LowerOperand(MO);

    if (MCOp.isValid())
      OutMI.addOperand(MCOp);
  }
}

#endif // #if CH >= CH3_2
