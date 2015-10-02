//===-- Cpu0InstPrinter.cpp - Convert Cpu0 MCInst to assembly syntax ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class prints an Cpu0 MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#include "Cpu0InstPrinter.h"
#if CH >= CH3_2

#include "Cpu0InstrInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "asm-printer"

#define PRINT_ALIAS_INSTR
#include "Cpu0GenAsmWriter.inc"

void Cpu0InstPrinter::printRegName(raw_ostream &OS, unsigned RegNo) const {
//- getRegisterName(RegNo) defined in Cpu0GenAsmWriter.inc which indicate in 
//   Cpu0.td.
  OS << '$' << StringRef(getRegisterName(RegNo)).lower();
}

//@1 {
void Cpu0InstPrinter::printInst(const MCInst *MI, raw_ostream &O,
                                StringRef Annot, const MCSubtargetInfo &STI) {
  // Try to print any aliases first.
  if (!printAliasInstr(MI, O))
//@1 }
    //- printInstruction(MI, O) defined in Cpu0GenAsmWriter.inc which came from 
    //   Cpu0.td indicate.
    printInstruction(MI, O);
  printAnnotation(O, Annot);
}

//@printExpr {
static void printExpr(const MCExpr *Expr, const MCAsmInfo *MAI,
                      raw_ostream &OS) {
//@printExpr body {
  int Offset = 0;
  const MCSymbolRefExpr *SRE;

  if (const MCBinaryExpr *BE = dyn_cast<MCBinaryExpr>(Expr)) {
    SRE = dyn_cast<MCSymbolRefExpr>(BE->getLHS());
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(BE->getRHS());
    assert(SRE && CE && "Binary expression must be sym+const.");
    Offset = CE->getValue();
  }
  else if (!(SRE = dyn_cast<MCSymbolRefExpr>(Expr)))
    assert(false && "Unexpected MCExpr type.");

  MCSymbolRefExpr::VariantKind Kind = SRE->getKind();

  switch (Kind) {
  default:                                 llvm_unreachable("Invalid kind!");
  case MCSymbolRefExpr::VK_None:           break;
#if CH >= CH6_1 //VK_Cpu0_GPREL
// Cpu0_GPREL is for llc -march=cpu0 -relocation-model=static
  case MCSymbolRefExpr::VK_Cpu0_GPREL:     OS << "%gp_rel("; break;
#endif
#if CH >= CH9_1
  case MCSymbolRefExpr::VK_Cpu0_GOT_CALL:  OS << "%call16("; break;
#endif
#if CH >= CH6_1 //VK_Cpu0_GOT16
  case MCSymbolRefExpr::VK_Cpu0_GOT16:     OS << "%got(";    break;
  case MCSymbolRefExpr::VK_Cpu0_GOT:       OS << "%got(";    break;
  case MCSymbolRefExpr::VK_Cpu0_ABS_HI:    OS << "%hi(";     break;
  case MCSymbolRefExpr::VK_Cpu0_ABS_LO:    OS << "%lo(";     break;
#endif
#if CH >= CH12_1
  case MCSymbolRefExpr::VK_Cpu0_TLSGD:     OS << "%tlsgd(";  break;
  case MCSymbolRefExpr::VK_Cpu0_TLSLDM:    OS << "%tlsldm(";  break;
  case MCSymbolRefExpr::VK_Cpu0_DTP_HI:    OS << "%dtp_hi(";  break;
  case MCSymbolRefExpr::VK_Cpu0_DTP_LO:    OS << "%dtp_lo(";  break;
  case MCSymbolRefExpr::VK_Cpu0_GOTTPREL:  OS << "%gottprel("; break;
  case MCSymbolRefExpr::VK_Cpu0_TP_HI:     OS << "%tp_hi("; break;
  case MCSymbolRefExpr::VK_Cpu0_TP_LO:     OS << "%tp_lo("; break;
#endif
#if CH >= CH6_1
  case MCSymbolRefExpr::VK_Cpu0_GOT_HI16:  OS << "%got_hi("; break;
  case MCSymbolRefExpr::VK_Cpu0_GOT_LO16:  OS << "%got_lo("; break;
#endif
  }

  SRE->getSymbol().print(OS, MAI);

  if (Offset) {
    if (Offset > 0)
      OS << '+';
    OS << Offset;
  }

  if ((Kind == MCSymbolRefExpr::VK_Cpu0_GPOFF_HI) ||
      (Kind == MCSymbolRefExpr::VK_Cpu0_GPOFF_LO))
    OS << ")))";
  else if (Kind != MCSymbolRefExpr::VK_None)
    OS << ')';
}

void Cpu0InstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                   raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    printRegName(O, Op.getReg());
    return;
  }

  if (Op.isImm()) {
    O << Op.getImm();
    return;
  }

  assert(Op.isExpr() && "unknown operand kind in printOperand");
  printExpr(Op.getExpr(), &MAI, O);
}

void Cpu0InstPrinter::printUnsignedImm(const MCInst *MI, int opNum,
                                       raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(opNum);
  if (MO.isImm())
    O << (unsigned short int)MO.getImm();
  else
    printOperand(MI, opNum, O);
}

void Cpu0InstPrinter::
printMemOperand(const MCInst *MI, int opNum, raw_ostream &O) {
  // Load/Store memory operands -- imm($reg)
  // If PIC target the target is loaded as the
  // pattern ld $t9,%call16($gp)
  printOperand(MI, opNum+1, O);
  O << "(";
  printOperand(MI, opNum, O);
  O << ")";
}

//#if CH >= CH7_1
// The DAG data node, mem_ea of Cpu0InstrInfo.td, cannot be disabled by
// ch7_1, only opcode node can be disabled.
void Cpu0InstPrinter::
printMemOperandEA(const MCInst *MI, int opNum, raw_ostream &O) {
  // when using stack locations for not load/store instructions
  // print the same way as all normal 3 operand instructions.
  printOperand(MI, opNum, O);
  O << ", ";
  printOperand(MI, opNum+1, O);
  return;
}
//#endif

#endif // #if CH >= CH3_2
