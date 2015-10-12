//===-- Cpu0MCExpr.cpp - Cpu0 specific MC expression classes --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Cpu0.h"

#if CH >= CH5_1

#include "Cpu0MCExpr.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectStreamer.h"

using namespace llvm;

#define DEBUG_TYPE "cpu0mcexpr"

bool Cpu0MCExpr::isSupportedBinaryExpr(MCSymbolRefExpr::VariantKind VK,
                                       const MCBinaryExpr *BE) {
  switch (VK) {
  case MCSymbolRefExpr::VK_Cpu0_ABS_LO:
  case MCSymbolRefExpr::VK_Cpu0_ABS_HI:
    break;
  default:
    return false;
  }

  // We support expressions of the form "(sym1 binop1 sym2) binop2 const",
  // where "binop2 const" is optional.
  if (isa<MCBinaryExpr>(BE->getLHS())) {
    if (!isa<MCConstantExpr>(BE->getRHS()))
      return false;
    BE = cast<MCBinaryExpr>(BE->getLHS());
  }
  return (isa<MCSymbolRefExpr>(BE->getLHS())
          && isa<MCSymbolRefExpr>(BE->getRHS()));
}

const Cpu0MCExpr*
Cpu0MCExpr::create(MCSymbolRefExpr::VariantKind VK, const MCExpr *Expr,
                   MCContext &Ctx) {
  VariantKind Kind;
  switch (VK) {
  case MCSymbolRefExpr::VK_Cpu0_ABS_LO:
    Kind = VK_Cpu0_LO;
    break;
  case MCSymbolRefExpr::VK_Cpu0_ABS_HI:
    Kind = VK_Cpu0_HI;
    break;
  default:
    llvm_unreachable("Invalid kind!");
  }

  return new (Ctx) Cpu0MCExpr(Kind, Expr);
}

void Cpu0MCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
  switch (Kind) {
  default: llvm_unreachable("Invalid kind!");
  case VK_Cpu0_LO: OS << "%lo"; break;
  case VK_Cpu0_HI: OS << "%hi"; break;
  }

  OS << '(';
  Expr->print(OS, MAI);
  OS << ')';
}

bool
Cpu0MCExpr::evaluateAsRelocatableImpl(MCValue &Res,
                                      const MCAsmLayout *Layout,
                                      const MCFixup *Fixup) const {
  return getSubExpr()->evaluateAsRelocatable(Res, Layout, Fixup);
}

void Cpu0MCExpr::visitUsedExpr(MCStreamer &Streamer) const {
  Streamer.visitUsedExpr(*getSubExpr());
}

#endif
