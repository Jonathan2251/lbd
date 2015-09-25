//===-- Cpu0BaseInfo.h - Top level definitions for CPU0 MC ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone helper functions and enum definitions for
// the Cpu0 target useful for the compiler back-end and the MC libraries.
//
//===----------------------------------------------------------------------===//
#ifndef CPU0BASEINFO_H
#define CPU0BASEINFO_H

#include "Cpu0Config.h"
#if CH >= CH3_2

#if CH >= CH5_1
#include "Cpu0FixupKinds.h"
#endif
#include "Cpu0MCTargetDesc.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

/// Cpu0II - This namespace holds all of the target specific flags that
/// instruction info tracks.
//@Cpu0II
namespace Cpu0II {
  /// Target Operand Flag enum.
  enum TOF {
    //===------------------------------------------------------------------===//
    // Cpu0 Specific MachineOperand flags.

    MO_NO_FLAG,

#if CH >= CH6_1
    /// MO_GOT16 - Represents the offset into the global offset table at which
    /// the address the relocation entry symbol resides during execution.
    MO_GOT16,
    MO_GOT,
#endif

    /// MO_GOT_CALL - Represents the offset into the global offset table at
    /// which the address of a call site relocation entry symbol resides
    /// during execution. This is different from the above since this flag
    /// can only be present in call instructions.
    MO_GOT_CALL,

    /// MO_GPREL - Represents the offset from the current gp value to be used
    /// for the relocatable object file being produced.
    MO_GPREL,

    /// MO_ABS_HI/LO - Represents the hi or low part of an absolute symbol
    /// address.
    MO_ABS_HI,
    MO_ABS_LO,

#if CH >= CH12_1
    /// MO_TLSGD - Represents the offset into the global offset table at which
    // the module ID and TSL block offset reside during execution (General
    // Dynamic TLS).
    MO_TLSGD,

    /// MO_TLSLDM - Represents the offset into the global offset table at which
    // the module ID and TSL block offset reside during execution (Local
    // Dynamic TLS).
    MO_TLSLDM,
    MO_DTP_HI,
    MO_DTP_LO,

    /// MO_GOTTPREL - Represents the offset from the thread pointer (Initial
    // Exec TLS).
    MO_GOTTPREL,

    /// MO_TPREL_HI/LO - Represents the hi and low part of the offset from
    // the thread pointer (Local Exec TLS).
    MO_TP_HI,
    MO_TP_LO,
#endif
    MO_GOT_DISP,
    MO_GOT_PAGE,
    MO_GOT_OFST,

    // N32/64 Flags.
    MO_GPOFF_HI,
    MO_GPOFF_LO,

    /// MO_GOT_HI16/LO16 - Relocations used for large GOTs.
    MO_GOT_HI16,
    MO_GOT_LO16
  }; // enum TOF {

  enum {
    //===------------------------------------------------------------------===//
    // Instruction encodings.  These are the standard/most common forms for
    // Cpu0 instructions.
    //

    // Pseudo - This represents an instruction that is a pseudo instruction
    // or one that has not been implemented yet.  It is illegal to code generate
    // it, but tolerated for intermediate implementation stages.
    Pseudo   = 0,

    /// FrmR - This form is for instructions of the format R.
    FrmR  = 1,
    /// FrmI - This form is for instructions of the format I.
    FrmI  = 2,
    /// FrmJ - This form is for instructions of the format J.
    FrmJ  = 3,
    /// FrmOther - This form is for instructions that have no specific format.
    FrmOther = 4,

    FormMask = 15
  };
}

//@get register number
/// getCpu0RegisterNumbering - Given the enum value for some register,
/// return the number that it corresponds to.
inline static unsigned getCpu0RegisterNumbering(unsigned RegEnum)
{
  switch (RegEnum) {
  //@1
  case Cpu0::ZERO:
    return 0;
  case Cpu0::AT:
    return 1;
  case Cpu0::V0:
    return 2;
  case Cpu0::V1:
    return 3;
  case Cpu0::A0:
    return 4;
  case Cpu0::A1:
    return 5;
  case Cpu0::T9:
    return 6;
  case Cpu0::T0:
    return 7;
  case Cpu0::T1:
    return 8;
  case Cpu0::S0:
    return 9;
  case Cpu0::S1:
    return 10;
  case Cpu0::GP:
    return 11;
  case Cpu0::FP:
    return 12;
  case Cpu0::SP:
    return 13;
  case Cpu0::LR:
    return 14;
  case Cpu0::SW:
    return 15;
#if CH >= CH4_1
  case Cpu0::HI:
    return 18;
  case Cpu0::LO:
    return 19;
#endif
  case Cpu0::PC:
    return 0;
  case Cpu0::EPC:
    return 1;
  default: llvm_unreachable("Unknown register number!");
  }
}

}

#endif // #if CH >= CH3_2

#endif
