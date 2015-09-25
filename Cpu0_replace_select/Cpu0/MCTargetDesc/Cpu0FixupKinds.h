//===-- Cpu0FixupKinds.h - Cpu0 Specific Fixup Entries ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CPU0_CPU0FIXUPKINDS_H
#define LLVM_CPU0_CPU0FIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
namespace Cpu0 {
  // Although most of the current fixup types reflect a unique relocation
  // one can have multiple fixup types for a given relocation and thus need
  // to be uniquely named.
  //
  // This table *must* be in the save order of
  // MCFixupKindInfo Infos[Cpu0::NumTargetFixupKinds]
  // in Cpu0AsmBackend.cpp.
  //
  enum Fixups {
    // Branch fixups resulting in R_CPU0_16.
    fixup_Cpu0_16 = FirstTargetFixupKind,

    // Pure 32 bit data fixup resulting in - R_CPU0_32.
    fixup_Cpu0_32,

    // Full 32 bit data relative data fixup resulting in - R_CPU0_REL32.
    fixup_Cpu0_REL32,

    // Jump 24 bit fixup resulting in - R_CPU0_24.
    fixup_Cpu0_24,

    // Pure upper 16 bit fixup resulting in - R_CPU0_HI16.
    fixup_Cpu0_HI16,

    // Pure lower 16 bit fixup resulting in - R_CPU0_LO16.
    fixup_Cpu0_LO16,

    // 16 bit fixup for GP offest resulting in - R_CPU0_GPREL16.
    fixup_Cpu0_GPREL16,

    // 16 bit literal fixup resulting in - R_CPU0_LITERAL.
    fixup_Cpu0_LITERAL,

    // Global symbol fixup resulting in - R_CPU0_GOT16.
    fixup_Cpu0_GOT_Global,

    // Local symbol fixup resulting in - R_CPU0_GOT16.
    fixup_Cpu0_GOT_Local,

    // PC relative branch fixup resulting in - R_CPU0_PC24.
    // cpu0 PC24, e.g. jeq
    fixup_Cpu0_PC24,

    // resulting in - R_CPU0_CALL24.
    fixup_Cpu0_CALL24,

    // resulting in - R_CPU0_GPREL32.
    fixup_Cpu0_GPREL32,

    // resulting in - R_CPU0_SHIFT5.
    fixup_Cpu0_SHIFT5,

    // resulting in - R_CPU0_SHIFT6.
    fixup_Cpu0_SHIFT6,

    // Pure 64 bit data fixup resulting in - R_CPU0_64.
    fixup_Cpu0_64,

    // resulting in - R_CPU0_TLS_GD.
    fixup_Cpu0_TLSGD,

    // resulting in - R_CPU0_TLS_GOTTPREL.
    fixup_Cpu0_GOTTPREL,

    // resulting in - R_CPU0_TLS_TPREL_HI16.
    fixup_Cpu0_TPREL_HI,

    // resulting in - R_CPU0_TLS_TPREL_LO16.
    fixup_Cpu0_TPREL_LO,

    // resulting in - R_CPU0_TLS_LDM.
    fixup_Cpu0_TLSLDM,

    // resulting in - R_CPU0_TLS_DTPREL_HI16.
    fixup_Cpu0_DTPREL_HI,

    // resulting in - R_CPU0_TLS_DTPREL_LO16.
    fixup_Cpu0_DTPREL_LO,

    // PC relative branch fixup resulting in - R_CPU0_PC16
    fixup_Cpu0_Branch_PCRel,

    // resulting in - R_MIPS_GOT_HI16
    fixup_Cpu0_GOT_HI16,

    // resulting in - R_MIPS_GOT_LO16
    fixup_Cpu0_GOT_LO16,

    // Marker
    LastTargetFixupKind,
    NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
  };
} // namespace Cpu0
} // namespace llvm


#endif // LLVM_CPU0_CPU0FIXUPKINDS_H
