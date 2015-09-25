//===-- Cpu0MCTargetDesc.h - Cpu0 Target Descriptions -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Cpu0 specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef CPU0MCTARGETDESC_H
#define CPU0MCTARGETDESC_H

#include "Cpu0Config.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
#if CH >= CH3_2
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCObjectWriter;
class MCRegisterInfo;
class MCSubtargetInfo;
class StringRef;
#endif
class Target;
#if CH >= CH3_2
class raw_ostream;
#endif

extern Target TheCpu0Target;
extern Target TheCpu0elTarget;

#if CH >= CH5_1
MCCodeEmitter *createCpu0MCCodeEmitterEB(const MCInstrInfo &MCII,
                                         const MCRegisterInfo &MRI,
                                         const MCSubtargetInfo &STI,
                                         MCContext &Ctx);
MCCodeEmitter *createCpu0MCCodeEmitterEL(const MCInstrInfo &MCII,
                                         const MCRegisterInfo &MRI,
                                         const MCSubtargetInfo &STI,
                                         MCContext &Ctx);

MCAsmBackend *createCpu0AsmBackendEB32(const Target &T, const MCRegisterInfo &MRI,
                                       StringRef TT, StringRef CPU);
MCAsmBackend *createCpu0AsmBackendEL32(const Target &T, const MCRegisterInfo &MRI,
                                       StringRef TT, StringRef CPU);

MCObjectWriter *createCpu0ELFObjectWriter(raw_ostream &OS,
                                          uint8_t OSABI,
                                          bool IsLittleEndian);
#endif
} // End llvm namespace

// Defines symbolic names for Cpu0 registers.  This defines a mapping from
// register name to register number.
#define GET_REGINFO_ENUM
#include "Cpu0GenRegisterInfo.inc"

// Defines symbolic names for the Cpu0 instructions.
#define GET_INSTRINFO_ENUM
#include "Cpu0GenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "Cpu0GenSubtargetInfo.inc"

#endif
