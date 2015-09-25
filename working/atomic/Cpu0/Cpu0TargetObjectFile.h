//===-- llvm/Target/Cpu0TargetObjectFile.h - Cpu0 Object Info ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_CPU0_TARGETOBJECTFILE_H
#define LLVM_TARGET_CPU0_TARGETOBJECTFILE_H

#include "Cpu0Config.h"
#if CH >= CH3_1

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"

namespace llvm {

  class Cpu0TargetObjectFile : public TargetLoweringObjectFileELF {
    const MCSection *SmallDataSection;
    const MCSection *SmallBSSSection;
    const TargetMachine *TM;
  public:

    void Initialize(MCContext &Ctx, const TargetMachine &TM) override;

#if CH >= CH6_1
    /// IsGlobalInSmallSection - Return true if this global address should be
    /// placed into small data/bss section.
    bool IsGlobalInSmallSection(const GlobalValue *GV,
                                const TargetMachine &TM, SectionKind Kind) const;
    bool IsGlobalInSmallSection(const GlobalValue *GV,
                                const TargetMachine &TM) const;

    const MCSection *SelectSectionForGlobal(const GlobalValue *GV,
                                        SectionKind Kind, Mangler &Mang,
                                        const TargetMachine &TM) const override;
#endif
  };
} // end namespace llvm

#endif // #if CH >= CH3_1

#endif
