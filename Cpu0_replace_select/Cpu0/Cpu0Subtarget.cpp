//===-- Cpu0Subtarget.cpp - Cpu0 Subtarget Information --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Cpu0 specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "Cpu0Subtarget.h"
#include "Cpu0.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "Cpu0GenSubtargetInfo.inc"

using namespace llvm;

static cl::opt<bool>
UseSmallSectionOpt("cpu0-use-small-section", cl::Hidden, cl::init(false),
                 cl::desc("Use small section. Only work when -relocation-model="
                 "static. pic always not use small section."));

extern bool FixGlobalBaseReg;

void Cpu0Subtarget::anchor() { }

Cpu0Subtarget::Cpu0Subtarget(const std::string &TT, const std::string &CPU,
                             const std::string &FS, bool little, 
                             Reloc::Model _RM) :
  Cpu0GenSubtargetInfo(TT, CPU, FS),
  Cpu0ABI(UnknownABI), IsLittle(little), RM(_RM)
{
  std::string CPUName = CPU;
  if (CPUName.empty())
    CPUName = "cpu032";

  // Parse features string.
  ParseSubtargetFeatures(CPUName, FS);

  // Initialize scheduling itinerary for the specified CPU.
  InstrItins = getInstrItineraryForCPU(CPUName);

  // Set Cpu0ABI if it hasn't been set yet.
  if (Cpu0ABI == UnknownABI)
    Cpu0ABI = O32;

  // Set UseSmallSection.
  UseSmallSection = UseSmallSectionOpt;
  if (RM == Reloc::Static && !UseSmallSection)
    FixGlobalBaseReg = false;
  else
    FixGlobalBaseReg = true;
}

