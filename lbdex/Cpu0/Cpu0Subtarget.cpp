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

#if CH >= CH3_1
#include "Cpu0MachineFunction.h"
#include "Cpu0.h"
#include "Cpu0RegisterInfo.h"

#include "Cpu0TargetMachine.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "cpu0-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "Cpu0GenSubtargetInfo.inc"

#if CH >= CH4_1 //1
static cl::opt<bool> EnableOverflowOpt
                ("cpu0-enable-overflow", cl::Hidden, cl::init(false),
                 cl::desc("Use trigger overflow instructions add and sub \
                 instead of non-overflow instructions addu and subu"));
#endif

#if CH >= CH6_1 //1
static cl::opt<bool> UseSmallSectionOpt
                ("cpu0-use-small-section", cl::Hidden, cl::init(false),
                 cl::desc("Use small section. Only work when -relocation-model="
                 "static. pic always not use small section."));

static cl::opt<bool> ReserveGPOpt
                ("cpu0-reserve-gp", cl::Hidden, cl::init(false),
                 cl::desc("Never allocate $gp to variable"));

static cl::opt<bool> NoCploadOpt
                ("cpu0-no-cpload", cl::Hidden, cl::init(false),
                 cl::desc("No issue .cpload"));

bool Cpu0ReserveGP;
bool Cpu0NoCpload;
#endif

extern bool FixGlobalBaseReg;

/// Select the Cpu0 CPU for the given triple and cpu name.
/// FIXME: Merge with the copy in Cpu0MCTargetDesc.cpp
static StringRef selectCpu0CPU(Triple TT, StringRef CPU) {
  if (CPU.empty() || CPU == "generic") {
    if (TT.getArch() == Triple::cpu0 || TT.getArch() == Triple::cpu0el)
      CPU = "cpu032II";
  }
  return CPU;
}

void Cpu0Subtarget::anchor() { }

//@1 {
Cpu0Subtarget::Cpu0Subtarget(const Triple &TT, const std::string &CPU,
                             const std::string &FS, bool little, 
                             const Cpu0TargetMachine &_TM) :
//@1 }
  // Cpu0GenSubtargetInfo will display features by llc -march=cpu0 -mcpu=help
  Cpu0GenSubtargetInfo(TT, CPU, FS),
  IsLittle(little), TM(_TM), TargetTriple(TT), TSInfo(),
      InstrInfo(
          Cpu0InstrInfo::create(initializeSubtargetDependencies(CPU, FS, TM))),
      FrameLowering(Cpu0FrameLowering::create(*this)),
      TLInfo(Cpu0TargetLowering::create(TM, *this)) {

#if CH >= CH4_1 //2
  EnableOverflow = EnableOverflowOpt;
#endif
#if CH >= CH6_1 //2
  // Set UseSmallSection.
  UseSmallSection = UseSmallSectionOpt;
  Cpu0ReserveGP = ReserveGPOpt;
  Cpu0NoCpload = NoCploadOpt;
#ifdef ENABLE_GPRESTORE
  if (TM.getRelocationModel() == Reloc::Static == Reloc::Static && !UseSmallSection && !Cpu0ReserveGP)
    FixGlobalBaseReg = false;
  else
#endif
    FixGlobalBaseReg = true;
#endif //#if CH >= CH6_1
}

Cpu0Subtarget &
Cpu0Subtarget::initializeSubtargetDependencies(StringRef CPU, StringRef FS,
                                               const TargetMachine &TM) {
  std::string CPUName = selectCpu0CPU(TargetTriple, CPU);

  if (CPUName == "help")
    CPUName = "cpu032II";
  
  if (CPUName == "cpu032I")
    Cpu0ArchVersion = Cpu032I;
  else if (CPUName == "cpu032II")
    Cpu0ArchVersion = Cpu032II;

  if (isCpu032I()) {
    HasCmp = true;
    HasSlt = false;
  }
  else if (isCpu032II()) {
    HasCmp = true;
    HasSlt = true;
  }
  else {
    errs() << "-mcpu must be empty(default:cpu032II), cpu032I or cpu032II" << "\n";
  }

  // Parse features string.
  ParseSubtargetFeatures(CPUName, FS);
  // Initialize scheduling itinerary for the specified CPU.
  InstrItins = getInstrItineraryForCPU(CPUName);

  return *this;
}

bool Cpu0Subtarget::abiUsesSoftFloat() const {
//  return TM->Options.UseSoftFloat;
  return true;
}

#endif // #if CH >= CH3_1
