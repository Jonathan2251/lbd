//===-- Cpu0MCTargetDesc.cpp - Cpu0 Target Descriptions -------------------===//
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

#include "Cpu0MCTargetDesc.h"
#if CH >= CH3_2 //1
#include "InstPrinter/Cpu0InstPrinter.h"
#include "Cpu0MCAsmInfo.h"
#endif
#if CH >= CH5_1
#include "Cpu0TargetStreamer.h"
#endif
#include "llvm/MC/MachineLocation.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define GET_INSTRINFO_MC_DESC
#include "Cpu0GenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "Cpu0GenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "Cpu0GenRegisterInfo.inc"

#if CH >= CH3_2 //2
//@1 {
/// Select the Cpu0 Architecture Feature for the given triple and cpu name.
/// The function will be called at command 'llvm-objdump -d' for Cpu0 elf input.
static StringRef selectCpu0ArchFeature(const Triple &TT, StringRef CPU) {
  std::string Cpu0ArchFeature;
  if (CPU.empty() || CPU == "generic") {
    if (TT.getArch() == Triple::cpu0 || TT.getArch() == Triple::cpu0el) {
      if (CPU.empty() || CPU == "cpu032II") {
        Cpu0ArchFeature = "+cpu032II";
      }
      else {
        if (CPU == "cpu032I") {
          Cpu0ArchFeature = "+cpu032I";
        }
      }
    }
  }
  return Cpu0ArchFeature;
}
//@1 }

static MCInstrInfo *createCpu0MCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitCpu0MCInstrInfo(X); // defined in Cpu0GenInstrInfo.inc
  return X;
}

static MCRegisterInfo *createCpu0MCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitCpu0MCRegisterInfo(X, Cpu0::SW); // defined in Cpu0GenRegisterInfo.inc
  return X;
}

static MCSubtargetInfo *createCpu0MCSubtargetInfo(const Triple &TT,
                                                  StringRef CPU, StringRef FS) {
  std::string ArchFS = selectCpu0ArchFeature(TT,CPU);
  if (!FS.empty()) {
    if (!ArchFS.empty())
      ArchFS = ArchFS + "," + FS.str();
    else
      ArchFS = FS;
  }
  return createCpu0MCSubtargetInfoImpl(TT, CPU, ArchFS);
// createCpu0MCSubtargetInfoImpl defined in Cpu0GenSubtargetInfo.inc
}

static MCAsmInfo *createCpu0MCAsmInfo(const MCRegisterInfo &MRI,
                                      const Triple &TT) {
  MCAsmInfo *MAI = new Cpu0MCAsmInfo(TT);

  unsigned SP = MRI.getDwarfRegNum(Cpu0::SP, true);
  MCCFIInstruction Inst = MCCFIInstruction::createDefCfa(nullptr, SP, 0);
  MAI->addInitialFrameState(Inst);

  return MAI;
}

static MCCodeGenInfo *createCpu0MCCodeGenInfo(const Triple &TT, Reloc::Model RM,
                                              CodeModel::Model CM,
                                              CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  if (CM == CodeModel::JITDefault)
    RM = Reloc::Static;
  else if (RM == Reloc::Default)
    RM = Reloc::PIC_;
  X->initMCCodeGenInfo(RM, CM, OL); // defined in lib/MC/MCCodeGenInfo.cpp
  return X;
}

static MCInstPrinter *createCpu0MCInstPrinter(const Triple &T,
                                              unsigned SyntaxVariant,
                                              const MCAsmInfo &MAI,
                                              const MCInstrInfo &MII,
                                              const MCRegisterInfo &MRI) {
  return new Cpu0InstPrinter(MAI, MII, MRI);
}
#endif

#if CH >= CH5_1 //1
static MCStreamer *createMCStreamer(const Triple &TT, MCContext &Context, 
                                    MCAsmBackend &MAB, raw_pwrite_stream &OS, 
                                    MCCodeEmitter *Emitter, bool RelaxAll) {
  return createELFStreamer(Context, MAB, OS, Emitter, RelaxAll);
}

static MCTargetStreamer *createCpu0AsmTargetStreamer(MCStreamer &S,
                                                     formatted_raw_ostream &OS,
                                                     MCInstPrinter *InstPrint,
                                                     bool isVerboseAsm) {
  return new Cpu0TargetAsmStreamer(S, OS);
}
#endif

//@2 {
extern "C" void LLVMInitializeCpu0TargetMC() {
#if CH >= CH3_2 //3
  for (Target *T : {&TheCpu0Target, &TheCpu0elTarget}) {
    // Register the MC asm info.
    RegisterMCAsmInfoFn X(*T, createCpu0MCAsmInfo);

    // Register the MC codegen info.
    TargetRegistry::RegisterMCCodeGenInfo(*T,
	                                      createCpu0MCCodeGenInfo);

    // Register the MC instruction info.
    TargetRegistry::RegisterMCInstrInfo(*T, createCpu0MCInstrInfo);

    // Register the MC register info.
    TargetRegistry::RegisterMCRegInfo(*T, createCpu0MCRegisterInfo);

#if CH >= CH5_1 //2
     // Register the elf streamer.
    TargetRegistry::RegisterELFStreamer(*T, createMCStreamer);

    // Register the asm target streamer.
    TargetRegistry::RegisterAsmTargetStreamer(*T, createCpu0AsmTargetStreamer);
#endif

    // Register the MC subtarget info.
    TargetRegistry::RegisterMCSubtargetInfo(*T,
	                                        createCpu0MCSubtargetInfo);
    // Register the MCInstPrinter.
    TargetRegistry::RegisterMCInstPrinter(*T,
	                                      createCpu0MCInstPrinter);
  }
#endif // #if CH >= CH3_2

#if CH >= CH5_1 //3
  // Register the MC Code Emitter
  TargetRegistry::RegisterMCCodeEmitter(TheCpu0Target,
                                        createCpu0MCCodeEmitterEB);
  TargetRegistry::RegisterMCCodeEmitter(TheCpu0elTarget,
                                        createCpu0MCCodeEmitterEL);

  // Register the asm backend.
  TargetRegistry::RegisterMCAsmBackend(TheCpu0Target,
                                       createCpu0AsmBackendEB32);
  TargetRegistry::RegisterMCAsmBackend(TheCpu0elTarget,
                                       createCpu0AsmBackendEL32);
#endif
}
//@2 }
