//===-- Cpu0TargetMachine.cpp - Define TargetMachine for Cpu0 -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the info about Cpu0 target spec.
//
//===----------------------------------------------------------------------===//

#include "Cpu0TargetMachine.h"
#include "Cpu0.h"
#if CH >= CH3_1
#include "Cpu0Subtarget.h"
#include "Cpu0TargetObjectFile.h"
#endif
#include "llvm/PassManager.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

#define DEBUG_TYPE "cpu0"

extern "C" void LLVMInitializeCpu0Target() {
#if CH >= CH3_1
  // Register the target.
  //- Big endian Target Machine
  RegisterTargetMachine<Cpu0ebTargetMachine> X(TheCpu0Target);
  //- Little endian Target Machine
  RegisterTargetMachine<Cpu0elTargetMachine> Y(TheCpu0elTarget);
#endif
}

#if CH >= CH3_1
// DataLayout --> Big-endian, 32-bit pointer/ABI/alignment
// The stack is always 8 byte aligned
// On function prologue, the stack is created by decrementing
// its pointer. Once decremented, all references are done with positive
// offset from the stack/frame pointer, using StackGrowsUp enables
// an easier handling.
// Using CodeModel::Large enables different CALL behavior.
Cpu0TargetMachine::
Cpu0TargetMachine(const Target &T, StringRef TT,
                  StringRef CPU, StringRef FS, const TargetOptions &Options,
                  Reloc::Model RM, CodeModel::Model CM,
                  CodeGenOpt::Level OL,
                  bool isLittle)
  //- Default is big endian
    : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
      isLittle(isLittle),
      TLOF(make_unique<Cpu0TargetObjectFile>()),
      Subtarget(nullptr), DefaultSubtarget(TT, CPU, FS, isLittle, RM, this) {
  Subtarget = &DefaultSubtarget;
  initAsmInfo();
}

Cpu0TargetMachine::~Cpu0TargetMachine() {}

void Cpu0ebTargetMachine::anchor() { }

Cpu0ebTargetMachine::
Cpu0ebTargetMachine(const Target &T, StringRef TT,
                    StringRef CPU, StringRef FS, const TargetOptions &Options,
                    Reloc::Model RM, CodeModel::Model CM,
                    CodeGenOpt::Level OL)
  : Cpu0TargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, false) {}

void Cpu0elTargetMachine::anchor() { }

Cpu0elTargetMachine::
Cpu0elTargetMachine(const Target &T, StringRef TT,
                    StringRef CPU, StringRef FS, const TargetOptions &Options,
                    Reloc::Model RM, CodeModel::Model CM,
                    CodeGenOpt::Level OL)
  : Cpu0TargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, true) {}

namespace {
//@Cpu0PassConfig {
/// Cpu0 Code Generator Pass Configuration Options.
class Cpu0PassConfig : public TargetPassConfig {
public:
  Cpu0PassConfig(Cpu0TargetMachine *TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {}

  Cpu0TargetMachine &getCpu0TargetMachine() const {
    return getTM<Cpu0TargetMachine>();
  }

  const Cpu0Subtarget &getCpu0Subtarget() const {
    return *getCpu0TargetMachine().getSubtargetImpl();
  }
#if CH >= CH12_1 //1
  void addIRPasses() override;
#endif
#if CH >= CH3_3 //1
  bool addInstSelector() override;
#endif
#if CH >= CH9_3 //1
#ifdef ENABLE_GPRESTORE
  void addPreRegAlloc() override;
#endif
#endif //#if CH >= CH9_3 //1
#if CH >= CH8_2 //1
  void addPreEmitPass() override;
#endif
};
} // namespace

TargetPassConfig *Cpu0TargetMachine::createPassConfig(PassManagerBase &PM) {
  return new Cpu0PassConfig(this, PM);
}

#if CH >= CH12_1 //2
void Cpu0PassConfig::addIRPasses() {
  TargetPassConfig::addIRPasses();
  addPass(createAtomicExpandPass(&getCpu0TargetMachine()));
}
#endif

#if CH >= CH3_3 //2
// Install an instruction selector pass using
// the ISelDag to gen Cpu0 code.
bool Cpu0PassConfig::addInstSelector() {
  addPass(createCpu0ISelDag(getCpu0TargetMachine()));
  return false;
}
#endif

#if CH >= CH9_3 //2
#ifdef ENABLE_GPRESTORE
void Cpu0PassConfig::addPreRegAlloc() {
  if (!Cpu0ReserveGP) {
    // $gp is a caller-saved register.
    addPass(createCpu0EmitGPRestorePass(getCpu0TargetMachine()));
  }
  return;
}
#endif
#endif //#if CH >= CH9_3 //2

#if CH >= CH8_2 //2
// Implemented by targets that want to run passes immediately before
// machine code is emitted. return true if -print-machineinstrs should
// print out the code after the passes.
void Cpu0PassConfig::addPreEmitPass() {
  Cpu0TargetMachine &TM = getCpu0TargetMachine();
//@8_2 1{
  addPass(createCpu0DelJmpPass(TM));
//@8_2 1}
  addPass(createCpu0DelaySlotFillerPass(TM));
//@8_2 2}
  return;
}
#endif

#endif // #if CH >= CH3_1
