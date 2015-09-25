//===-- Cpu0DelUselessJMP.cpp - Cpu0 DelJmp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "replace-select"

#include "Cpu0.h"
#include "Cpu0TargetMachine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Module.h"

using namespace llvm;

STATISTIC(NumReplaceSelect, "Number of SELECT_CC replaced");

static cl::opt<bool> EnableReplaceSelect(
  "enable-cpu0-replace-select",
  cl::init(true),
  cl::desc("Replace select IR instruction."),
  cl::Hidden);

namespace {
  struct ReplaceSelect : public MachineFunctionPass {

    TargetMachine &TM;
    const TargetInstrInfo *TII;

    static char ID;
    ReplaceSelect(TargetMachine &tm)
      : MachineFunctionPass(ID), TM(tm), TII(tm.getInstrInfo()) { }

    virtual const char *getPassName() const {
      return "Cpu0 replace select";
    }
#if 1
    bool runOnMachineBasicBlock(MachineBasicBlock &MBB);
    bool runOnMachineFunction(MachineFunction &F) {
      bool Changed = false;
      MachineFunction::iterator FI = F.begin();
      MachineFunction::iterator FE = F.end();
      if (FI == FE) {
        MachineBasicBlock* pMBB1 = F.CreateMachineBasicBlock();
      }
      else if (EnableReplaceSelect) {
        for (;
             FI != FE; ++FI) {
          // In STL style, F.end() is the dummy BasicBlock() like '\0' in 
          //  C string. 
          Changed |= runOnMachineBasicBlock(*FI);
        }
      }
      return Changed;
    }
#endif
#if 0
    bool runOnFunction(Function &F) {
      bool Changed = false;
/*      if (EnableReplaceSelect) {
        MachineFunction::iterator FJ = F.begin();
        if (FJ != F.end())
          FJ++;
        if (FJ == F.end())
          return Changed;
        for (Function::iterator FI = F.begin(), FE = F.end();
             FJ != FE; ++FI, ++FJ)
          // In STL style, F.end() is the dummy BasicBlock() like '\0' in 
          //  C string. 
          // FJ is the next BasicBlock of FI; When FI range from F.begin() to 
          //  the PreviousBasicBlock of F.end() call runOnMachineBasicBlock().
 //         Changed |= runOnMachineBasicBlock(*FI, *FJ);
          Changed |= 1;
      }*/
      return Changed;
    }
    bool runOnModule(Module &M) {
      DEBUG(errs() << "Run on Module MipsOs16\n");
      bool modified = false;
      for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
        if (F->isDeclaration()) continue;
        DEBUG(dbgs() << "Working on " << F->getName() << "\n");
        runOnFunction(*F);
      }
      return modified;
    }

#endif
  };
  char ReplaceSelect::ID = 0;
} // end of anonymous namespace
#if 1
bool ReplaceSelect::
runOnMachineBasicBlock(MachineBasicBlock &MBB) {
  bool Changed = false;

  for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); I++) {
    unsigned Opcode = I->getOpcode();
    if (Opcode == ISD::SELECT_CC) {
      DEBUG(outs() << "Got ISD::SELECT_CC\n");
    }
  }
#if 0
  MachineBasicBlock::iterator I = MBB.end();
  if (I != MBB.begin())
    I--;	// set I to the last instruction
  else
    return Changed;
    
  if (I->getOpcode() == Cpu0::JMP && I->getOperand(0).getMBB() == &MBBN) {
    // I is the instruction of "jmp #offset=0", as follows,
    //     jmp	$BB0_3
    // $BB0_3:
    //     ld	$4, 28($sp)
    ++NumReplaceSelect;
//    MBB.erase(I);	// delete the "JMP 0" instruction
    Changed = true;	// Notify LLVM kernel Changed
  }
#endif
  return Changed;

}
#endif
/// createCpu0ReplaceSelectPass - Returns a pass that ReplaceSelect in Cpu0 MachineFunctions
FunctionPass *llvm::createCpu0ReplaceSelectPass(Cpu0TargetMachine &tm) {
  return new ReplaceSelect(tm);
}
