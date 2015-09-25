//===-- Cpu0MachineFunctionInfo.cpp - Private data used for Cpu0 ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Cpu0MachineFunction.h"
#if CH >= CH3_1

#if CH >= CH3_2
#include "MCTargetDesc/Cpu0BaseInfo.h"
#endif
#include "Cpu0InstrInfo.h"
#include "Cpu0Subtarget.h"
#include "llvm/IR/Function.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

bool FixGlobalBaseReg;

// class Cpu0CallEntry.
Cpu0CallEntry::Cpu0CallEntry(const StringRef &N) {
#ifndef NDEBUG
  Name = N;
  Val = nullptr;
#endif
}

Cpu0CallEntry::Cpu0CallEntry(const GlobalValue *V) {
#ifndef NDEBUG
  Val = V;
#endif
}

bool Cpu0CallEntry::isConstant(const MachineFrameInfo *) const {
  return false;
}

bool Cpu0CallEntry::isAliased(const MachineFrameInfo *) const {
  return false;
}

bool Cpu0CallEntry::mayAlias(const MachineFrameInfo *) const {
  return false;
}

void Cpu0CallEntry::printCustom(raw_ostream &O) const {
  O << "Cpu0CallEntry: ";
#ifndef NDEBUG
  if (Val)
    O << Val->getName();
  else
    O << Name;
#endif
}

Cpu0FunctionInfo::~Cpu0FunctionInfo() {
  for (StringMap<const Cpu0CallEntry *>::iterator
       I = ExternalCallEntries.begin(), E = ExternalCallEntries.end(); I != E;
       ++I)
    delete I->getValue();

  for (const auto &Entry : GlobalCallEntries)
    delete Entry.second;
}

#if CH >= CH6_1
bool Cpu0FunctionInfo::globalBaseRegFixed() const {
  return FixGlobalBaseReg;
}

bool Cpu0FunctionInfo::globalBaseRegSet() const {
  return GlobalBaseReg;
}

unsigned Cpu0FunctionInfo::getGlobalBaseReg() {
  return GlobalBaseReg = Cpu0::GP;
}
#endif

#if CH >= CH9_2
MachinePointerInfo Cpu0FunctionInfo::callPtrInfo(const StringRef &Name) {
  const Cpu0CallEntry *&E = ExternalCallEntries[Name];

  if (!E)
    E = new Cpu0CallEntry(Name);

  return MachinePointerInfo(E);
}

MachinePointerInfo Cpu0FunctionInfo::callPtrInfo(const GlobalValue *Val) {
  const Cpu0CallEntry *&E = GlobalCallEntries[Val];

  if (!E)
    E = new Cpu0CallEntry(Val);

  return MachinePointerInfo(E);
}
#endif

void Cpu0FunctionInfo::anchor() { }

#endif // #if CH >= CH3_1
