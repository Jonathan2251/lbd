//===-- Cpu0SelectionDAGInfo.h - Cpu0 SelectionDAG Info ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Cpu0 subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef CPU0SELECTIONDAGINFO_H
#define CPU0SELECTIONDAGINFO_H

#include "Cpu0Config.h"
#if CH >= CH3_1

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class Cpu0TargetMachine;

class Cpu0SelectionDAGInfo : public TargetSelectionDAGInfo {
public:
  explicit Cpu0SelectionDAGInfo(const DataLayout &DL);
  ~Cpu0SelectionDAGInfo();
};

}

#endif // #if CH >= CH3_1

#endif
