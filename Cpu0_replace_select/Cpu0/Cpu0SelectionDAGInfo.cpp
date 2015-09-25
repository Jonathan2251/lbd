//===-- Cpu0SelectionDAGInfo.cpp - Cpu0 SelectionDAG Info -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Cpu0SelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "cpu0-selectiondag-info"
#include "Cpu0TargetMachine.h"
using namespace llvm;

Cpu0SelectionDAGInfo::Cpu0SelectionDAGInfo(const Cpu0TargetMachine &TM)
  : TargetSelectionDAGInfo(TM) {
}

Cpu0SelectionDAGInfo::~Cpu0SelectionDAGInfo() {
}
