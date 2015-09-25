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

#include "Cpu0SelectionDAGInfo.h"
#if CH >= CH3_1

#include "Cpu0TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "cpu0-selectiondag-info"

Cpu0SelectionDAGInfo::Cpu0SelectionDAGInfo(const DataLayout &DL)
    : TargetSelectionDAGInfo(&DL) {}

Cpu0SelectionDAGInfo::~Cpu0SelectionDAGInfo() {
}

#endif
