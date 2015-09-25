//===-- Cpu0SEISelDAGToDAG.h - A Dag to Dag Inst Selector for Cpu0SE -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Subclass of Cpu0DAGToDAGISel specialized for mips32/64.
//
//===----------------------------------------------------------------------===//

#ifndef CPU0SEISELDAGTODAG_H
#define CPU0SEISELDAGTODAG_H

#include "Cpu0Config.h"
#if CH >= CH3_3

#include "Cpu0ISelDAGToDAG.h"

namespace llvm {

class Cpu0SEDAGToDAGISel : public Cpu0DAGToDAGISel {

public:
  explicit Cpu0SEDAGToDAGISel(Cpu0TargetMachine &TM) : Cpu0DAGToDAGISel(TM) {}

private:

  bool runOnMachineFunction(MachineFunction &MF) override;

  std::pair<bool, SDNode*> selectNode(SDNode *Node) override;

  void processFunctionAfterISel(MachineFunction &MF) override;

  // Insert instructions to initialize the global base register in the
  // first MBB of the function.
//  void initGlobalBaseReg(MachineFunction &MF);

#if CH >= CH4_1
  std::pair<SDNode*, SDNode*> SelectMULT(SDNode *N, unsigned Opc, SDLoc DL,
                                         EVT Ty, bool HasLo, bool HasHi);
#endif

#if CH >= CH7_1
  SDNode *selectAddESubE(unsigned MOp, SDValue InFlag, SDValue CmpLHS,
                         SDLoc DL, SDNode *Node) const;
#endif
};

FunctionPass *createCpu0SEISelDag(Cpu0TargetMachine &TM);

}

#endif // #if CH >= CH3_3

#endif
