/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Pseudo-instruction MC lowering Source Fragment                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

bool Cpu0AsmPrinter::
emitPseudoExpansionLowering(MCStreamer &OutStreamer,
                            const MachineInstr *MI) {
  switch (MI->getOpcode()) {
  default: return false;
  case Cpu0::TAILCALL: {
    MCInst TmpInst;
    MCOperand MCOp;
    TmpInst.setOpcode(Cpu0::JMP);
    // Operand: addr
    lowerOperand(MI->getOperand(0), MCOp);
    TmpInst.addOperand(MCOp);
    EmitToStreamer(OutStreamer, TmpInst);
    break;
  }
  case Cpu0::TAILCALL_R: {
    MCInst TmpInst;
    MCOperand MCOp;
    TmpInst.setOpcode(Cpu0::JR);
    // Operand: ra
    lowerOperand(MI->getOperand(0), MCOp);
    TmpInst.addOperand(MCOp);
    EmitToStreamer(OutStreamer, TmpInst);
    break;
  }
  }
  return true;
}

