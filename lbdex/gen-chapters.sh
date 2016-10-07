#!/usr/bin/env bash

# During run this .sh, it compile preprocess.cpp by:
# clang++ preprocess.cpp -o preprocess 

files="CMakeLists.txt Cpu0.h Cpu0AnalyzeImmediate.cpp Cpu0AnalyzeImmediate.h \
       Cpu0AsmPrinter.cpp Cpu0AsmPrinter.h Cpu0Config.h \
       Cpu0DelaySlotFiller.cpp Cpu0DelUselessJMP.cpp Cpu0EmitGPRestore.cpp \
       Cpu0FrameLowering.cpp Cpu0FrameLowering.h Cpu0InstrInfo.cpp \
       Cpu0InstrInfo.h Cpu0ISelDAGToDAG.cpp Cpu0ISelDAGToDAG.h \
       Cpu0ISelLowering.cpp Cpu0ISelLowering.h Cpu0LongBranch.cpp \
       Cpu0MachineFunction.cpp Cpu0MachineFunction.h \
       Cpu0MCInstLower.cpp Cpu0MCInstLower.h \
       Cpu0RegisterInfo.cpp Cpu0RegisterInfo.h \
       Cpu0SEFrameLowering.cpp Cpu0SEFrameLowering.h \
       Cpu0SEInstrInfo.cpp Cpu0SEInstrInfo.h \
       Cpu0SEISelDAGToDAG.cpp Cpu0SEISelDAGToDAG.h \
       Cpu0SEISelLowering.cpp Cpu0SEISelLowering.h \
       Cpu0SetChapter.h \
       Cpu0SERegisterInfo.cpp Cpu0SERegisterInfo.h \
       Cpu0Subtarget.cpp Cpu0Subtarget.h \
       Cpu0TargetMachine.cpp Cpu0TargetMachine.h \
       Cpu0TargetObjectFile.cpp Cpu0TargetObjectFile.h \
       Cpu0TargetStreamer.h \
       LLVMBuild.txt \
       AsmParser/Cpu0AsmParser.cpp \
       AsmParser/CMakeLists.txt AsmParser/LLVMBuild.txt \
       Disassembler/Cpu0Disassembler.cpp \
       Disassembler/CMakeLists.txt Disassembler/LLVMBuild.txt \
       InstPrinter/Cpu0InstPrinter.cpp InstPrinter/Cpu0InstPrinter.h \
       InstPrinter/CMakeLists.txt InstPrinter/LLVMBuild.txt \
       MCTargetDesc/Cpu0AsmBackend.cpp MCTargetDesc/Cpu0AsmBackend.h \
       MCTargetDesc/Cpu0BaseInfo.h MCTargetDesc/Cpu0ELFObjectWriter.cpp \
       MCTargetDesc/Cpu0FixupKinds.h \
       MCTargetDesc/Cpu0ABIInfo.cpp MCTargetDesc/Cpu0ABIInfo.h \
       MCTargetDesc/Cpu0MCAsmInfo.cpp MCTargetDesc/Cpu0MCAsmInfo.h \
       MCTargetDesc/Cpu0MCCodeEmitter.cpp MCTargetDesc/Cpu0MCCodeEmitter.h \
       MCTargetDesc/Cpu0MCExpr.cpp MCTargetDesc/Cpu0MCExpr.h \
       MCTargetDesc/Cpu0MCTargetDesc.cpp MCTargetDesc/Cpu0MCTargetDesc.h \
       MCTargetDesc/Cpu0TargetStreamer.cpp \
       MCTargetDesc/CMakeLists.txt MCTargetDesc/LLVMBuild.txt \
       TargetInfo/Cpu0TargetInfo.cpp \
       TargetInfo/CMakeLists.txt TargetInfo/LLVMBuild.txt \
       Cpu0.td Cpu0Asm.td Cpu0CallingConv.td Cpu0CondMov.td Cpu0CallingConv.td \
       Cpu0InstrFormats.td Cpu0InstrInfo.td Cpu0Other.td Cpu0RegisterInfo.td \
       Cpu0RegisterInfoGPROutForAsm.td Cpu0RegisterInfoGPROutForOther.td \
       Cpu0Schedule.td"


gen_Chapter()
{
  pushd $dest_dir
  mkdir AsmParser Disassembler InstPrinter MCTargetDesc TargetInfo
  popd

  # All other Chapters
  for file in $files
  do
    echo "./preprocess $src_dir/$file $dest_dir/$file CH$ch"
    ./preprocess $src_dir/$file $dest_dir/$file CH$ch
    if [ "$?" != "0" ]; then
      return 1;
    fi
  done
}

remove_files()
{
  rm -rf $dest_parent_dir/"Chapter2/AsmParser"
  rm -rf $dest_parent_dir/"Chapter2/Disassembler"
  rm -rf $dest_parent_dir/"Chapter2/InstPrinter"
  rm -f $dest_parent_dir/"Chapter2/MCTargetDesc/Cpu0ABIInfo.cpp"
  rm -f $dest_parent_dir/"Chapter2/MCTargetDesc/Cpu0ABIInfo.h"
  rm -f $dest_parent_dir/"Chapter2/MCTargetDesc/Cpu0AsmBackend.cpp"
  rm -f $dest_parent_dir/"Chapter2/MCTargetDesc/Cpu0AsmBackend.h"
  rm -f $dest_parent_dir/"Chapter2/MCTargetDesc/Cpu0BaseInfo.h"
  rm -f $dest_parent_dir/"Chapter2/MCTargetDesc/Cpu0ELFObjectWriter.cpp"
  rm -f $dest_parent_dir/"Chapter2/MCTargetDesc/Cpu0FixupKinds.h"
  rm -f $dest_parent_dir/"Chapter2/MCTargetDesc/Cpu0MCAsmInfo.cpp"
  rm -f $dest_parent_dir/"Chapter2/MCTargetDesc/Cpu0MCAsmInfo.h"
  rm -f $dest_parent_dir/"Chapter2/MCTargetDesc/Cpu0MCCodeEmitter.cpp"
  rm -f $dest_parent_dir/"Chapter2/MCTargetDesc/Cpu0MCCodeEmitter.h"
  rm -f $dest_parent_dir/"Chapter2/MCTargetDesc/Cpu0MCExpr.cpp"
  rm -f $dest_parent_dir/"Chapter2/MCTargetDesc/Cpu0MCExpr.h"
  rm -f $dest_parent_dir/"Chapter2/MCTargetDesc/Cpu0TargetStreamer.cpp"
  rm -f $dest_parent_dir/"Chapter2/Cpu0AnalyzeImmediate.cpp"
  rm -f $dest_parent_dir/"Chapter2/Cpu0AnalyzeImmediate.h"
  rm -f $dest_parent_dir/"Chapter2/Cpu0Asm.td"
  rm -f $dest_parent_dir/"Chapter2/Cpu0AsmPrinter.cpp"
  rm -f $dest_parent_dir/"Chapter2/Cpu0AsmPrinter.h"
  rm -f $dest_parent_dir/"Chapter2/Cpu0CallingConv.td"
  rm -f $dest_parent_dir/"Chapter2/Cpu0CondMov.td"
  rm -f $dest_parent_dir/"Chapter2/Cpu0DelaySlotFiller.cpp"
  rm -f $dest_parent_dir/"Chapter2/Cpu0DelUselessJMP.cpp"
  rm -f $dest_parent_dir/"Chapter2/Cpu0EmitGPRestore.cpp"
  rm -f $dest_parent_dir/"Chapter2/Cpu0FrameLowering.cpp"
  rm -f $dest_parent_dir/"Chapter2/Cpu0FrameLowering.h"
  rm -f $dest_parent_dir/"Chapter2/Cpu0InstrInfo.cpp"
  rm -f $dest_parent_dir/"Chapter2/Cpu0InstrInfo.h"
  rm -f $dest_parent_dir/"Chapter2/Cpu0ISelDAGToDAG.cpp"
  rm -f $dest_parent_dir/"Chapter2/Cpu0ISelDAGToDAG.h"
  rm -f $dest_parent_dir/"Chapter2/Cpu0ISelLowering.cpp"
  rm -f $dest_parent_dir/"Chapter2/Cpu0ISelLowering.h"
  rm -f $dest_parent_dir/"Chapter2/Cpu0LongBranch.cpp"
  rm -f $dest_parent_dir/"Chapter2/Cpu0MachineFunction.cpp"
  rm -f $dest_parent_dir/"Chapter2/Cpu0MachineFunction.h"
  rm -f $dest_parent_dir/"Chapter2/Cpu0MCInstLower.cpp"
  rm -f $dest_parent_dir/"Chapter2/Cpu0MCInstLower.h"
  rm -f $dest_parent_dir/"Chapter2/Cpu0RegisterInfo.cpp"
  rm -f $dest_parent_dir/"Chapter2/Cpu0RegisterInfo.h"
  rm -f $dest_parent_dir/"Chapter2/Cpu0RegisterInfoGPROutForAsm.td"
  rm -f $dest_parent_dir/"Chapter2/Cpu0SEFrameLowering.cpp"
  rm -f $dest_parent_dir/"Chapter2/Cpu0SEFrameLowering.h"
  rm -f $dest_parent_dir/"Chapter2/Cpu0SEInstrInfo.cpp"
  rm -f $dest_parent_dir/"Chapter2/Cpu0SEInstrInfo.h"
  rm -f $dest_parent_dir/"Chapter2/Cpu0SEISelDAGToDAG.cpp"
  rm -f $dest_parent_dir/"Chapter2/Cpu0SEISelDAGToDAG.h"
  rm -f $dest_parent_dir/"Chapter2/Cpu0SEISelLowering.cpp"
  rm -f $dest_parent_dir/"Chapter2/Cpu0SEISelLowering.h"
  rm -f $dest_parent_dir/"Chapter2/Cpu0SESelectionDAGInfo.cpp"
  rm -f $dest_parent_dir/"Chapter2/Cpu0SESelectionDAGInfo.h"
  rm -f $dest_parent_dir/"Chapter2/Cpu0SERegisterInfo.cpp"
  rm -f $dest_parent_dir/"Chapter2/Cpu0SERegisterInfo.h"
  rm -f $dest_parent_dir/"Chapter2/Cpu0Subtarget.cpp"
  rm -f $dest_parent_dir/"Chapter2/Cpu0Subtarget.h"
  rm -f $dest_parent_dir/"Chapter2/Cpu0TargetObjectFile.cpp"
  rm -f $dest_parent_dir/"Chapter2/Cpu0TargetObjectFile.h"
  rm -f $dest_parent_dir/"Chapter2/Cpu0TargetStreamer.h"

  rm -rf $dest_parent_dir/"Chapter3_1/AsmParser"
  rm -rf $dest_parent_dir/"Chapter3_1/Disassembler"
  rm -rf $dest_parent_dir/"Chapter3_1/InstPrinter"
  rm -f $dest_parent_dir/"Chapter3_1/MCTargetDesc/Cpu0AsmBackend.cpp"
  rm -f $dest_parent_dir/"Chapter3_1/MCTargetDesc/Cpu0AsmBackend.h"
  rm -f $dest_parent_dir/"Chapter3_1/MCTargetDesc/Cpu0BaseInfo.h"
  rm -f $dest_parent_dir/"Chapter3_1/MCTargetDesc/Cpu0ELFObjectWriter.cpp"
  rm -f $dest_parent_dir/"Chapter3_1/MCTargetDesc/Cpu0FixupKinds.h"
  rm -f $dest_parent_dir/"Chapter3_1/MCTargetDesc/Cpu0MCAsmInfo.cpp"
  rm -f $dest_parent_dir/"Chapter3_1/MCTargetDesc/Cpu0MCAsmInfo.h"
  rm -f $dest_parent_dir/"Chapter3_1/MCTargetDesc/Cpu0MCCodeEmitter.cpp"
  rm -f $dest_parent_dir/"Chapter3_1/MCTargetDesc/Cpu0MCCodeEmitter.h"
  rm -f $dest_parent_dir/"Chapter3_1/MCTargetDesc/Cpu0MCExpr.cpp"
  rm -f $dest_parent_dir/"Chapter3_1/MCTargetDesc/Cpu0MCExpr.h"
  rm -f $dest_parent_dir/"Chapter3_1/MCTargetDesc/Cpu0TargetStreamer.cpp"
  rm -f $dest_parent_dir/"Chapter3_1/Cpu0AnalyzeImmediate.cpp"
  rm -f $dest_parent_dir/"Chapter3_1/Cpu0AnalyzeImmediate.h"
  rm -f $dest_parent_dir/"Chapter3_1/Cpu0Asm.td"
  rm -f $dest_parent_dir/"Chapter3_1/Cpu0AsmPrinter.cpp"
  rm -f $dest_parent_dir/"Chapter3_1/Cpu0AsmPrinter.h"
  rm -f $dest_parent_dir/"Chapter3_1/Cpu0CondMov.td"
  rm -f $dest_parent_dir/"Chapter3_1/Cpu0DelaySlotFiller.cpp"
  rm -f $dest_parent_dir/"Chapter3_1/Cpu0DelUselessJMP.cpp"
  rm -f $dest_parent_dir/"Chapter3_1/Cpu0EmitGPRestore.cpp"
  rm -f $dest_parent_dir/"Chapter3_1/Cpu0ISelDAGToDAG.cpp"
  rm -f $dest_parent_dir/"Chapter3_1/Cpu0ISelDAGToDAG.h"
  rm -f $dest_parent_dir/"Chapter3_1/Cpu0LongBranch.cpp"
  rm -f $dest_parent_dir/"Chapter3_1/Cpu0MCInstLower.cpp"
  rm -f $dest_parent_dir/"Chapter3_1/Cpu0MCInstLower.h"
  rm -f $dest_parent_dir/"Chapter3_1/Cpu0RegisterInfoGPROutForAsm.td"
  rm -f $dest_parent_dir/"Chapter3_1/Cpu0SEISelDAGToDAG.cpp"
  rm -f $dest_parent_dir/"Chapter3_1/Cpu0SEISelDAGToDAG.h"
  rm -f $dest_parent_dir/"Chapter3_1/Cpu0TargetStreamer.h"

  rm -rf $dest_parent_dir/"Chapter3_2/AsmParser"
  rm -rf $dest_parent_dir/"Chapter3_2/Disassembler"
  rm -f $dest_parent_dir/"Chapter3_2/MCTargetDesc/Cpu0AsmBackend.cpp"
  rm -f $dest_parent_dir/"Chapter3_2/MCTargetDesc/Cpu0AsmBackend.h"
  rm -f $dest_parent_dir/"Chapter3_2/MCTargetDesc/Cpu0ELFObjectWriter.cpp"
  rm -f $dest_parent_dir/"Chapter3_2/MCTargetDesc/Cpu0FixupKinds.h"
  rm -f $dest_parent_dir/"Chapter3_2/MCTargetDesc/Cpu0MCCodeEmitter.cpp"
  rm -f $dest_parent_dir/"Chapter3_2/MCTargetDesc/Cpu0MCCodeEmitter.h"
  rm -f $dest_parent_dir/"Chapter3_2/MCTargetDesc/Cpu0MCExpr.cpp"
  rm -f $dest_parent_dir/"Chapter3_2/MCTargetDesc/Cpu0MCExpr.h"
  rm -f $dest_parent_dir/"Chapter3_2/MCTargetDesc/Cpu0TargetStreamer.cpp"
  rm -f $dest_parent_dir/"Chapter3_2/Cpu0AnalyzeImmediate.cpp"
  rm -f $dest_parent_dir/"Chapter3_2/Cpu0AnalyzeImmediate.h"
  rm -f $dest_parent_dir/"Chapter3_2/Cpu0Asm.td"
  rm -f $dest_parent_dir/"Chapter3_2/Cpu0CondMov.td"
  rm -f $dest_parent_dir/"Chapter3_2/Cpu0DelaySlotFiller.cpp"
  rm -f $dest_parent_dir/"Chapter3_2/Cpu0DelUselessJMP.cpp"
  rm -f $dest_parent_dir/"Chapter3_2/Cpu0EmitGPRestore.cpp"
  rm -f $dest_parent_dir/"Chapter3_2/Cpu0ISelDAGToDAG.cpp"
  rm -f $dest_parent_dir/"Chapter3_2/Cpu0ISelDAGToDAG.h"
  rm -f $dest_parent_dir/"Chapter3_2/Cpu0LongBranch.cpp"
  rm -f $dest_parent_dir/"Chapter2/Cpu0RegisterInfoGPROutForAsm.td"
  rm -f $dest_parent_dir/"Chapter3_2/Cpu0SEISelDAGToDAG.cpp"
  rm -f $dest_parent_dir/"Chapter3_2/Cpu0SEISelDAGToDAG.h"
  rm -f $dest_parent_dir/"Chapter3_2/Cpu0TargetStreamer.h"

  rm -rf $dest_parent_dir/"Chapter3_3/AsmParser"
  rm -rf $dest_parent_dir/"Chapter3_3/Disassembler"
  rm -f $dest_parent_dir/"Chapter3_3/MCTargetDesc/Cpu0AsmBackend.cpp"
  rm -f $dest_parent_dir/"Chapter3_3/MCTargetDesc/Cpu0AsmBackend.h"
  rm -f $dest_parent_dir/"Chapter3_3/MCTargetDesc/Cpu0ELFObjectWriter.cpp"
  rm -f $dest_parent_dir/"Chapter3_3/MCTargetDesc/Cpu0FixupKinds.h"
  rm -f $dest_parent_dir/"Chapter3_3/MCTargetDesc/Cpu0MCCodeEmitter.cpp"
  rm -f $dest_parent_dir/"Chapter3_3/MCTargetDesc/Cpu0MCCodeEmitter.h"
  rm -f $dest_parent_dir/"Chapter3_3/MCTargetDesc/Cpu0MCExpr.cpp"
  rm -f $dest_parent_dir/"Chapter3_3/MCTargetDesc/Cpu0MCExpr.h"
  rm -f $dest_parent_dir/"Chapter3_3/MCTargetDesc/Cpu0TargetStreamer.cpp"
  rm -f $dest_parent_dir/"Chapter3_3/Cpu0AnalyzeImmediate.cpp"
  rm -f $dest_parent_dir/"Chapter3_3/Cpu0AnalyzeImmediate.h"
  rm -f $dest_parent_dir/"Chapter3_3/Cpu0Asm.td"
  rm -f $dest_parent_dir/"Chapter3_3/Cpu0CondMov.td"
  rm -f $dest_parent_dir/"Chapter3_3/Cpu0DelaySlotFiller.cpp"
  rm -f $dest_parent_dir/"Chapter3_3/Cpu0DelUselessJMP.cpp"
  rm -f $dest_parent_dir/"Chapter3_3/Cpu0EmitGPRestore.cpp"
  rm -f $dest_parent_dir/"Chapter3_3/Cpu0LongBranch.cpp"
  rm -f $dest_parent_dir/"Chapter3_3/Cpu0RegisterInfoGPROutForAsm.td"
  rm -f $dest_parent_dir/"Chapter3_3/Cpu0TargetStreamer.h"

  rm -rf $dest_parent_dir/"Chapter3_4/AsmParser"
  rm -rf $dest_parent_dir/"Chapter3_4/Disassembler"
  rm -f $dest_parent_dir/"Chapter3_4/MCTargetDesc/Cpu0AsmBackend.cpp"
  rm -f $dest_parent_dir/"Chapter3_4/MCTargetDesc/Cpu0AsmBackend.h"
  rm -f $dest_parent_dir/"Chapter3_4/MCTargetDesc/Cpu0ELFObjectWriter.cpp"
  rm -f $dest_parent_dir/"Chapter3_4/MCTargetDesc/Cpu0FixupKinds.h"
  rm -f $dest_parent_dir/"Chapter3_4/MCTargetDesc/Cpu0MCCodeEmitter.cpp"
  rm -f $dest_parent_dir/"Chapter3_4/MCTargetDesc/Cpu0MCCodeEmitter.h"
  rm -f $dest_parent_dir/"Chapter3_4/MCTargetDesc/Cpu0MCExpr.cpp"
  rm -f $dest_parent_dir/"Chapter3_4/MCTargetDesc/Cpu0MCExpr.h"
  rm -f $dest_parent_dir/"Chapter3_4/MCTargetDesc/Cpu0TargetStreamer.cpp"
  rm -f $dest_parent_dir/"Chapter3_4/Cpu0AnalyzeImmediate.cpp"
  rm -f $dest_parent_dir/"Chapter3_4/Cpu0AnalyzeImmediate.h"
  rm -f $dest_parent_dir/"Chapter3_4/Cpu0Asm.td"
  rm -f $dest_parent_dir/"Chapter3_4/Cpu0CondMov.td"
  rm -f $dest_parent_dir/"Chapter3_4/Cpu0DelaySlotFiller.cpp"
  rm -f $dest_parent_dir/"Chapter3_4/Cpu0DelUselessJMP.cpp"
  rm -f $dest_parent_dir/"Chapter3_4/Cpu0EmitGPRestore.cpp"
  rm -f $dest_parent_dir/"Chapter3_4/Cpu0LongBranch.cpp"
  rm -f $dest_parent_dir/"Chapter3_4/Cpu0RegisterInfoGPROutForAsm.td"
  rm -f $dest_parent_dir/"Chapter3_4/Cpu0TargetStreamer.h"

  rm -rf $dest_parent_dir/"Chapter3_5/AsmParser"
  rm -rf $dest_parent_dir/"Chapter3_5/Disassembler"
  rm -f $dest_parent_dir/"Chapter3_5/MCTargetDesc/Cpu0AsmBackend.cpp"
  rm -f $dest_parent_dir/"Chapter3_5/MCTargetDesc/Cpu0AsmBackend.h"
  rm -f $dest_parent_dir/"Chapter3_5/MCTargetDesc/Cpu0ELFObjectWriter.cpp"
  rm -f $dest_parent_dir/"Chapter3_5/MCTargetDesc/Cpu0FixupKinds.h"
  rm -f $dest_parent_dir/"Chapter3_5/MCTargetDesc/Cpu0MCCodeEmitter.cpp"
  rm -f $dest_parent_dir/"Chapter3_5/MCTargetDesc/Cpu0MCCodeEmitter.h"
  rm -f $dest_parent_dir/"Chapter3_5/MCTargetDesc/Cpu0MCExpr.cpp"
  rm -f $dest_parent_dir/"Chapter3_5/MCTargetDesc/Cpu0MCExpr.h"
  rm -f $dest_parent_dir/"Chapter3_5/MCTargetDesc/Cpu0TargetStreamer.cpp"
  rm -f $dest_parent_dir/"Chapter3_5/Cpu0Asm.td"
  rm -f $dest_parent_dir/"Chapter3_5/Cpu0CondMov.td"
  rm -f $dest_parent_dir/"Chapter3_5/Cpu0DelaySlotFiller.cpp"
  rm -f $dest_parent_dir/"Chapter3_5/Cpu0DelUselessJMP.cpp"
  rm -f $dest_parent_dir/"Chapter3_5/Cpu0EmitGPRestore.cpp"
  rm -f $dest_parent_dir/"Chapter3_5/Cpu0LongBranch.cpp"
  rm -f $dest_parent_dir/"Chapter3_5/Cpu0RegisterInfoGPROutForAsm.td"
  rm -f $dest_parent_dir/"Chapter3_5/Cpu0TargetStreamer.h"

  rm -rf $dest_parent_dir/"Chapter4_1/AsmParser"
  rm -rf $dest_parent_dir/"Chapter4_1/Disassembler"
  rm -f $dest_parent_dir/"Chapter4_1/MCTargetDesc/Cpu0AsmBackend.cpp"
  rm -f $dest_parent_dir/"Chapter4_1/MCTargetDesc/Cpu0AsmBackend.h"
  rm -f $dest_parent_dir/"Chapter4_1/MCTargetDesc/Cpu0ELFObjectWriter.cpp"
  rm -f $dest_parent_dir/"Chapter4_1/MCTargetDesc/Cpu0FixupKinds.h"
  rm -f $dest_parent_dir/"Chapter4_1/MCTargetDesc/Cpu0MCCodeEmitter.cpp"
  rm -f $dest_parent_dir/"Chapter4_1/MCTargetDesc/Cpu0MCCodeEmitter.h"
  rm -f $dest_parent_dir/"Chapter4_1/MCTargetDesc/Cpu0MCExpr.cpp"
  rm -f $dest_parent_dir/"Chapter4_1/MCTargetDesc/Cpu0MCExpr.h"
  rm -f $dest_parent_dir/"Chapter4_1/MCTargetDesc/Cpu0TargetStreamer.cpp"
  rm -f $dest_parent_dir/"Chapter4_1/Cpu0Asm.td"
  rm -f $dest_parent_dir/"Chapter4_1/Cpu0CondMov.td"
  rm -f $dest_parent_dir/"Chapter4_1/Cpu0DelaySlotFiller.cpp"
  rm -f $dest_parent_dir/"Chapter4_1/Cpu0DelUselessJMP.cpp"
  rm -f $dest_parent_dir/"Chapter4_1/Cpu0EmitGPRestore.cpp"
  rm -f $dest_parent_dir/"Chapter4_1/Cpu0LongBranch.cpp"
  rm -f $dest_parent_dir/"Chapter4_1/Cpu0RegisterInfoGPROutForAsm.td"
  rm -f $dest_parent_dir/"Chapter4_1/Cpu0TargetStreamer.h"

  rm -rf $dest_parent_dir/"Chapter4_2/AsmParser"
  rm -rf $dest_parent_dir/"Chapter4_2/Disassembler"
  rm -f $dest_parent_dir/"Chapter4_2/MCTargetDesc/Cpu0AsmBackend.cpp"
  rm -f $dest_parent_dir/"Chapter4_2/MCTargetDesc/Cpu0AsmBackend.h"
  rm -f $dest_parent_dir/"Chapter4_2/MCTargetDesc/Cpu0ELFObjectWriter.cpp"
  rm -f $dest_parent_dir/"Chapter4_2/MCTargetDesc/Cpu0FixupKinds.h"
  rm -f $dest_parent_dir/"Chapter4_2/MCTargetDesc/Cpu0MCCodeEmitter.cpp"
  rm -f $dest_parent_dir/"Chapter4_2/MCTargetDesc/Cpu0MCCodeEmitter.h"
  rm -f $dest_parent_dir/"Chapter4_2/MCTargetDesc/Cpu0MCExpr.cpp"
  rm -f $dest_parent_dir/"Chapter4_2/MCTargetDesc/Cpu0MCExpr.h"
  rm -f $dest_parent_dir/"Chapter4_2/MCTargetDesc/Cpu0TargetStreamer.cpp"
  rm -f $dest_parent_dir/"Chapter4_2/Cpu0Asm.td"
  rm -f $dest_parent_dir/"Chapter4_2/Cpu0CondMov.td"
  rm -f $dest_parent_dir/"Chapter4_2/Cpu0DelaySlotFiller.cpp"
  rm -f $dest_parent_dir/"Chapter4_2/Cpu0DelUselessJMP.cpp"
  rm -f $dest_parent_dir/"Chapter4_2/Cpu0EmitGPRestore.cpp"
  rm -f $dest_parent_dir/"Chapter4_2/Cpu0LongBranch.cpp"
  rm -f $dest_parent_dir/"Chapter4_2/Cpu0RegisterInfoGPROutForAsm.td"
  rm -f $dest_parent_dir/"Chapter4_2/Cpu0TargetStreamer.h"

  rm -rf $dest_parent_dir/"Chapter5_1/AsmParser"
  rm -rf $dest_parent_dir/"Chapter5_1/Disassembler"
  rm -f $dest_parent_dir/"Chapter5_1/Cpu0Asm.td"
  rm -f $dest_parent_dir/"Chapter5_1/Cpu0CondMov.td"
  rm -f $dest_parent_dir/"Chapter5_1/Cpu0DelaySlotFiller.cpp"
  rm -f $dest_parent_dir/"Chapter5_1/Cpu0DelUselessJMP.cpp"
  rm -f $dest_parent_dir/"Chapter5_1/Cpu0EmitGPRestore.cpp"
  rm -f $dest_parent_dir/"Chapter5_1/Cpu0LongBranch.cpp"
  rm -f $dest_parent_dir/"Chapter5_1/Cpu0RegisterInfoGPROutForAsm.td"

  rm -rf $dest_parent_dir/"Chapter6_1/AsmParser"
  rm -rf $dest_parent_dir/"Chapter6_1/Disassembler"
  rm -f $dest_parent_dir/"Chapter6_1/Cpu0Asm.td"
  rm -f $dest_parent_dir/"Chapter6_1/Cpu0CondMov.td"
  rm -f $dest_parent_dir/"Chapter6_1/Cpu0DelaySlotFiller.cpp"
  rm -f $dest_parent_dir/"Chapter6_1/Cpu0DelUselessJMP.cpp"
  rm -f $dest_parent_dir/"Chapter6_1/Cpu0EmitGPRestore.cpp"
  rm -f $dest_parent_dir/"Chapter6_1/Cpu0LongBranch.cpp"
  rm -f $dest_parent_dir/"Chapter6_1/Cpu0RegisterInfoGPROutForAsm.td"

  rm -rf $dest_parent_dir/"Chapter7_1/AsmParser"
  rm -rf $dest_parent_dir/"Chapter7_1/Disassembler"
  rm -f $dest_parent_dir/"Chapter7_1/Cpu0Asm.td"
  rm -f $dest_parent_dir/"Chapter7_1/Cpu0CondMov.td"
  rm -f $dest_parent_dir/"Chapter7_1/Cpu0DelaySlotFiller.cpp"
  rm -f $dest_parent_dir/"Chapter7_1/Cpu0DelUselessJMP.cpp"
  rm -f $dest_parent_dir/"Chapter7_1/Cpu0EmitGPRestore.cpp"
  rm -f $dest_parent_dir/"Chapter7_1/Cpu0LongBranch.cpp"
  rm -f $dest_parent_dir/"Chapter7_1/Cpu0RegisterInfoGPROutForAsm.td"

  rm -rf $dest_parent_dir/"Chapter8_1/AsmParser"
  rm -rf $dest_parent_dir/"Chapter8_1/Disassembler"
  rm -f $dest_parent_dir/"Chapter8_1/Cpu0Asm.td"
  rm -f $dest_parent_dir/"Chapter8_1/Cpu0CondMov.td"
  rm -f $dest_parent_dir/"Chapter8_1/Cpu0DelaySlotFiller.cpp"
  rm -f $dest_parent_dir/"Chapter8_1/Cpu0DelUselessJMP.cpp"
  rm -f $dest_parent_dir/"Chapter8_1/Cpu0EmitGPRestore.cpp"
  rm -f $dest_parent_dir/"Chapter8_1/Cpu0LongBranch.cpp"
  rm -f $dest_parent_dir/"Chapter8_1/Cpu0RegisterInfoGPROutForAsm.td"

  rm -rf $dest_parent_dir/"Chapter8_2/AsmParser"
  rm -rf $dest_parent_dir/"Chapter8_2/Disassembler"
  rm -f $dest_parent_dir/"Chapter8_2/Cpu0Asm.td"
  rm -f $dest_parent_dir/"Chapter8_2/Cpu0EmitGPRestore.cpp"
  rm -f $dest_parent_dir/"Chapter8_2/Cpu0RegisterInfoGPROutForAsm.td"

  rm -rf $dest_parent_dir/"Chapter9_1/AsmParser"
  rm -rf $dest_parent_dir/"Chapter9_1/Disassembler"
  rm -f $dest_parent_dir/"Chapter9_1/Cpu0Asm.td"
  rm -f $dest_parent_dir/"Chapter9_1/Cpu0EmitGPRestore.cpp"
  rm -f $dest_parent_dir/"Chapter9_1/Cpu0RegisterInfoGPROutForAsm.td"

  rm -rf $dest_parent_dir/"Chapter9_2/AsmParser"
  rm -rf $dest_parent_dir/"Chapter9_2/Disassembler"
  rm -f $dest_parent_dir/"Chapter9_2/Cpu0Asm.td"
  rm -f $dest_parent_dir/"Chapter9_2/Cpu0EmitGPRestore.cpp"
  rm -f $dest_parent_dir/"Chapter9_2/Cpu0RegisterInfoGPROutForAsm.td"

  rm -rf $dest_parent_dir/"Chapter9_3/AsmParser"
  rm -rf $dest_parent_dir/"Chapter9_3/Disassembler"
  rm -f $dest_parent_dir/"Chapter9_3/Cpu0Asm.td"
  rm -f $dest_parent_dir/"Chapter9_3/Cpu0RegisterInfoGPROutForAsm.td"

  rm -rf $dest_parent_dir/"Chapter10_1/AsmParser"
  rm -f $dest_parent_dir/"Chapter10_1/Cpu0Asm.td"
  rm -f $dest_parent_dir/"Chapter10_1/Cpu0RegisterInfoGPROutForAsm.td"
}

src_dir=Cpu0
dest_parent_dir=chapters

rm -rf $dest_parent_dir
mkdir $dest_parent_dir

clang++ preprocess.cpp -o preprocess

allch="2 3_1 3_2 3_3 3_4 3_5 4_1 4_2 5_1 6_1 7_1 8_1 8_2 9_1 9_2 9_3 \
      10_1 11_1 11_2 12_1"
# All other Chapters
for ch in $allch
do
  dest_dir=$dest_parent_dir/"Chapter"$ch
  mkdir $dest_dir
  gen_Chapter;
  if [ "$?" != "0" ]; then
    exit 1;
  fi
done

remove_files;

