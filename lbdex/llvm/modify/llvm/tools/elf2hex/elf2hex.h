//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_ELF2HEX_ELF2HEX_H
#define LLVM_TOOLS_ELF2HEX_ELF2HEX_H

#include "llvm/DebugInfo/DIContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Object/Archive.h"

#include <stdio.h>
#include "llvm/Support/raw_ostream.h"

#define BOOT_SIZE 16

#define DLINK
//#define ELF2HEX_DEBUG

namespace llvm {

namespace elf2hex {

using namespace object;

class HexOut {
public:
  virtual void ProcessDisAsmInstruction(MCInst inst, uint64_t Size, 
                                ArrayRef<uint8_t> Bytes, const ObjectFile *Obj) = 0;
  virtual void ProcessDataSection(SectionRef Section) {};
  virtual ~HexOut() {};
};

// Split HexOut from Reader::DisassembleObject() for separating hex output 
// functions.
class VerilogHex : public HexOut {
public:
  VerilogHex(std::unique_ptr<MCInstPrinter>& instructionPointer, 
             std::unique_ptr<const MCSubtargetInfo>& subTargetInfo,
             const ObjectFile *Obj);
  void ProcessDisAsmInstruction(MCInst inst, uint64_t Size, 
                                ArrayRef<uint8_t> Bytes, const ObjectFile *Obj) override;
  void ProcessDataSection(SectionRef Section) override;

private:
  void PrintBootSection(uint64_t textOffset, uint64_t isrAddr, bool isLittleEndian);
  void Fill0s(uint64_t startAddr, uint64_t endAddr);
  void PrintDataSection(SectionRef Section);
  std::unique_ptr<MCInstPrinter>& IP;
  std::unique_ptr<const MCSubtargetInfo>& STI;
  uint64_t lastDumpAddr;
  unsigned si;
  StringRef sectionName;
};

class Reader {
public:
  void DisassembleObject(const ObjectFile *Obj, 
                         std::unique_ptr<MCDisassembler>& DisAsm, 
                         std::unique_ptr<MCInstPrinter>& IP, 
                         std::unique_ptr<const MCSubtargetInfo>& STI);
  StringRef CurrentSymbol();
  SectionRef CurrentSection();
  unsigned CurrentSi();
  uint64_t CurrentIndex();

private:
  SectionRef _section;
  std::vector<std::pair<uint64_t, StringRef> > Symbols;
  unsigned si;
  uint64_t Index;
};

} // end namespace elf2hex
} // end namespace llvm

//using namespace llvm;

#endif
