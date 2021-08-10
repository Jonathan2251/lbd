//===-- llvm-objdump.cpp - Object file dumping utility for llvm -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program is a utility that works like binutils "objdump", that is, it
// dumps out a plethora of information about an object file depending on the
// flags.
//
// The flags and output of this program should be near identical to those of
// binutils objdump.
//
//===----------------------------------------------------------------------===//

#define ELF2HEX

#include "elf2hex.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Object/MachO.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;
using namespace llvm::object;

static StringRef ToolName;
static StringRef CurrInputFile;

// copy from llvm-objdump.cpp
LLVM_ATTRIBUTE_NORETURN void reportError(StringRef File,
                                                  const Twine &Message) {
  outs().flush();
  WithColor::error(errs(), ToolName) << "'" << File << "': " << Message << "\n";
  exit(1);
}

// copy from llvm-objdump.h
template <typename T, typename... Ts>
T unwrapOrError(Expected<T> EO, Ts &&... Args) {
  if (EO)
    return std::move(*EO);
  assert(0 && "error in unwrapOrError()");
}

// copy from llvm-objdump.cpp
static cl::OptionCategory Elf2hexCat("elf2hex Options");

static cl::list<std::string> InputFilenames(cl::Positional,
                                            cl::desc("<input object files>"),
                                            cl::ZeroOrMore,
                                            cl::cat(Elf2hexCat));
std::string TripleName = "";
                                            
static const Target *getTarget(const ObjectFile *Obj) {
  // Figure out the target triple.
  Triple TheTriple("unknown-unknown-unknown");
  TheTriple = Obj->makeTriple();

  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget("", TheTriple,
                                                         Error);
  if (!TheTarget)
    reportError(Obj->getFileName(), "can't find target: " + Error);

  // Update the triple name and return the found target.
  TripleName = TheTriple.getTriple();
  return TheTarget;
}

bool isRelocAddressLess(RelocationRef A, RelocationRef B) {
  return A.getOffset() < B.getOffset();
}

void error(std::error_code EC) {
  if (!EC)
    return;
  WithColor::error(errs(), ToolName)
      << "reading file: " << EC.message() << ".\n";
  errs().flush();
  exit(1);
}

static void getName(llvm::object::SectionRef const &Section, StringRef Name) {
  Name = unwrapOrError(Section.getName(), CurrInputFile);
#ifdef ELF2HEX_DEBUG
  llvm::dbgs() << Name << "\n";
#endif
}


static cl::opt<bool>
LittleEndian("le", 
cl::desc("Little endian format"));

#ifdef ELF2HEX_DEBUG
// Modified from PrintSectionHeaders()
uint64_t GetSectionHeaderStartAddress(const ObjectFile *Obj, 
  StringRef sectionName) {
//  outs() << "Sections:\n"
//            "Idx Name          Size      Address          Type\n";
  std::error_code ec;
  unsigned i = 0;
  for (const SectionRef &Section : Obj->sections()) {
    error(ec);
    StringRef Name;
    error(getName(Section, Name));
    uint64_t Address;
    Address = Section.getAddress();
    uint64_t Size;
    Size = Section.getSize();
    bool Text;
    Text = Section.isText();
    if (Name == sectionName)
      return Address;
    else
      return 0;
    ++i;
  }
  return 0;
}
#endif

// Reference from llvm::printSymbolTable of llvm-objdump.cpp
uint64_t GetSymbolAddress(const ObjectFile *o, StringRef SymbolName) {
  for (const SymbolRef &Symbol : o->symbols()) {
    Expected<uint64_t> AddressOrError = Symbol.getAddress();
    if (!AddressOrError)
      reportError(o->getFileName(), SymbolName);
    uint64_t Address = *AddressOrError;
    Expected<SymbolRef::Type> TypeOrError = Symbol.getType();
    if (!TypeOrError)
      reportError(o->getFileName(), SymbolName);
    SymbolRef::Type Type = *TypeOrError;
    section_iterator Section = unwrapOrError(Symbol.getSection(), CurrInputFile);
    StringRef Name;
    if (Type == SymbolRef::ST_Debug && Section != o->section_end()) {
      if (Expected<StringRef> NameOrErr = Section->getName())
        Name = *NameOrErr;
      else
        consumeError(NameOrErr.takeError());
    } else {
      Name = unwrapOrError(Symbol.getName(), o->getFileName());
    }
    if (Name == SymbolName)
      return Address;
  }
  return 0;
}

uint64_t SectionOffset(const ObjectFile *o, StringRef secName) {
  for (const SectionRef &Section : o->sections()) {
    StringRef Name;
    uint64_t BaseAddr;
    Name = unwrapOrError(Section.getName(), o->getFileName());
    unwrapOrError(Section.getContents(), o->getFileName());
    BaseAddr = Section.getAddress();

    if (Name == secName)
      return BaseAddr;
  }
  return 0;
}

using namespace llvm::elf2hex;

Reader reader;

VerilogHex::VerilogHex(std::unique_ptr<MCInstPrinter>& instructionPointer, 
  std::unique_ptr<const MCSubtargetInfo>& subTargetInfo, const ObjectFile *Obj) :
  IP(instructionPointer), STI(subTargetInfo) {
  lastDumpAddr = 0;
#ifdef ELF2HEX_DEBUG
  //uint64_t startAddr = GetSectionHeaderStartAddress(Obj, "_start");
  //errs() << format("_start address:%08" PRIx64 "\n", startAddr);
#endif
  uint64_t isrAddr = GetSymbolAddress(Obj, "ISR");
  errs() << format("ISR address:%08" PRIx64 "\n", isrAddr);

  //uint64_t pltOffset = SectionOffset(Obj, ".plt");
  uint64_t textOffset = SectionOffset(Obj, ".text");
  PrintBootSection(textOffset, isrAddr, LittleEndian);
  lastDumpAddr = BOOT_SIZE;
  Fill0s(lastDumpAddr, 0x100);
  lastDumpAddr = 0x100;
}

void VerilogHex::PrintBootSection(uint64_t textOffset, uint64_t isrAddr, 
                                  bool isLittleEndian) {
  uint64_t offset = textOffset - 4;

  // isr instruction at 0x8 and PC counter point to next instruction
  uint64_t isrOffset = isrAddr - 8 - 4;
  if (isLittleEndian) {
    outs() << "/*       0:*/	";
    outs() << format("%02" PRIx64 " ", (offset & 0xff));
    outs() << format("%02" PRIx64 " ", (offset & 0xff00) >> 8);
    outs() << format("%02" PRIx64 "", (offset & 0xff0000) >> 16);
    outs() << " 36";
    outs() << "                                  /*	jmp	0x";
    outs() << format("%02" PRIx64 "%02" PRIx64 "%02" PRIx64 " */\n",
      (offset & 0xff0000) >> 16, (offset & 0xff00) >> 8, (offset & 0xff));
    outs() <<
      "/*       4:*/	04 00 00 36                                  /*	jmp	4 */\n";
    offset -= 8;
    outs() << "/*       8:*/	";
    outs() << format("%02" PRIx64 " ", (isrOffset & 0xff));
    outs() << format("%02" PRIx64 " ", (isrOffset & 0xff00) >> 8);
    outs() << format("%02" PRIx64 "", (isrOffset & 0xff0000) >> 16);
    outs() << " 36";
    outs() << "                                  /*	jmp	0x";
    outs() << format("%02" PRIx64 "%02" PRIx64 "%02" PRIx64 " */\n",
      (isrOffset & 0xff0000) >> 16, (isrOffset & 0xff00) >> 8, (isrOffset & 0xff));
    outs() <<
      "/*       c:*/	fc ff ff 36                                  /*	jmp	-4 */\n";
  }
  else {
    outs() << "/*       0:*/	36 ";
    outs() << format("%02" PRIx64 " ", (offset & 0xff0000) >> 16);
    outs() << format("%02" PRIx64 " ", (offset & 0xff00) >> 8);
    outs() << format("%02" PRIx64 "", (offset & 0xff));
    outs() << "                                  /*	jmp	0x";
    outs() << format("%02" PRIx64 "%02" PRIx64 "%02" PRIx64 " */\n",
      (offset & 0xff0000) >> 16, (offset & 0xff00) >> 8, (offset & 0xff));
    outs() <<
      "/*       4:*/	36 00 00 04                                  /*	jmp	4 */\n";
    offset -= 8;
    outs() << "/*       8:*/	36 ";
    outs() << format("%02" PRIx64 " ", (isrOffset & 0xff0000) >> 16);
    outs() << format("%02" PRIx64 " ", (isrOffset & 0xff00) >> 8);
    outs() << format("%02" PRIx64 "", (isrOffset & 0xff));
    outs() << "                                  /*	jmp	0x";
    outs() << format("%02" PRIx64 "%02" PRIx64 "%02" PRIx64 " */\n",
      (isrOffset & 0xff0000) >> 16, (isrOffset & 0xff00) >> 8, (isrOffset & 0xff));
    outs() <<
      "/*       c:*/	36 ff ff fc                                  /*	jmp	-4 */\n";
  }
}

// Fill /*address*/ 00 00 00 00 [startAddr..endAddr] from startAddr to endAddr. 
// Include startAddr and endAddr.
void VerilogHex::Fill0s(uint64_t startAddr, uint64_t endAddr) {
  std::size_t addr;

  assert((startAddr <= endAddr) && "startAddr must <= BaseAddr");
  // Fill /*address*/ 00 00 00 00 for 4 bytes alignment (1 Cpu0 word size)
  for (addr = startAddr; addr < endAddr; addr += 4) {
    outs() << format("/*%8" PRIx64 " */", addr);
    outs() << format("%02" PRIx64 " ", 0) << format("%02" PRIx64 " ", 0) \
    << format("%02" PRIx64 " ", 0) << format("%02" PRIx64 " ", 0) << '\n';
  }

  return;
}

void VerilogHex::ProcessDisAsmInstruction(MCInst inst, uint64_t Size, 
                                ArrayRef<uint8_t> Bytes, const ObjectFile *Obj) {
  SectionRef Section = reader.CurrentSection();
  StringRef Name;
  StringRef Contents;
  Name = unwrapOrError(Section.getName(), Obj->getFileName());
  unwrapOrError(Section.getContents(), Obj->getFileName());
  uint64_t SectionAddr = Section.getAddress();
  uint64_t Index = reader.CurrentIndex();
#ifdef ELF2HEX_DEBUG
  errs() << format("SectionAddr + Index = %8" PRIx64 "\n", SectionAddr + Index);
  errs() << format("lastDumpAddr %8" PRIx64 "\n", lastDumpAddr);
#endif
  if (lastDumpAddr < SectionAddr) {
    Fill0s(lastDumpAddr, SectionAddr - 1);
    lastDumpAddr = SectionAddr;
  }

  // print section name when meeting it first time
  if (sectionName != Name) {
    StringRef SegmentName = "";
    if (const MachOObjectFile *MachO =
        dyn_cast<const MachOObjectFile>(Obj)) {
      DataRefImpl DR = Section.getRawDataRefImpl();
      SegmentName = MachO->getSectionFinalSegmentName(DR);
    }
    outs() << "/*" << "Disassembly of section ";
    if (!SegmentName.empty())
      outs() << SegmentName << ",";
    outs() << Name << ':' << "*/";
    sectionName = Name;
  }

  if (si != reader.CurrentSi()) {
    // print function name in section .text just before the first instruction 
    // is printed
    outs() << '\n' << "/*" << reader.CurrentSymbol() << ":*/\n";
    si = reader.CurrentSi();
  }

  // print instruction address
  outs() << format("/*%8" PRIx64 ":*/", SectionAddr + Index);
 
  // print instruction in hex format
  outs() << "\t";
  dumpBytes(Bytes.slice(Index, Size), outs());

  outs() << "/*";
  // print disassembly instruction to outs()
  IP->printInst(&inst, 0, "", *STI, outs());
  outs() << "*/";
  outs() << "\n";

  // In section .plt or .text, the Contents.size() maybe < (SectionAddr + Index + 4)
  if (Contents.size() < (SectionAddr + Index + 4))
    lastDumpAddr = SectionAddr + Index + 4;
  else
    lastDumpAddr = SectionAddr + Contents.size();
}

void VerilogHex::ProcessDataSection(SectionRef Section) {
  std::string Error;
  StringRef Name;
  StringRef Contents;
  uint64_t BaseAddr;
  uint64_t size;
  getName(Section, Name);
  unwrapOrError(Section.getContents(), CurrInputFile);
  BaseAddr = Section.getAddress();

#ifdef ELF2HEX_DEBUG
  errs() << format("BaseAddr = %8" PRIx64 "\n", BaseAddr);
  errs() << format("lastDumpAddr %8" PRIx64 "\n", lastDumpAddr);
#endif
  if (lastDumpAddr < BaseAddr) {
    Fill0s(lastDumpAddr, BaseAddr - 1);
    lastDumpAddr = BaseAddr;
  }
  if ((Name == ".bss" || Name == ".sbss") && Contents.size() > 0) {
    size = (Contents.size() + 3)/4*4;
    Fill0s(BaseAddr, BaseAddr + size - 1);
    lastDumpAddr = BaseAddr + size;
    return;
  }
  else {
    PrintDataSection(Section);
  }
}

void VerilogHex::PrintDataSection(SectionRef Section) {
  std::string Error;
  StringRef Name;
  uint64_t BaseAddr;
  uint64_t size;
  getName(Section, Name);
  StringRef Contents = unwrapOrError(Section.getContents(), CurrInputFile);
  BaseAddr = Section.getAddress();

  if (Contents.size() <= 0) {
    return;
  }
  size = (Contents.size()+3)/4*4;

  outs() << "/*Contents of section " << Name << ":*/\n";
  // Dump out the content as hex and printable ascii characters.
  for (std::size_t addr = 0, end = Contents.size(); addr < end; addr += 16) {
    outs() << format("/*%8" PRIx64 " */", BaseAddr + addr);
    // Dump line of hex.
    for (std::size_t i = 0; i < 16; ++i) {
      if (i != 0 && i % 4 == 0)
        outs() << ' ';
      if (addr + i < end)
        outs() << hexdigit((Contents[addr + i] >> 4) & 0xF, true)
               << hexdigit(Contents[addr + i] & 0xF, true) << " ";
    }
    // Print ascii.
    outs() << "/*" << "  ";
    for (std::size_t i = 0; i < 16 && addr + i < end; ++i) {
      if (std::isprint(static_cast<unsigned char>(Contents[addr + i]) & 0xFF))
        outs() << Contents[addr + i];
      else
        outs() << ".";
    }
    outs() << "*/" << "\n";
  }
  for (std::size_t i = Contents.size(); i < size; i++) {
    outs() << "00 ";
  }
  outs() << "\n";
#ifdef ELF2HEX_DEBUG
  errs() << "Name " << Name << "  BaseAddr ";
  errs() << format("%8" PRIx64 " Contents.size() ", BaseAddr);
  errs() << format("%8" PRIx64 " size ", Contents.size());
  errs() << format("%8" PRIx64 " \n", size);
#endif
  // save the end address of this section to lastDumpAddr
  lastDumpAddr = BaseAddr + size;
}

StringRef Reader::CurrentSymbol() {
  return Symbols[si].second;
}

SectionRef Reader::CurrentSection() {
  return _section;
}

unsigned Reader::CurrentSi() {
  return si;
}

uint64_t Reader::CurrentIndex() {
  return Index;
}

// Porting from DisassembleObject() of llvm-objump.cpp
void Reader::DisassembleObject(const ObjectFile *Obj
/*, bool InlineRelocs*/  , std::unique_ptr<MCDisassembler>& DisAsm, 
  std::unique_ptr<MCInstPrinter>& IP,
  std::unique_ptr<const MCSubtargetInfo>& STI) {
  VerilogHex hexOut(IP, STI, Obj);
  std::error_code ec;
  for (const SectionRef &Section : Obj->sections()) {
    _section = Section;
    uint64_t BaseAddr;
    unwrapOrError(Section.getContents(), Obj->getFileName());
    BaseAddr = Section.getAddress();
    uint64_t SectSize = Section.getSize();
    if (!SectSize)
      continue;

    if (BaseAddr < 0x100)
      continue;
 
  #ifdef ELF2HEX_DEBUG
    StringRef SectionName = unwrapOrError(Section.getName(), Obj->getFileName());
    errs() << "SectionName " << SectionName << format("  BaseAddr %8" PRIx64 "\n", BaseAddr);
  #endif
 
    bool text;
    text = Section.isText();
    if (!text) {
      hexOut.ProcessDataSection(Section);
      continue;
    }
    // It's .text section
    uint64_t SectionAddr;
    SectionAddr = Section.getAddress();
 
    // Make a list of all the symbols in this section.
    for (const SymbolRef &Symbol : Obj->symbols()) {
      if (Section.containsSymbol(Symbol)) {
        Expected<uint64_t> AddressOrErr = Symbol.getAddress();
        error(errorToErrorCode(AddressOrErr.takeError()));
        uint64_t Address = *AddressOrErr;
        Address -= SectionAddr;
        if (Address >= SectSize)
          continue;

        Expected<StringRef> Name = Symbol.getName();
        error(errorToErrorCode(Name.takeError()));
        Symbols.push_back(std::make_pair(Address, *Name));
      }
    }

    // Sort the symbols by address, just in case they didn't come in that way.
    array_pod_sort(Symbols.begin(), Symbols.end());
  #ifdef ELF2HEX_DEBUG
    for (unsigned si = 0, se = Symbols.size(); si != se; ++si) {
        errs() << '\n' << "/*" << Symbols[si].first << "  " << Symbols[si].second << ":*/\n";
    }
  #endif

    // Make a list of all the relocations for this section.
    std::vector<RelocationRef> Rels;

    // Sort relocations by address.
    std::sort(Rels.begin(), Rels.end(), isRelocAddressLess);

    StringRef name;
    getName(Section, name);

    // If the section has no symbols just insert a dummy one and disassemble
    // the whole section.
    if (Symbols.empty())
      Symbols.push_back(std::make_pair(0, name));

    SmallString<40> Comments;
    raw_svector_ostream CommentStream(Comments);

    ArrayRef<uint8_t> Bytes = arrayRefFromStringRef(
        unwrapOrError(Section.getContents(), Obj->getFileName()));
#if 0
    Section.getContents();
    ArrayRef<uint8_t> Bytes(reinterpret_cast<const uint8_t *>(BytesStr.data()),
                            BytesStr.size());
#endif
    uint64_t Size;
    SectSize = Section.getSize();

    // Disassemble symbol by symbol.
    unsigned se;
    for (si = 0, se = Symbols.size(); si != se; ++si) {
      uint64_t Start = Symbols[si].first;
      uint64_t End;
      // The end is either the size of the section or the beginning of the next
      // symbol.
      if (si == se - 1)
        End = SectSize;
      // Make sure this symbol takes up space.
      else if (Symbols[si + 1].first != Start)
        End = Symbols[si + 1].first - 1;
      else {
        continue;
      }

      for (Index = Start; Index < End; Index += Size) {
        MCInst Inst;
        if (DisAsm->getInstruction(Inst, Size, Bytes.slice(Index),
                                   SectionAddr + Index, CommentStream)) {
          hexOut.ProcessDisAsmInstruction(Inst, Size, Bytes, Obj);
        } else {
          errs() << ToolName << ": warning: invalid instruction encoding\n";
          if (Size == 0)
            Size = 1; // skip illegible bytes
        }
      } // for
    } // for
  }
}

// Porting from disassembleObject() of llvm-objump.cpp
static void Elf2Hex(const ObjectFile *Obj) {

  const Target *TheTarget = getTarget(Obj);

  // Package up features to be passed to target/subtarget
  SubtargetFeatures Features = Obj->getFeatures();

  std::unique_ptr<const MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TripleName));
  if (!MRI)
    report_fatal_error("error: no register info for target " + TripleName);

  // Set up disassembler.
  MCTargetOptions MCOptions;
  std::unique_ptr<const MCAsmInfo> AsmInfo(
    TheTarget->createMCAsmInfo(*MRI, TripleName, MCOptions));
  if (!AsmInfo)
    report_fatal_error("error: no assembly info for target " + TripleName);

  std::unique_ptr<const MCSubtargetInfo> STI(
    TheTarget->createMCSubtargetInfo(TripleName, "", Features.getString()));
  if (!STI)
    report_fatal_error("error: no subtarget info for target " + TripleName);

  std::unique_ptr<const MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  if (!MII)
    report_fatal_error("error: no instruction info for target " + TripleName);

  MCObjectFileInfo MOFI;
  MCContext Ctx(AsmInfo.get(), MRI.get(), &MOFI);
  // FIXME: for now initialize MCObjectFileInfo with default values
  MOFI.InitMCObjectFileInfo(Triple(TripleName), false, Ctx);

  std::unique_ptr<MCDisassembler> DisAsm(
    TheTarget->createMCDisassembler(*STI, Ctx));
  if (!DisAsm)
    report_fatal_error("error: no disassembler for target " + TripleName);

  std::unique_ptr<const MCInstrAnalysis> MIA(
      TheTarget->createMCInstrAnalysis(MII.get()));

  int AsmPrinterVariant = AsmInfo->getAssemblerDialect();
  std::unique_ptr<MCInstPrinter> IP(TheTarget->createMCInstPrinter(
      Triple(TripleName), AsmPrinterVariant, *AsmInfo, *MII, *MRI));
  if (!IP)
    report_fatal_error("error: no instruction printer for target " +
                       TripleName);

  std::error_code EC;
  reader.DisassembleObject(Obj, DisAsm, IP, STI);
}

static void DumpObject(const ObjectFile *o) {
  outs() << "/*";
  outs() << o->getFileName()
         << ":\tfile format " << o->getFileFormatName() << "*/";
  outs() << "\n\n";

  Elf2Hex(o);
}

/// @brief Open file and figure out how to dump it.
static void DumpInput(StringRef file) {
  CurrInputFile = file;
  // Attempt to open the binary.
  Expected<OwningBinary<Binary>> BinaryOrErr = createBinary(file);
  if (!BinaryOrErr)
    reportError(file, "no this file");

  Binary &Binary = *BinaryOrErr.get().getBinary();

  if (ObjectFile *o = dyn_cast<ObjectFile>(&Binary))
    DumpObject(o);
  else
    reportError(file, "invalid_file_type");
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  //sys::PrintStackTraceOnErrorSignal(argv[0]);
  //PrettyStackTraceProgram X(argc, argv);
  //llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  using namespace llvm;
  InitLLVM X(argc, argv);

  // Initialize targets and assembly printers/parsers.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();

  // Register the target printer for --version.
  cl::AddExtraVersionPrinter(TargetRegistry::printRegisteredTargetsForVersion);

  cl::ParseCommandLineOptions(argc, argv, "llvm object file dumper\n");
//  TripleName = Triple::normalize(TripleName);

  ToolName = argv[0];

  // Defaults to a.out if no filenames specified.
  if (InputFilenames.size() == 0)
    InputFilenames.push_back("a.out");

  std::for_each(InputFilenames.begin(), InputFilenames.end(),
                DumpInput);

  return EXIT_SUCCESS;
}

