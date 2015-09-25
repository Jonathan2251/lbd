//===- lib/ReaderWriter/ELF/Cpu0/Cpu0RelocationHandler.cpp ------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Cpu0TargetHandler.h"
#include "Cpu0LinkingContext.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"

using namespace lld;
using namespace elf;
using namespace llvm;
using namespace object;

static bool error(error_code ec) {
  if (!ec) return false;

  outs() << "Cpu0RelocationHandler.cpp : error reading file: " 
         << ec.message() << ".\n";
  outs().flush();
  return true;
}

namespace {
/// \brief R_CPU0_HI16 - word64: (S + A) >> 16
void relocHI16(uint8_t *location, uint64_t P, uint64_t S, int64_t A) {
 // Don't know why A, ref.addend(), = 9
  uint32_t result = (uint32_t)(S >> 16);
  *reinterpret_cast<llvm::support::ubig32_t *>(location) =
      result |
      (uint32_t) * reinterpret_cast<llvm::support::ubig32_t *>(location);
}

void relocLO16(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
 // Don't know why A, ref.addend(), = 9
  uint32_t result = (uint32_t)(S & 0x0000ffff);
  *reinterpret_cast<llvm::support::ubig32_t *>(location) =
      result |
      (uint32_t) * reinterpret_cast<llvm::support::ubig32_t *>(location);
}

/// \brief R_CPU0_GOT16 - word32: S
void relocGOT16(uint8_t *location, uint64_t P, uint64_t S, int64_t A) {
  uint32_t result = (uint32_t)(S);
  *reinterpret_cast<llvm::support::ubig32_t *>(location) =
      result |
      (uint32_t) * reinterpret_cast<llvm::support::ubig32_t *>(location);
}

/// \brief R_CPU0_PC24 - word32: S + A - P
void relocPC24(uint8_t *location, uint64_t P, uint64_t S, int64_t A) {
  uint32_t result = (uint32_t)(S  - P);
  uint32_t machinecode = (uint32_t) * 
                         reinterpret_cast<llvm::support::ubig32_t *>(location);
  uint32_t opcode = (machinecode & 0xff000000);
  uint32_t offset = (machinecode & 0x00ffffff);
  *reinterpret_cast<llvm::support::ubig32_t *>(location) =
      (((result + offset) & 0x00ffffff) | opcode);
}

/// \brief R_CPU0_32 - word24:  S
void reloc24(uint8_t *location, uint64_t P, uint64_t S, int64_t A) {
  int32_t addr = (uint32_t)(S & 0x00ffffff);
  uint32_t machinecode = (uint32_t) * 
                         reinterpret_cast<llvm::support::ubig32_t *>(location);
  uint32_t opcode = (machinecode & 0xff000000);
  *reinterpret_cast<llvm::support::ubig32_t *>(location) =
      (opcode | addr);
  // TODO: Make sure that the result zero extends to the 64bit value.
}

/// \brief R_CPU0_32 - word32:  S
void reloc32(uint8_t *location, uint64_t P, uint64_t S, int64_t A) {
  int32_t result = (uint32_t)(S);
  *reinterpret_cast<llvm::support::ubig32_t *>(location) =
      result |
      (uint32_t) * reinterpret_cast<llvm::support::ubig32_t *>(location);
  // TODO: Make sure that the result zero extends to the 64bit value.
}
} // end anon namespace

int64_t Cpu0TargetRelocationHandler::relocAddend(const Reference &ref) const {
  switch (ref.kind()) {
  case R_CPU0_PC24:
    return 4;
  default:
    return 0;
  }
  return 0;
}

#ifdef DLINKER
class Cpu0SoPlt {
private:
  uint32_t funAddr[100];
  int funAddrSize = 0;
public:
  void createFunAddr(const Cpu0LinkingContext &context, 
                     llvm::FileOutputBuffer &buf);
  // Return function index, 1: 1st function appear on section .text of .so.
  //   2: 2nd function ...
  // For example: 3 functions _Z2laii, _Z3fooii and _Z3barv. 1: is _Z2laii 
  //   2 is _Z3fooii, 3: is _Z3barv.
  int getDynFunIndexByTargetAddr(uint64_t fAddr);
};

void Cpu0SoPlt::createFunAddr(const Cpu0LinkingContext &context, 
                   llvm::FileOutputBuffer &buf) {
  auto dynsymSection = context.getTargetHandler<Cpu0ELFType>().targetLayout().
                       findOutputSection(".dynsym");
  uint64_t dynsymFileOffset, dynsymSize;
  if (dynsymSection) {
    dynsymFileOffset = dynsymSection->fileOffset();
    dynsymSize = dynsymSection->memSize();
    uint8_t *atomContent = buf.getBufferStart() + dynsymFileOffset;
    for (uint64_t i = 4; i < dynsymSize; i += 16) {
      funAddr[funAddrSize] = 
        *reinterpret_cast<llvm::support::ubig32_t*>((uint32_t*)
        (atomContent + i));
      funAddrSize++;
    }
  }
}

int Cpu0SoPlt::getDynFunIndexByTargetAddr(uint64_t fAddr) {
  for (int i = 0; i < funAddrSize; i++) {
    // Below statement fix the issue that both __tls_get_addr and first 
    // function has the same file offset 0 issue.
    if (i < (funAddrSize-1) && funAddr[i] == funAddr[i+1])
      continue;

    if (fAddr == funAddr[i]) {
      return i;
    }
  }
  return -1;
}

Cpu0SoPlt cpu0SoPlt;
#endif // DLINKER

ErrorOr<void> Cpu0TargetRelocationHandler::applyRelocation(
    ELFWriter &writer, llvm::FileOutputBuffer &buf, const lld::AtomLayout &atom,
    const Reference &ref) const {
#ifdef DLINKER
  static bool firstTime = true;
  std::string soName("libfoobar.cpu0.so");
  int idx = 0;
  if (firstTime) {
    if (_context.getOutputELFType() == llvm::ELF::ET_DYN) {
      cpu0SoPlt.createFunAddr(_context, buf);
    }
    else if (_context.getOutputELFType() == llvm::ELF::ET_EXEC && 
             !_context.isStaticExecutable()) {
      cpu0SoPlt.createFunAddr(_context, buf);
    }
    firstTime = false;
  }
#endif // DLINKER
  uint8_t *atomContent = buf.getBufferStart() + atom._fileOffset;
  uint8_t *location = atomContent + ref.offsetInAtom();
  uint64_t targetVAddress = writer.addressOfAtom(ref.target());
  uint64_t relocVAddress = atom._virtualAddr + ref.offsetInAtom();
#if 1 // For case R_CPU0_GOT16:
//  auto gotAtomIter = _context.getTargetHandler<Cpu0ELFType>().targetLayout().
//                     findAbsoluteAtom("_GLOBAL_OFFSET_TABLE_");
//  uint64_t globalOffsetTableAddress = writer.addressOfAtom(*gotAtomIter);
// .got.plt start from _GLOBAL_OFFSET_TABLE_
  auto gotpltSection = _context.getTargetHandler<Cpu0ELFType>().targetLayout().
                       findOutputSection(".got.plt");
  uint64_t gotPltFileOffset;
  if (gotpltSection)
    gotPltFileOffset = gotpltSection->fileOffset();
  else
    gotPltFileOffset = 0;
#endif

  switch (ref.kind()) {
  case R_CPU0_NONE:
    break;
  case R_CPU0_HI16:
    relocHI16(location, relocVAddress, targetVAddress, ref.addend());
    break;
  case R_CPU0_LO16:
    relocLO16(location, relocVAddress, targetVAddress, ref.addend());
    break;
#if 0 // Not support yet
  case R_CPU0_GOT16:
#if 1
    idx = cpu0SoPlt.getDynFunIndexByTargetAddr(targetVAddress);
    relocGOT16(location, relocVAddress, idx, ref.addend());
#else
    relocGOT16(location, relocVAddress, (targetVAddress - gotPltFileOffset), 
               ref.addend());
#endif
    break;
#endif
  case R_CPU0_PC24:
    relocPC24(location, relocVAddress, targetVAddress, ref.addend());
    break;
#ifdef DLINKER
  case R_CPU0_CALL16:
  // offset at _GLOBAL_OFFSET_TABLE_ and $gp point to _GLOBAL_OFFSET_TABLE_.
    idx = cpu0SoPlt.getDynFunIndexByTargetAddr(targetVAddress);
    reloc32(location, relocVAddress, idx*0x04+16, ref.addend());
    break;
#endif // DLINKER
  case R_CPU0_24:
    reloc24(location, relocVAddress, targetVAddress, ref.addend());
    break;
  case R_CPU0_32:
    reloc32(location, relocVAddress, targetVAddress, ref.addend());
    break;

  // Runtime only relocations. Ignore here.
  case R_CPU0_JUMP_SLOT:
    break;
  case lld::Reference::kindLayoutAfter:
  case lld::Reference::kindLayoutBefore:
  case lld::Reference::kindInGroup:
    break;

  default: {
    std::string str;
    llvm::raw_string_ostream s(str);
    auto name = _context.stringFromRelocKind(ref.kind());
    s << "Unhandled relocation: " << atom._atom->file().path() << ":"
      << atom._atom->name() << "@" << ref.offsetInAtom() << " "
      << (name ? *name : "<unknown>") << " (" << ref.kind() << ")";
    s.flush();
    llvm_unreachable(str.c_str());
  }
  }

  return error_code::success();
}
