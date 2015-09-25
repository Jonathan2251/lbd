//===- lib/ReaderWriter/ELF/Cpu0/Cpu0LinkingContext.cpp -------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Cpu0LinkingContext.h"

#include "lld/Core/File.h"
#include "lld/Core/Instrumentation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSwitch.h"

#include "Atoms.h"
#include "Cpu0RelocationPass.h"


using namespace lld;

using namespace lld::elf;

namespace {
using namespace llvm::ELF;
const uint8_t cpu0InitFiniAtomContent[8] = { 0 };

// Cpu0_64InitFini Atom
class Cpu0InitAtom : public InitFiniAtom {
public:
  Cpu0InitAtom(const File &f, StringRef function)
      : InitFiniAtom(f, ".init_array") {
#ifndef NDEBUG
    _name = "__init_fn_";
    _name += function;
#endif
  }
  virtual ArrayRef<uint8_t> rawContent() const {
    return ArrayRef<uint8_t>(cpu0InitFiniAtomContent, 8);
  }
  virtual Alignment alignment() const { return Alignment(3); }
};

class Cpu0FiniAtom : public InitFiniAtom {
public:
  Cpu0FiniAtom(const File &f, StringRef function)
      : InitFiniAtom(f, ".fini_array") {
#ifndef NDEBUG
    _name = "__fini_fn_";
    _name += function;
#endif
  }
  virtual ArrayRef<uint8_t> rawContent() const {
    return ArrayRef<uint8_t>(cpu0InitFiniAtomContent, 8);
  }

  virtual Alignment alignment() const { return Alignment(3); }
};

class Cpu0InitFiniFile : public SimpleFile {
public:
  Cpu0InitFiniFile(const ELFLinkingContext &context)
      : SimpleFile(context, "command line option -init/-fini"), _ordinal(0) {}

  void addInitFunction(StringRef name) {
    Atom *initFunctionAtom = new (_allocator) SimpleUndefinedAtom(*this, name);
    Cpu0InitAtom *initAtom =
           (new (_allocator) Cpu0InitAtom(*this, name));
    initAtom->addReference(llvm::ELF::R_CPU0_32, 0, initFunctionAtom, 0);
    initAtom->setOrdinal(_ordinal++);
    addAtom(*initFunctionAtom);
    addAtom(*initAtom);
  }

  void addFiniFunction(StringRef name) {
    Atom *finiFunctionAtom = new (_allocator) SimpleUndefinedAtom(*this, name);
    Cpu0FiniAtom *finiAtom =
           (new (_allocator) Cpu0FiniAtom(*this, name));
    finiAtom->addReference(llvm::ELF::R_CPU0_32, 0, finiFunctionAtom, 0);
    finiAtom->setOrdinal(_ordinal++);
    addAtom(*finiFunctionAtom);
    addAtom(*finiAtom);
  }

private:
  llvm::BumpPtrAllocator _allocator;
  uint64_t _ordinal;
};

} // end anon namespace

void elf::Cpu0LinkingContext::addPasses(PassManager &pm) {
  auto pass = createCpu0RelocationPass(*this);
  if (pass)
    pm.add(std::move(pass));
  ELFLinkingContext::addPasses(pm);
}

bool elf::Cpu0LinkingContext::createInternalFiles(
    std::vector<std::unique_ptr<File> > &result) const {
  ELFLinkingContext::createInternalFiles(result);
  std::unique_ptr<Cpu0InitFiniFile> initFiniFile(
      new Cpu0InitFiniFile(*this));
  for (auto ai : initFunctions())
    initFiniFile->addInitFunction(ai);
  for (auto ai:finiFunctions())
    initFiniFile->addFiniFunction(ai);
  result.push_back(std::move(initFiniFile));
  return true;
}

#define LLD_CASE(name) .Case(#name, llvm::ELF::name)

ErrorOr<Reference::Kind>
elf::Cpu0LinkingContext::relocKindFromString(StringRef str) const {
  int32_t ret = llvm::StringSwitch<int32_t>(str)
  LLD_CASE(R_CPU0_NONE)
  LLD_CASE(R_CPU0_24)
  LLD_CASE(R_CPU0_32)
  LLD_CASE(R_CPU0_HI16)
  LLD_CASE(R_CPU0_LO16)
  LLD_CASE(R_CPU0_GPREL16)
  LLD_CASE(R_CPU0_LITERAL)
  LLD_CASE(R_CPU0_GOT16)
  LLD_CASE(R_CPU0_PC24)
  LLD_CASE(R_CPU0_CALL16)
  LLD_CASE(R_CPU0_JUMP_SLOT)
    .Case("LLD_R_CPU0_GOTRELINDEX", LLD_R_CPU0_GOTRELINDEX)
    .Default(-1);

  if (ret == -1)
    return make_error_code(YamlReaderError::illegal_value);
  return ret;
}

#undef LLD_CASE

#define LLD_CASE(name) case llvm::ELF::name: return std::string(#name);

ErrorOr<std::string>
elf::Cpu0LinkingContext::stringFromRelocKind(Reference::Kind kind) const {
  switch (kind) {
  LLD_CASE(R_CPU0_NONE)
  LLD_CASE(R_CPU0_24)
  LLD_CASE(R_CPU0_32)
  LLD_CASE(R_CPU0_HI16)
  LLD_CASE(R_CPU0_LO16)
  LLD_CASE(R_CPU0_GPREL16)
  LLD_CASE(R_CPU0_LITERAL)
  LLD_CASE(R_CPU0_GOT16)
  LLD_CASE(R_CPU0_PC24)
  LLD_CASE(R_CPU0_CALL16)
  LLD_CASE(R_CPU0_JUMP_SLOT)
  case LLD_R_CPU0_GOTRELINDEX:
    return std::string("LLD_R_CPU0_GOTRELINDEX");
  }

  return make_error_code(YamlReaderError::illegal_value);
}
