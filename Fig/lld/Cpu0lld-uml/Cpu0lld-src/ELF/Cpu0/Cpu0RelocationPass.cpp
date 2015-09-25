//===- lib/ReaderWriter/ELF/Cpu0/Cpu0RelocationPass.cpp ---------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the relocation processing pass for Cpu0. This includes
///   GOT and PLT entries, TLS, and ifunc.
///
/// This also includes aditional behaivor that gnu-ld and gold implement but
/// which is not specified anywhere.
///
//===----------------------------------------------------------------------===//

#include "Cpu0RelocationPass.h"

#include "lld/ReaderWriter/Simple.h"

#include "llvm/ADT/DenseMap.h"

#include "Atoms.h"
#include "Cpu0LinkingContext.h"

using namespace lld;
using namespace lld::elf;
using namespace llvm::ELF;

namespace {

// .plt value (entry 0)
const uint8_t cpu0BootAtomContent[16] = {
  0x36, 0xff, 0xff, 0xfc, // jmp _start
  0x36, 0x00, 0x00, 0x04, // jmp 4
  0x36, 0x00, 0x00, 0x04, // jmp 4
  0x36, 0xff, 0xff, 0xfc // jmp -4
};

#ifdef DLINKER
// .got values
const uint8_t cpu0GotAtomContent[16] = { 0 };

// .plt value (entry 0)
const uint8_t cpu0Plt0AtomContent[16] = {
  0x02, 0xeb, 0x00, 0x04, // st $lr, $zero, reloc-index ($gp)
  0x02, 0xcb, 0x00, 0x08, // st $fp, $zero, reloc-index ($gp)
  0x02, 0xdb, 0x00, 0x0c, // st $sp, $zero, reloc-index ($gp)
  0x36, 0xff, 0xff, 0xfc  // jmp dynamic_linker
};

// .plt values (other entries)
const uint8_t cpu0PltAtomContent[16] = {
  0x09, 0x60, 0x00, 0x00, // addiu $t9, $zero, reloc-index (=.dynsym_index)
  0x02, 0x6b, 0x00, 0x00, // st $t9, $zero, reloc-index ($gp)
  0x01, 0x6b, 0x00, 0x10, // ld $t9, 0x10($gp) (0x10($gp) point to plt0
  0x3c, 0x60, 0x00, 0x00  // ret $t9 // jump to Cpu0.Stub
};
#endif // DLINKER

/// boot record
class Cpu0BootAtom : public PLT0Atom {
public:
  Cpu0BootAtom(const File &f) : PLT0Atom(f) {
#ifndef NDEBUG
    _name = ".PLT0";
#endif
  }
  virtual ArrayRef<uint8_t> rawContent() const {
    return ArrayRef<uint8_t>(cpu0BootAtomContent, 16);
  }
};

#ifdef DLINKER
/// \brief Atoms that are used by Cpu0 dynamic linking
class Cpu0GOTAtom : public GOTAtom {
public:
  Cpu0GOTAtom(const File &f, StringRef secName) : GOTAtom(f, secName) {}

  virtual ArrayRef<uint8_t> rawContent() const {
    return ArrayRef<uint8_t>(cpu0GotAtomContent, 16);
  }
};

class Cpu0PLT0Atom : public PLT0Atom {
public:
  Cpu0PLT0Atom(const File &f) : PLT0Atom(f) {
#ifndef NDEBUG
    _name = ".PLT0";
#endif
  }
  virtual ArrayRef<uint8_t> rawContent() const {
    return ArrayRef<uint8_t>(cpu0Plt0AtomContent, 16);
  }
};

class Cpu0PLTAtom : public PLTAtom {
public:
  Cpu0PLTAtom(const File &f, StringRef secName) : PLTAtom(f, secName) {}

  virtual ArrayRef<uint8_t> rawContent() const {
    return ArrayRef<uint8_t>(cpu0PltAtomContent, 16);
  }
};
#endif // DLINKER

class ELFPassFile : public SimpleFile {
public:
  ELFPassFile(const ELFLinkingContext &eti) : SimpleFile(eti, "ELFPassFile") {
    setOrdinal(eti.getNextOrdinalAndIncrement());
  }

  llvm::BumpPtrAllocator _alloc;
};

/// \brief CRTP base for handling relocations.
template <class Derived> class RelocationPass : public Pass {
  /// \brief Handle a specific reference.
  void handleReference(const DefinedAtom &atom, const Reference &ref) {
    switch (ref.kind()) {
    case R_CPU0_CALL16:
      static_cast<Derived *>(this)->handlePLT32(ref);
      break;

    case R_CPU0_PC24:
      static_cast<Derived *>(this)->handlePlain(ref);
      break;
    }
  }

protected:
#ifdef DLINKER
  /// \brief get the PLT entry for a given IFUNC Atom.
  ///
  /// If the entry does not exist. Both the GOT and PLT entry is created.
  const PLTAtom *getIFUNCPLTEntry(const DefinedAtom *da) {
    auto plt = _pltMap.find(da);
    if (plt != _pltMap.end())
      return plt->second;
    auto ga = new (_file._alloc) Cpu0GOTAtom(_file, ".got.plt");
    ga->addReference(R_CPU0_RELGOT, 0, da, 0);
    auto pa = new (_file._alloc) Cpu0PLTAtom(_file, ".plt");
    pa->addReference(R_CPU0_PC24, 2, ga, -4);
#ifndef NDEBUG
    ga->_name = "__got_ifunc_";
    ga->_name += da->name();
    pa->_name = "__plt_ifunc_";
    pa->_name += da->name();
#endif
    _gotMap[da] = ga;
    _pltMap[da] = pa;
    _gotVector.push_back(ga);
    _pltVector.push_back(pa);
    return pa;
  }
#endif // DLINKER

  /// \brief Redirect the call to the PLT stub for the target IFUNC.
  ///
  /// This create a PLT and GOT entry for the IFUNC if one does not exist. The
  /// GOT entry and a IRELATIVE relocation to the original target resolver.
  ErrorOr<void> handleIFUNC(const Reference &ref) {
    auto target = dyn_cast_or_null<const DefinedAtom>(ref.target());
#ifdef DLINKER
    if (target && target->contentType() == DefinedAtom::typeResolver)
      const_cast<Reference &>(ref).setTarget(getIFUNCPLTEntry(target));
#endif // DLINKER
    return error_code::success();
  }

#ifdef DLINKER
  /// \brief Create a GOT entry for the TP offset of a TLS atom.
  const GOTAtom *getGOTTPOFF(const Atom *atom) {
    auto got = _gotMap.find(atom);
    if (got == _gotMap.end()) {
      auto g = new (_file._alloc) Cpu0GOTAtom(_file, ".got");
      g->addReference(R_CPU0_TLS_TPREL32, 0, atom, 0);
#ifndef NDEBUG
      g->_name = "__got_tls_";
      g->_name += atom->name();
#endif
      _gotMap[atom] = g;
      _gotVector.push_back(g);
      return g;
    }
    return got->second;
  }

  /// \brief Create a GOT entry containing 0.
  const GOTAtom *getNullGOT() {
    if (!_null) {
      _null = new (_file._alloc) Cpu0GOTAtom(_file, ".got.plt");
#ifndef NDEBUG
      _null->_name = "__got_null";
#endif
    }
    return _null;
  }

  const GOTAtom *getGOT(const DefinedAtom *da) {
    auto got = _gotMap.find(da);
    if (got == _gotMap.end()) {
      auto g = new (_file._alloc) Cpu0GOTAtom(_file, ".got");
      g->addReference(R_CPU0_32, 0, da, 0);
#ifndef NDEBUG
      g->_name = "__got_";
      g->_name += da->name();
#endif
      _gotMap[da] = g;
      _gotVector.push_back(g);
      return g;
    }
    return got->second;
  }
#endif // DLINKER

public:
  RelocationPass(const ELFLinkingContext &ctx)
      : _file(ctx), _ctx(ctx), _null(nullptr), _PLT0(nullptr), _got0(nullptr), 
        _boot(new Cpu0BootAtom(_file)) {}

  /// \brief Do the pass.
  ///
  /// The goal here is to first process each reference individually. Each call
  /// to handleReference may modify the reference itself and/or create new
  /// atoms which must be stored in one of the maps below.
  ///
  /// After all references are handled, the atoms created during that are all
  /// added to mf.
  virtual void perform(std::unique_ptr<MutableFile> &mf) {
    ScopedTask task(getDefaultDomain(), "Cpu0 GOT/PLT Pass");
    // Process all references.
    for (const auto &atom : mf->defined())
      for (const auto &ref : *atom)
        handleReference(*atom, *ref);

    // Add all created atoms to the link.
    uint64_t ordinal = 0;
    if (_ctx.getOutputELFType() == llvm::ELF::ET_EXEC) {
      MutableFile::DefinedAtomRange atomRange = mf->definedAtoms();
      auto it = atomRange.begin();
      bool find = false;
      for (it = atomRange.begin(); it < atomRange.end(); it++) {
        if ((*it)->name() == "_Z5startv") {
          find = true;
          break;
        }
      }
      assert(find && "not found _Z5startv\n");
      _boot->addReference(R_CPU0_PC24, 0, *it, -3);
      _boot->setOrdinal(ordinal++);
      mf->addAtom(*_boot);
    }
#ifdef DLINKER
    if (_PLT0) {
      MutableFile::DefinedAtomRange atomRange = mf->definedAtoms();
      auto it = atomRange.begin();
      bool find = false;
      for (it = atomRange.begin(); it < atomRange.end(); it++) {
        if ((*it)->name() == "_Z14dynamic_linkerv") {
          find = true;
          break;
        }
      }
      assert(find && "Cannot find _Z14dynamic_linkerv()");
      _PLT0->addReference(R_CPU0_PC24, 12, *it, -3);
      _PLT0->setOrdinal(ordinal++);
      mf->addAtom(*_PLT0);
    }
    for (auto &plt : _pltVector) {
      plt->setOrdinal(ordinal++);
      mf->addAtom(*plt);
    }
    if (_null) {
      _null->setOrdinal(ordinal++);
      mf->addAtom(*_null);
    }
    if (_PLT0) {
      _got0->setOrdinal(ordinal++);
      mf->addAtom(*_got0);
    }
    for (auto &got : _gotVector) {
      got->setOrdinal(ordinal++);
      mf->addAtom(*got);
    }
#endif // DLINKER
  }

protected:
  /// \brief Owner of all the Atoms created by this pass.
  ELFPassFile _file;
  const ELFLinkingContext &_ctx;

  /// \brief Map Atoms to their GOT entries.
  llvm::DenseMap<const Atom *, GOTAtom *> _gotMap;

  /// \brief Map Atoms to their PLT entries.
  llvm::DenseMap<const Atom *, PLTAtom *> _pltMap;
  /// \brief the list of GOT/PLT atoms
  std::vector<GOTAtom *> _gotVector;
  std::vector<PLTAtom *> _pltVector;
  PLT0Atom *_boot;

  /// \brief GOT entry that is always 0. Used for undefined weaks.
  GOTAtom *_null;

  /// \brief The got and plt entries for .PLT0. This is used to call into the
  /// dynamic linker for symbol resolution.
  /// @{
  PLT0Atom *_PLT0;
  GOTAtom *_got0;
  /// @}
};

/// This implements the static relocation model. Meaning GOT and PLT entries are
/// not created for references that can be directly resolved. These are
/// converted to a direct relocation. For entries that do require a GOT or PLT
/// entry, that entry is statically bound.
///
/// TLS always assumes module 1 and attempts to remove indirection.
class StaticRelocationPass 
    : public RelocationPass<StaticRelocationPass> {
public:
  StaticRelocationPass(const elf::Cpu0LinkingContext &ctx)
      : RelocationPass(ctx) {}

  ErrorOr<void> handlePlain(const Reference &ref) { return handleIFUNC(ref); }

  ErrorOr<void> handlePLT32(const Reference &ref) {
    // __tls_get_addr is handled elsewhere.
    if (ref.target() && ref.target()->name() == "__tls_get_addr") {
      const_cast<Reference &>(ref).setKind(R_CPU0_NONE);
      return error_code::success();
    } else
      // Static code doesn't need PLTs.
      const_cast<Reference &>(ref).setKind(R_CPU0_PC24);
    // Handle IFUNC.
    if (const DefinedAtom *da =
            dyn_cast_or_null<const DefinedAtom>(ref.target()))
      if (da->contentType() == DefinedAtom::typeResolver)
        return handleIFUNC(ref);
    return error_code::success();
  }

  ErrorOr<void> handleGOT(const Reference &ref) {
    if (isa<UndefinedAtom>(ref.target()))
      const_cast<Reference &>(ref).setTarget(getNullGOT());
    else if (const DefinedAtom *da = dyn_cast<const DefinedAtom>(ref.target()))
      const_cast<Reference &>(ref).setTarget(getGOT(da));
    return error_code::success();
  }
};

#ifdef DLINKER
class DynamicRelocationPass 
    : public RelocationPass<DynamicRelocationPass> {
public:
  DynamicRelocationPass(const elf::Cpu0LinkingContext &ctx)
      : RelocationPass(ctx) {}

  const PLT0Atom *getPLT0() {
    if (_PLT0)
      return _PLT0;
    // Fill in the null entry.
    getNullGOT();
    _PLT0 = new (_file._alloc) Cpu0PLT0Atom(_file);
    _got0 = new (_file._alloc) Cpu0GOTAtom(_file, ".got.plt");
#ifndef NDEBUG
    _got0->_name = "__got0";
#endif
    return _PLT0;
  }

  const PLTAtom *getPLTEntry(const Atom *a) {
    auto plt = _pltMap.find(a);
    if (plt != _pltMap.end())
      return plt->second;
    auto ga = new (_file._alloc) Cpu0GOTAtom(_file, ".got.plt");
    ga->addReference(R_CPU0_JUMP_SLOT, 0, a, 0);
    auto pa = new (_file._alloc) Cpu0PLTAtom(_file, ".plt");
    getPLT0();  // add _PLT0 and _got0
    // Set the starting address of the got entry to the second instruction in
    // the plt entry.
    ga->addReference(R_CPU0_32, 0, pa, 4);
#ifndef NDEBUG
    ga->_name = "__got_";
    ga->_name += a->name();
    pa->_name = "__plt_";
    pa->_name += a->name();
#endif
    _gotMap[a] = ga;
    _pltMap[a] = pa;
    _gotVector.push_back(ga);
    _pltVector.push_back(pa);
    return pa;
  }
  ErrorOr<void> handlePlain(const Reference &ref) {
    if (!ref.target())
      return error_code::success();
    if (auto sla = dyn_cast<SharedLibraryAtom>(ref.target())) {
      if (sla->type() == SharedLibraryAtom::Type::Code) {
        const_cast<Reference &>(ref).setTarget(getPLTEntry(sla));
        // When caller of execution file call shared library function
        // Turn this into a PC24 to the PLT entry.
        const_cast<Reference &>(ref).setKind(R_CPU0_PC24);
      }
    } else
      return handleIFUNC(ref);
    return error_code::success();
  }

  ErrorOr<void> handlePLT32(const Reference &ref) {
    // Handle IFUNC.
    if (const DefinedAtom *da =
            dyn_cast_or_null<const DefinedAtom>(ref.target()))
      if (da->contentType() == DefinedAtom::typeResolver)
        return handleIFUNC(ref);
    if (isa<const SharedLibraryAtom>(ref.target())) {
      const_cast<Reference &>(ref).setTarget(getPLTEntry(ref.target()));
      // Turn this into a PC24 to the PLT entry.
    #if 1
      const_cast<Reference &>(ref).setKind(R_CPU0_PC24);
    #endif
    }
    return error_code::success();
  }

  const GOTAtom *getSharedGOT(const SharedLibraryAtom *sla) {
    auto got = _gotMap.find(sla);
    if (got == _gotMap.end()) {
      auto g = new (_file._alloc) Cpu0GOTAtom(_file, ".got.dyn");
      g->addReference(R_CPU0_GLOB_DAT, 0, sla, 0);
#ifndef NDEBUG
      g->_name = "__got_";
      g->_name += sla->name();
#endif
      _gotMap[sla] = g;
      _gotVector.push_back(g);
      return g;
    }
    return got->second;
  }

  ErrorOr<void> handleGOT(const Reference &ref) {
    if (isa<UndefinedAtom>(ref.target()))
      const_cast<Reference &>(ref).setTarget(getNullGOT());
    else if (const DefinedAtom *da = dyn_cast<const DefinedAtom>(ref.target()))
      const_cast<Reference &>(ref).setTarget(getGOT(da));
    else if (const auto sla = dyn_cast<const SharedLibraryAtom>(ref.target()))
      const_cast<Reference &>(ref).setTarget(getSharedGOT(sla));
    return error_code::success();
  }
};
#endif // DLINKER
} // end anon namespace

std::unique_ptr<Pass>
lld::elf::createCpu0RelocationPass(const Cpu0LinkingContext &ctx) {
  switch (ctx.getOutputELFType()) {
  case llvm::ELF::ET_EXEC:
#ifdef DLINKER
    if (ctx.isDynamic())
      return std::unique_ptr<Pass>(new DynamicRelocationPass(ctx));
    else
#endif // DLINKER
      return std::unique_ptr<Pass>(new StaticRelocationPass(ctx));
#ifdef DLINKER
  case llvm::ELF::ET_DYN:
    return std::unique_ptr<Pass>(new DynamicRelocationPass(ctx));
#endif // DLINKER
  case llvm::ELF::ET_REL:
    return std::unique_ptr<Pass>();
  default:
    llvm_unreachable("Unhandled output file type");
  }
}
