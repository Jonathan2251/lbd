/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Subtarget Enumeration Source Fragment                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


#ifdef GET_SUBTARGETINFO_ENUM
#undef GET_SUBTARGETINFO_ENUM

namespace llvm {
namespace Cpu0 {
enum {
  FeatureChapter3_1 = 0,
  FeatureChapter3_2 = 1,
  FeatureChapter3_3 = 2,
  FeatureChapter3_4 = 3,
  FeatureChapter3_5 = 4,
  FeatureChapter4_1 = 5,
  FeatureChapter4_2 = 6,
  FeatureChapter5_1 = 7,
  FeatureChapter6_1 = 8,
  FeatureChapter7_1 = 9,
  FeatureChapter8_1 = 10,
  FeatureChapter8_2 = 11,
  FeatureChapter9_1 = 12,
  FeatureChapter9_2 = 13,
  FeatureChapter9_3 = 14,
  FeatureChapter10_1 = 15,
  FeatureChapter11_1 = 16,
  FeatureChapter11_2 = 17,
  FeatureChapter12_1 = 18,
  FeatureChapterAll = 19,
  FeatureCmp = 20,
  FeatureCpu032I = 21,
  FeatureCpu032II = 22,
  FeatureSlt = 23,
  NumSubtargetFeatures = 24
};
} // end namespace Cpu0
} // end namespace llvm

#endif // GET_SUBTARGETINFO_ENUM


#ifdef GET_SUBTARGETINFO_MC_DESC
#undef GET_SUBTARGETINFO_MC_DESC

namespace llvm {
// Sorted (by key) array of values for CPU features.
extern const llvm::SubtargetFeatureKV Cpu0FeatureKV[] = {
  { "ch10_1", "Enable Chapter instructions.", Cpu0::FeatureChapter10_1, { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ch11_1", "Enable Chapter instructions.", Cpu0::FeatureChapter11_1, { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ch11_2", "Enable Chapter instructions.", Cpu0::FeatureChapter11_2, { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ch12_1", "Enable Chapter instructions.", Cpu0::FeatureChapter12_1, { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ch3_1", "Enable Chapter instructions.", Cpu0::FeatureChapter3_1, { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ch3_2", "Enable Chapter instructions.", Cpu0::FeatureChapter3_2, { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ch3_3", "Enable Chapter instructions.", Cpu0::FeatureChapter3_3, { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ch3_4", "Enable Chapter instructions.", Cpu0::FeatureChapter3_4, { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ch3_5", "Enable Chapter instructions.", Cpu0::FeatureChapter3_5, { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ch4_1", "Enable Chapter instructions.", Cpu0::FeatureChapter4_1, { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ch4_2", "Enable Chapter instructions.", Cpu0::FeatureChapter4_2, { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ch5_1", "Enable Chapter instructions.", Cpu0::FeatureChapter5_1, { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ch6_1", "Enable Chapter instructions.", Cpu0::FeatureChapter6_1, { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ch7_1", "Enable Chapter instructions.", Cpu0::FeatureChapter7_1, { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ch8_1", "Enable Chapter instructions.", Cpu0::FeatureChapter8_1, { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ch8_2", "Enable Chapter instructions.", Cpu0::FeatureChapter8_2, { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ch9_1", "Enable Chapter instructions.", Cpu0::FeatureChapter9_1, { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ch9_2", "Enable Chapter instructions.", Cpu0::FeatureChapter9_2, { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "ch9_3", "Enable Chapter instructions.", Cpu0::FeatureChapter9_3, { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "chall", "Enable Chapter instructions.", Cpu0::FeatureChapterAll, { { { 0x7ffffULL, 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "cmp", "Enable 'cmp' instructions.", Cpu0::FeatureCmp, { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "cpu032I", "Cpu032I ISA Support", Cpu0::FeatureCpu032I, { { { 0x180000ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "cpu032II", "Cpu032II ISA Support (slt)", Cpu0::FeatureCpu032II, { { { 0x980000ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } } },
  { "slt", "Enable 'slt' instructions.", Cpu0::FeatureSlt, { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } } },
};

#ifdef DBGFIELD
#error "<target>GenSubtargetInfo.inc requires a DBGFIELD macro"
#endif
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
#define DBGFIELD(x) x,
#else
#define DBGFIELD(x)
#endif

// Functional units for "Cpu0GenericItineraries"
namespace Cpu0GenericItinerariesFU {
  const InstrStage::FuncUnits ALU = 1ULL << 0;
  const InstrStage::FuncUnits IMULDIV = 1ULL << 1;
} // end namespace Cpu0GenericItinerariesFU

extern const llvm::InstrStage Cpu0Stages[] = {
  { 0, 0, 0, llvm::InstrStage::Required }, // No itinerary
  { 1, Cpu0GenericItinerariesFU::ALU, -1, (llvm::InstrStage::ReservationKinds)0 }, // 1
  { 3, Cpu0GenericItinerariesFU::ALU, -1, (llvm::InstrStage::ReservationKinds)0 }, // 2
  { 1, Cpu0GenericItinerariesFU::IMULDIV, -1, (llvm::InstrStage::ReservationKinds)0 }, // 3
  { 17, Cpu0GenericItinerariesFU::IMULDIV, -1, (llvm::InstrStage::ReservationKinds)0 }, // 4
  { 38, Cpu0GenericItinerariesFU::IMULDIV, -1, (llvm::InstrStage::ReservationKinds)0 }, // 5
  { 0, 0, 0, llvm::InstrStage::Required } // End stages
};
extern const unsigned Cpu0OperandCycles[] = {
  0, // No itinerary
  0 // End operand cycles
};
extern const unsigned Cpu0ForwardingPaths[] = {
 0, // No itinerary
 0 // End bypass tables
};

static const llvm::InstrItinerary Cpu0GenericItineraries[] = {
  { 0, 0, 0, 0, 0 }, // 0 NoInstrModel
  { 0, 0, 0, 0, 0 }, // 1 IIPseudo
  { 1, 1, 2, 0, 0 }, // 2 IIAlu
  { 1, 1, 2, 0, 0 }, // 3 IIBranch
  { 1, 1, 2, 0, 0 }, // 4 II_CLO
  { 1, 1, 2, 0, 0 }, // 5 II_CLZ
  { 1, 2, 3, 0, 0 }, // 6 IILoad
  { 1, 3, 4, 0, 0 }, // 7 IIHiLo
  { 1, 4, 5, 0, 0 }, // 8 IIImul
  { 1, 1, 2, 0, 0 }, // 9 IIStore
  { 1, 5, 6, 0, 0 }, // 10 IIIdiv
  { 0, uint16_t(~0U), uint16_t(~0U), uint16_t(~0U), uint16_t(~0U) }// end marker
};

// ===============================================================
// Data tables for the new per-operand machine model.

// {ProcResourceIdx, Cycles}
extern const llvm::MCWriteProcResEntry Cpu0WriteProcResTable[] = {
  { 0,  0}, // Invalid
}; // Cpu0WriteProcResTable

// {Cycles, WriteResourceID}
extern const llvm::MCWriteLatencyEntry Cpu0WriteLatencyTable[] = {
  { 0,  0}, // Invalid
}; // Cpu0WriteLatencyTable

// {UseIdx, WriteResourceID, Cycles}
extern const llvm::MCReadAdvanceEntry Cpu0ReadAdvanceTable[] = {
  {0,  0,  0}, // Invalid
}; // Cpu0ReadAdvanceTable

#undef DBGFIELD

static const llvm::MCSchedModel NoSchedModel = {
  MCSchedModel::DefaultIssueWidth,
  MCSchedModel::DefaultMicroOpBufferSize,
  MCSchedModel::DefaultLoopMicroOpBufferSize,
  MCSchedModel::DefaultLoadLatency,
  MCSchedModel::DefaultHighLatency,
  MCSchedModel::DefaultMispredictPenalty,
  false, // PostRAScheduler
  false, // CompleteModel
  0, // Processor ID
  nullptr, nullptr, 0, 0, // No instruction-level machine model.
  nullptr, // No Itinerary
  nullptr // No extra processor descriptor
};

static const llvm::MCSchedModel Cpu0GenericItinerariesModel = {
  MCSchedModel::DefaultIssueWidth,
  MCSchedModel::DefaultMicroOpBufferSize,
  MCSchedModel::DefaultLoopMicroOpBufferSize,
  MCSchedModel::DefaultLoadLatency,
  MCSchedModel::DefaultHighLatency,
  MCSchedModel::DefaultMispredictPenalty,
  false, // PostRAScheduler
  false, // CompleteModel
  1, // Processor ID
  nullptr, nullptr, 0, 0, // No instruction-level machine model.
  Cpu0GenericItineraries,
  nullptr // No extra processor descriptor
};

// Sorted (by key) array of values for CPU subtype.
extern const llvm::SubtargetSubTypeKV Cpu0SubTypeKV[] = {
 { "cpu032I", { { { 0x200000ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } }, &Cpu0GenericItinerariesModel },
 { "cpu032II", { { { 0x400000ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } }, { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } }, &Cpu0GenericItinerariesModel },
};

namespace Cpu0_MC {
unsigned resolveVariantSchedClassImpl(unsigned SchedClass,
    const MCInst *MI, const MCInstrInfo *MCII, unsigned CPUID) {
  // Don't know how to resolve this scheduling class.
  return 0;
}
} // end namespace Cpu0_MC

struct Cpu0GenMCSubtargetInfo : public MCSubtargetInfo {
  Cpu0GenMCSubtargetInfo(const Triple &TT,
    StringRef CPU, StringRef TuneCPU, StringRef FS,
    ArrayRef<SubtargetFeatureKV> PF,
    ArrayRef<SubtargetSubTypeKV> PD,
    const MCWriteProcResEntry *WPR,
    const MCWriteLatencyEntry *WL,
    const MCReadAdvanceEntry *RA, const InstrStage *IS,
    const unsigned *OC, const unsigned *FP) :
      MCSubtargetInfo(TT, CPU, TuneCPU, FS, PF, PD,
                      WPR, WL, RA, IS, OC, FP) { }

  unsigned resolveVariantSchedClass(unsigned SchedClass,
      const MCInst *MI, const MCInstrInfo *MCII,
      unsigned CPUID) const override {
    return Cpu0_MC::resolveVariantSchedClassImpl(SchedClass, MI, MCII, CPUID);
  }
};

static inline MCSubtargetInfo *createCpu0MCSubtargetInfoImpl(const Triple &TT, StringRef CPU, StringRef TuneCPU, StringRef FS) {
  return new Cpu0GenMCSubtargetInfo(TT, CPU, TuneCPU, FS, Cpu0FeatureKV, Cpu0SubTypeKV, 
                      Cpu0WriteProcResTable, Cpu0WriteLatencyTable, Cpu0ReadAdvanceTable, 
                      Cpu0Stages, Cpu0OperandCycles, Cpu0ForwardingPaths);
}

} // end namespace llvm

#endif // GET_SUBTARGETINFO_MC_DESC


#ifdef GET_SUBTARGETINFO_TARGET_DESC
#undef GET_SUBTARGETINFO_TARGET_DESC

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

// ParseSubtargetFeatures - Parses features string setting specified
// subtarget options.
void llvm::Cpu0Subtarget::ParseSubtargetFeatures(StringRef CPU, StringRef TuneCPU, StringRef FS) {
  LLVM_DEBUG(dbgs() << "\nFeatures:" << FS);
  LLVM_DEBUG(dbgs() << "\nCPU:" << CPU);
  LLVM_DEBUG(dbgs() << "\nTuneCPU:" << TuneCPU << "\n\n");
  InitMCProcessorInfo(CPU, TuneCPU, FS);
  const FeatureBitset &Bits = getFeatureBits();
  if (Bits[Cpu0::FeatureChapter3_1]) HasChapterDummy = true;
  if (Bits[Cpu0::FeatureChapter3_2]) HasChapterDummy = true;
  if (Bits[Cpu0::FeatureChapter3_3]) HasChapterDummy = true;
  if (Bits[Cpu0::FeatureChapter3_4]) HasChapterDummy = true;
  if (Bits[Cpu0::FeatureChapter3_5]) HasChapterDummy = true;
  if (Bits[Cpu0::FeatureChapter4_1]) HasChapterDummy = true;
  if (Bits[Cpu0::FeatureChapter4_2]) HasChapterDummy = true;
  if (Bits[Cpu0::FeatureChapter5_1]) HasChapterDummy = true;
  if (Bits[Cpu0::FeatureChapter6_1]) HasChapterDummy = true;
  if (Bits[Cpu0::FeatureChapter7_1]) HasChapterDummy = true;
  if (Bits[Cpu0::FeatureChapter8_1]) HasChapterDummy = true;
  if (Bits[Cpu0::FeatureChapter8_2]) HasChapterDummy = true;
  if (Bits[Cpu0::FeatureChapter9_1]) HasChapterDummy = true;
  if (Bits[Cpu0::FeatureChapter9_2]) HasChapterDummy = true;
  if (Bits[Cpu0::FeatureChapter9_3]) HasChapterDummy = true;
  if (Bits[Cpu0::FeatureChapter10_1]) HasChapterDummy = true;
  if (Bits[Cpu0::FeatureChapter11_1]) HasChapterDummy = true;
  if (Bits[Cpu0::FeatureChapter11_2]) HasChapterDummy = true;
  if (Bits[Cpu0::FeatureChapter12_1]) HasChapterDummy = true;
  if (Bits[Cpu0::FeatureChapterAll]) HasChapterDummy = true;
  if (Bits[Cpu0::FeatureCmp]) HasCmp = true;
  if (Bits[Cpu0::FeatureCpu032I] && Cpu0ArchVersion < Cpu032I) Cpu0ArchVersion = Cpu032I;
  if (Bits[Cpu0::FeatureCpu032II] && Cpu0ArchVersion < Cpu032II) Cpu0ArchVersion = Cpu032II;
  if (Bits[Cpu0::FeatureSlt]) HasSlt = true;
}
#endif // GET_SUBTARGETINFO_TARGET_DESC


#ifdef GET_SUBTARGETINFO_HEADER
#undef GET_SUBTARGETINFO_HEADER

namespace llvm {
class DFAPacketizer;
namespace Cpu0_MC {
unsigned resolveVariantSchedClassImpl(unsigned SchedClass, const MCInst *MI, const MCInstrInfo *MCII, unsigned CPUID);
} // end namespace Cpu0_MC

struct Cpu0GenSubtargetInfo : public TargetSubtargetInfo {
  explicit Cpu0GenSubtargetInfo(const Triple &TT, StringRef CPU, StringRef TuneCPU, StringRef FS);
public:
  unsigned resolveSchedClass(unsigned SchedClass,  const MachineInstr *DefMI, const TargetSchedModel *SchedModel) const override;
  unsigned resolveVariantSchedClass(unsigned SchedClass, const MCInst *MI, const MCInstrInfo *MCII, unsigned CPUID) const override;
  DFAPacketizer *createDFAPacketizer(const InstrItineraryData *IID) const;
};
} // end namespace llvm

#endif // GET_SUBTARGETINFO_HEADER


#ifdef GET_SUBTARGETINFO_CTOR
#undef GET_SUBTARGETINFO_CTOR

#include "llvm/CodeGen/TargetSchedule.h"

namespace llvm {
extern const llvm::SubtargetFeatureKV Cpu0FeatureKV[];
extern const llvm::SubtargetSubTypeKV Cpu0SubTypeKV[];
extern const llvm::MCWriteProcResEntry Cpu0WriteProcResTable[];
extern const llvm::MCWriteLatencyEntry Cpu0WriteLatencyTable[];
extern const llvm::MCReadAdvanceEntry Cpu0ReadAdvanceTable[];
extern const llvm::InstrStage Cpu0Stages[];
extern const unsigned Cpu0OperandCycles[];
extern const unsigned Cpu0ForwardingPaths[];
Cpu0GenSubtargetInfo::Cpu0GenSubtargetInfo(const Triple &TT, StringRef CPU, StringRef TuneCPU, StringRef FS)
  : TargetSubtargetInfo(TT, CPU, TuneCPU, FS, makeArrayRef(Cpu0FeatureKV, 24), makeArrayRef(Cpu0SubTypeKV, 2), 
                        Cpu0WriteProcResTable, Cpu0WriteLatencyTable, Cpu0ReadAdvanceTable, 
                        Cpu0Stages, Cpu0OperandCycles, Cpu0ForwardingPaths) {}

unsigned Cpu0GenSubtargetInfo
::resolveSchedClass(unsigned SchedClass, const MachineInstr *MI, const TargetSchedModel *SchedModel) const {
  report_fatal_error("Expected a variant SchedClass");
} // Cpu0GenSubtargetInfo::resolveSchedClass

unsigned Cpu0GenSubtargetInfo
::resolveVariantSchedClass(unsigned SchedClass, const MCInst *MI, const MCInstrInfo *MCII, unsigned CPUID) const {
  return Cpu0_MC::resolveVariantSchedClassImpl(SchedClass, MI, MCII, CPUID);
} // Cpu0GenSubtargetInfo::resolveVariantSchedClass

} // end namespace llvm

#endif // GET_SUBTARGETINFO_CTOR


#ifdef GET_STIPREDICATE_DECLS_FOR_MC_ANALYSIS
#undef GET_STIPREDICATE_DECLS_FOR_MC_ANALYSIS

#endif // GET_STIPREDICATE_DECLS_FOR_MC_ANALYSIS


#ifdef GET_STIPREDICATE_DEFS_FOR_MC_ANALYSIS
#undef GET_STIPREDICATE_DEFS_FOR_MC_ANALYSIS

#endif // GET_STIPREDICATE_DEFS_FOR_MC_ANALYSIS

