/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Machine Code Emitter                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

uint64_t Cpu0MCCodeEmitter::getBinaryCodeForInstr(const MCInst &MI,
    SmallVectorImpl<MCFixup> &Fixups,
    const MCSubtargetInfo &STI) const {
  static const uint64_t InstBits[] = {
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(0),
    UINT64_C(318767104),	// ADD
    UINT64_C(150994944),	// ADDiu
    UINT64_C(285212672),	// ADDu
    UINT64_C(402653184),	// AND
    UINT64_C(201326592),	// ANDi
    UINT64_C(973078528),	// BAL
    UINT64_C(922746880),	// BEQ
    UINT64_C(939524096),	// BNE
    UINT64_C(1375731712),	// C0MOVE
    UINT64_C(369098752),	// CLO
    UINT64_C(352321536),	// CLZ
    UINT64_C(704643072),	// CMP
    UINT64_C(721420288),	// CMPu
    UINT64_C(1610612736),	// GCD
    UINT64_C(970981376),	// JALR
    UINT64_C(805306368),	// JEQ
    UINT64_C(889192448),	// JGE
    UINT64_C(855638016),	// JGT
    UINT64_C(872415232),	// JLE
    UINT64_C(838860800),	// JLT
    UINT64_C(905969664),	// JMP
    UINT64_C(822083584),	// JNE
    UINT64_C(1006632960),	// JR
    UINT64_C(989855744),	// JSUB
    UINT64_C(50331648),	// LB
    UINT64_C(67108864),	// LBu
    UINT64_C(16777216),	// LD
    UINT64_C(150994944),	// LEA_ADDiu
    UINT64_C(100663296),	// LH
    UINT64_C(117440512),	// LHu
    UINT64_C(1627389952),	// LL
    UINT64_C(251658240),	// LUi
    UINT64_C(1342177280),	// MFC0
    UINT64_C(1174405120),	// MFHI
    UINT64_C(1191182336),	// MFLO
    UINT64_C(184549376),	// MOVN_I_I
    UINT64_C(167772160),	// MOVZ_I_I
    UINT64_C(1358954496),	// MTC0
    UINT64_C(1207959552),	// MTHI
    UINT64_C(1224736768),	// MTLO
    UINT64_C(385875968),	// MUL
    UINT64_C(1090519040),	// MULT
    UINT64_C(1107296256),	// MULTu
    UINT64_C(0),	// NOP
    UINT64_C(452984832),	// NOR
    UINT64_C(419430400),	// OR
    UINT64_C(218103808),	// ORi
    UINT64_C(1006632960),	// RET
    UINT64_C(469762048),	// ROL
    UINT64_C(603979776),	// ROLV
    UINT64_C(486539264),	// ROR
    UINT64_C(620756992),	// RORV
    UINT64_C(83886080),	// SB
    UINT64_C(1644167168),	// SC
    UINT64_C(1124073472),	// SDIV
    UINT64_C(134217728),	// SH
    UINT64_C(503316480),	// SHL
    UINT64_C(570425344),	// SHLV
    UINT64_C(520093696),	// SHR
    UINT64_C(587202560),	// SHRV
    UINT64_C(671088640),	// SLT
    UINT64_C(637534208),	// SLTi
    UINT64_C(654311424),	// SLTiu
    UINT64_C(687865856),	// SLTu
    UINT64_C(536870912),	// SRA
    UINT64_C(553648128),	// SRAV
    UINT64_C(33554432),	// ST
    UINT64_C(335544320),	// SUB
    UINT64_C(301989888),	// SUBu
    UINT64_C(1610612736),	// SYNC
    UINT64_C(1140850688),	// UDIV
    UINT64_C(436207616),	// XOR
    UINT64_C(234881024),	// XORi
    UINT64_C(0)
  };
  const unsigned opcode = MI.getOpcode();
  uint64_t Value = InstBits[opcode];
  uint64_t op = 0;
  (void)op;  // suppress warning
  switch (opcode) {
    case Cpu0::NOP: {
      break;
    }
    case Cpu0::JEQ:
    case Cpu0::JGE:
    case Cpu0::JGT:
    case Cpu0::JLE:
    case Cpu0::JLT:
    case Cpu0::JNE: {
      // op: addr
      op = getBranch24TargetOpValue(MI, 1, Fixups, STI);
      op &= UINT64_C(16777215);
      Value |= op;
      break;
    }
    case Cpu0::BAL:
    case Cpu0::JMP:
    case Cpu0::JSUB: {
      // op: addr
      op = getJumpTargetOpValue(MI, 0, Fixups, STI);
      op &= UINT64_C(16777215);
      Value |= op;
      break;
    }
    case Cpu0::JR:
    case Cpu0::MFHI:
    case Cpu0::MFLO:
    case Cpu0::MTHI:
    case Cpu0::MTLO:
    case Cpu0::RET: {
      // op: ra
      op = getMachineOpValue(MI, MI.getOperand(0), Fixups, STI);
      op &= UINT64_C(15);
      op <<= 20;
      Value |= op;
      break;
    }
    case Cpu0::LB:
    case Cpu0::LBu:
    case Cpu0::LD:
    case Cpu0::LEA_ADDiu:
    case Cpu0::LH:
    case Cpu0::LHu:
    case Cpu0::LL:
    case Cpu0::SB:
    case Cpu0::SH:
    case Cpu0::ST: {
      // op: ra
      op = getMachineOpValue(MI, MI.getOperand(0), Fixups, STI);
      op &= UINT64_C(15);
      op <<= 20;
      Value |= op;
      // op: addr
      op = getMemEncoding(MI, 1, Fixups, STI);
      op &= UINT64_C(1048575);
      Value |= op;
      break;
    }
    case Cpu0::LUi: {
      // op: ra
      op = getMachineOpValue(MI, MI.getOperand(0), Fixups, STI);
      op &= UINT64_C(15);
      op <<= 20;
      Value |= op;
      // op: imm16
      op = getMachineOpValue(MI, MI.getOperand(1), Fixups, STI);
      op &= UINT64_C(65535);
      Value |= op;
      break;
    }
    case Cpu0::C0MOVE:
    case Cpu0::CLO:
    case Cpu0::CLZ:
    case Cpu0::MFC0:
    case Cpu0::MTC0:
    case Cpu0::MULT:
    case Cpu0::MULTu:
    case Cpu0::SDIV:
    case Cpu0::UDIV: {
      // op: ra
      op = getMachineOpValue(MI, MI.getOperand(0), Fixups, STI);
      op &= UINT64_C(15);
      op <<= 20;
      Value |= op;
      // op: rb
      op = getMachineOpValue(MI, MI.getOperand(1), Fixups, STI);
      op &= UINT64_C(15);
      op <<= 16;
      Value |= op;
      break;
    }
    case Cpu0::BEQ:
    case Cpu0::BNE: {
      // op: ra
      op = getMachineOpValue(MI, MI.getOperand(0), Fixups, STI);
      op &= UINT64_C(15);
      op <<= 20;
      Value |= op;
      // op: rb
      op = getMachineOpValue(MI, MI.getOperand(1), Fixups, STI);
      op &= UINT64_C(15);
      op <<= 16;
      Value |= op;
      // op: imm16
      op = getBranch16TargetOpValue(MI, 2, Fixups, STI);
      op &= UINT64_C(65535);
      Value |= op;
      break;
    }
    case Cpu0::ADDiu:
    case Cpu0::ANDi:
    case Cpu0::ORi:
    case Cpu0::SLTi:
    case Cpu0::SLTiu:
    case Cpu0::XORi: {
      // op: ra
      op = getMachineOpValue(MI, MI.getOperand(0), Fixups, STI);
      op &= UINT64_C(15);
      op <<= 20;
      Value |= op;
      // op: rb
      op = getMachineOpValue(MI, MI.getOperand(1), Fixups, STI);
      op &= UINT64_C(15);
      op <<= 16;
      Value |= op;
      // op: imm16
      op = getMachineOpValue(MI, MI.getOperand(2), Fixups, STI);
      op &= UINT64_C(65535);
      Value |= op;
      break;
    }
    case Cpu0::ADD:
    case Cpu0::ADDu:
    case Cpu0::AND:
    case Cpu0::CMP:
    case Cpu0::CMPu:
    case Cpu0::GCD:
    case Cpu0::MOVN_I_I:
    case Cpu0::MOVZ_I_I:
    case Cpu0::MUL:
    case Cpu0::NOR:
    case Cpu0::OR:
    case Cpu0::ROLV:
    case Cpu0::RORV:
    case Cpu0::SHLV:
    case Cpu0::SHRV:
    case Cpu0::SLT:
    case Cpu0::SLTu:
    case Cpu0::SRAV:
    case Cpu0::SUB:
    case Cpu0::SUBu:
    case Cpu0::XOR: {
      // op: ra
      op = getMachineOpValue(MI, MI.getOperand(0), Fixups, STI);
      op &= UINT64_C(15);
      op <<= 20;
      Value |= op;
      // op: rb
      op = getMachineOpValue(MI, MI.getOperand(1), Fixups, STI);
      op &= UINT64_C(15);
      op <<= 16;
      Value |= op;
      // op: rc
      op = getMachineOpValue(MI, MI.getOperand(2), Fixups, STI);
      op &= UINT64_C(15);
      op <<= 12;
      Value |= op;
      break;
    }
    case Cpu0::ROL:
    case Cpu0::ROR:
    case Cpu0::SHL:
    case Cpu0::SHR:
    case Cpu0::SRA: {
      // op: ra
      op = getMachineOpValue(MI, MI.getOperand(0), Fixups, STI);
      op &= UINT64_C(15);
      op <<= 20;
      Value |= op;
      // op: rb
      op = getMachineOpValue(MI, MI.getOperand(1), Fixups, STI);
      op &= UINT64_C(15);
      op <<= 16;
      Value |= op;
      // op: shamt
      op = getMachineOpValue(MI, MI.getOperand(2), Fixups, STI);
      op &= UINT64_C(4095);
      Value |= op;
      break;
    }
    case Cpu0::SC: {
      // op: ra
      op = getMachineOpValue(MI, MI.getOperand(1), Fixups, STI);
      op &= UINT64_C(15);
      op <<= 20;
      Value |= op;
      // op: addr
      op = getMemEncoding(MI, 2, Fixups, STI);
      op &= UINT64_C(1048575);
      Value |= op;
      break;
    }
    case Cpu0::JALR: {
      // op: rb
      op = getMachineOpValue(MI, MI.getOperand(0), Fixups, STI);
      op &= UINT64_C(15);
      op <<= 16;
      Value |= op;
      break;
    }
    case Cpu0::SYNC: {
      // op: stype
      op = getMachineOpValue(MI, MI.getOperand(0), Fixups, STI);
      op &= UINT64_C(31);
      op <<= 6;
      Value |= op;
      break;
    }
  default:
    std::string msg;
    raw_string_ostream Msg(msg);
    Msg << "Not supported instr: " << MI;
    report_fatal_error(Msg.str());
  }
  return Value;
}

#ifdef ENABLE_INSTR_PREDICATE_VERIFIER
#undef ENABLE_INSTR_PREDICATE_VERIFIER
#include <sstream>

// Bits for subtarget features that participate in instruction matching.
enum SubtargetFeatureBits : uint8_t {
  Feature_Ch3_1Bit = 4,
  Feature_Ch3_2Bit = 5,
  Feature_Ch3_3Bit = 6,
  Feature_Ch3_4Bit = 7,
  Feature_Ch3_5Bit = 8,
  Feature_Ch4_1Bit = 9,
  Feature_Ch4_2Bit = 10,
  Feature_Ch5_1Bit = 11,
  Feature_Ch6_1Bit = 12,
  Feature_Ch7_1Bit = 13,
  Feature_Ch8_1Bit = 14,
  Feature_Ch8_2Bit = 15,
  Feature_Ch9_1Bit = 16,
  Feature_Ch9_2Bit = 17,
  Feature_Ch9_3Bit = 18,
  Feature_Ch10_1Bit = 0,
  Feature_Ch11_1Bit = 1,
  Feature_Ch11_2Bit = 2,
  Feature_Ch12_1Bit = 3,
  Feature_Ch_allBit = 19,
};

#ifndef NDEBUG
static const char *SubtargetFeatureNames[] = {
  "Feature_Ch10_1",
  "Feature_Ch11_1",
  "Feature_Ch11_2",
  "Feature_Ch12_1",
  "Feature_Ch3_1",
  "Feature_Ch3_2",
  "Feature_Ch3_3",
  "Feature_Ch3_4",
  "Feature_Ch3_5",
  "Feature_Ch4_1",
  "Feature_Ch4_2",
  "Feature_Ch5_1",
  "Feature_Ch6_1",
  "Feature_Ch7_1",
  "Feature_Ch8_1",
  "Feature_Ch8_2",
  "Feature_Ch9_1",
  "Feature_Ch9_2",
  "Feature_Ch9_3",
  "Feature_Ch_all",
  nullptr
};

#endif // NDEBUG
FeatureBitset Cpu0MCCodeEmitter::
computeAvailableFeatures(const FeatureBitset &FB) const {
  FeatureBitset Features;
  if (FB[Cpu0::FeatureChapter3_1])
    Features.set(Feature_Ch3_1Bit);
  if (FB[Cpu0::FeatureChapter3_2])
    Features.set(Feature_Ch3_2Bit);
  if (FB[Cpu0::FeatureChapter3_3])
    Features.set(Feature_Ch3_3Bit);
  if (FB[Cpu0::FeatureChapter3_4])
    Features.set(Feature_Ch3_4Bit);
  if (FB[Cpu0::FeatureChapter3_5])
    Features.set(Feature_Ch3_5Bit);
  if (FB[Cpu0::FeatureChapter4_1])
    Features.set(Feature_Ch4_1Bit);
  if (FB[Cpu0::FeatureChapter4_2])
    Features.set(Feature_Ch4_2Bit);
  if (FB[Cpu0::FeatureChapter5_1])
    Features.set(Feature_Ch5_1Bit);
  if (FB[Cpu0::FeatureChapter6_1])
    Features.set(Feature_Ch6_1Bit);
  if (FB[Cpu0::FeatureChapter7_1])
    Features.set(Feature_Ch7_1Bit);
  if (FB[Cpu0::FeatureChapter8_1])
    Features.set(Feature_Ch8_1Bit);
  if (FB[Cpu0::FeatureChapter8_2])
    Features.set(Feature_Ch8_2Bit);
  if (FB[Cpu0::FeatureChapter9_1])
    Features.set(Feature_Ch9_1Bit);
  if (FB[Cpu0::FeatureChapter9_2])
    Features.set(Feature_Ch9_2Bit);
  if (FB[Cpu0::FeatureChapter9_3])
    Features.set(Feature_Ch9_3Bit);
  if (FB[Cpu0::FeatureChapter10_1])
    Features.set(Feature_Ch10_1Bit);
  if (FB[Cpu0::FeatureChapter11_1])
    Features.set(Feature_Ch11_1Bit);
  if (FB[Cpu0::FeatureChapter11_2])
    Features.set(Feature_Ch11_2Bit);
  if (FB[Cpu0::FeatureChapter12_1])
    Features.set(Feature_Ch12_1Bit);
  if (FB[Cpu0::FeatureChapterAll])
    Features.set(Feature_Ch_allBit);
  return Features;
}

#ifndef NDEBUG
// Feature bitsets.
enum : uint8_t {
  CEFBS_None,
  CEFBS_Ch11_1,
  CEFBS_Ch12_1,
  CEFBS_Ch3_4,
  CEFBS_Ch3_5,
  CEFBS_Ch4_1,
  CEFBS_Ch4_2,
  CEFBS_Ch7_1,
  CEFBS_Ch8_1,
  CEFBS_Ch8_2,
  CEFBS_Ch9_1,
  CEFBS_Ch9_2,
};

static constexpr FeatureBitset FeatureBitsets[] = {
  {}, // CEFBS_None
  {Feature_Ch11_1Bit, },
  {Feature_Ch12_1Bit, },
  {Feature_Ch3_4Bit, },
  {Feature_Ch3_5Bit, },
  {Feature_Ch4_1Bit, },
  {Feature_Ch4_2Bit, },
  {Feature_Ch7_1Bit, },
  {Feature_Ch8_1Bit, },
  {Feature_Ch8_2Bit, },
  {Feature_Ch9_1Bit, },
  {Feature_Ch9_2Bit, },
};
#endif // NDEBUG

void Cpu0MCCodeEmitter::verifyInstructionPredicates(
    const MCInst &Inst, const FeatureBitset &AvailableFeatures) const {
#ifndef NDEBUG
  static uint8_t RequiredFeaturesRefs[] = {
    CEFBS_None, // PHI = 0
    CEFBS_None, // INLINEASM = 1
    CEFBS_None, // INLINEASM_BR = 2
    CEFBS_None, // CFI_INSTRUCTION = 3
    CEFBS_None, // EH_LABEL = 4
    CEFBS_None, // GC_LABEL = 5
    CEFBS_None, // ANNOTATION_LABEL = 6
    CEFBS_None, // KILL = 7
    CEFBS_None, // EXTRACT_SUBREG = 8
    CEFBS_None, // INSERT_SUBREG = 9
    CEFBS_None, // IMPLICIT_DEF = 10
    CEFBS_None, // SUBREG_TO_REG = 11
    CEFBS_None, // COPY_TO_REGCLASS = 12
    CEFBS_None, // DBG_VALUE = 13
    CEFBS_None, // DBG_INSTR_REF = 14
    CEFBS_None, // DBG_LABEL = 15
    CEFBS_None, // REG_SEQUENCE = 16
    CEFBS_None, // COPY = 17
    CEFBS_None, // BUNDLE = 18
    CEFBS_None, // LIFETIME_START = 19
    CEFBS_None, // LIFETIME_END = 20
    CEFBS_None, // PSEUDO_PROBE = 21
    CEFBS_None, // STACKMAP = 22
    CEFBS_None, // FENTRY_CALL = 23
    CEFBS_None, // PATCHPOINT = 24
    CEFBS_None, // LOAD_STACK_GUARD = 25
    CEFBS_None, // PREALLOCATED_SETUP = 26
    CEFBS_None, // PREALLOCATED_ARG = 27
    CEFBS_None, // STATEPOINT = 28
    CEFBS_None, // LOCAL_ESCAPE = 29
    CEFBS_None, // FAULTING_OP = 30
    CEFBS_None, // PATCHABLE_OP = 31
    CEFBS_None, // PATCHABLE_FUNCTION_ENTER = 32
    CEFBS_None, // PATCHABLE_RET = 33
    CEFBS_None, // PATCHABLE_FUNCTION_EXIT = 34
    CEFBS_None, // PATCHABLE_TAIL_CALL = 35
    CEFBS_None, // PATCHABLE_EVENT_CALL = 36
    CEFBS_None, // PATCHABLE_TYPED_EVENT_CALL = 37
    CEFBS_None, // ICALL_BRANCH_FUNNEL = 38
    CEFBS_None, // G_ADD = 39
    CEFBS_None, // G_SUB = 40
    CEFBS_None, // G_MUL = 41
    CEFBS_None, // G_SDIV = 42
    CEFBS_None, // G_UDIV = 43
    CEFBS_None, // G_SREM = 44
    CEFBS_None, // G_UREM = 45
    CEFBS_None, // G_AND = 46
    CEFBS_None, // G_OR = 47
    CEFBS_None, // G_XOR = 48
    CEFBS_None, // G_IMPLICIT_DEF = 49
    CEFBS_None, // G_PHI = 50
    CEFBS_None, // G_FRAME_INDEX = 51
    CEFBS_None, // G_GLOBAL_VALUE = 52
    CEFBS_None, // G_EXTRACT = 53
    CEFBS_None, // G_UNMERGE_VALUES = 54
    CEFBS_None, // G_INSERT = 55
    CEFBS_None, // G_MERGE_VALUES = 56
    CEFBS_None, // G_BUILD_VECTOR = 57
    CEFBS_None, // G_BUILD_VECTOR_TRUNC = 58
    CEFBS_None, // G_CONCAT_VECTORS = 59
    CEFBS_None, // G_PTRTOINT = 60
    CEFBS_None, // G_INTTOPTR = 61
    CEFBS_None, // G_BITCAST = 62
    CEFBS_None, // G_FREEZE = 63
    CEFBS_None, // G_INTRINSIC_TRUNC = 64
    CEFBS_None, // G_INTRINSIC_ROUND = 65
    CEFBS_None, // G_INTRINSIC_LRINT = 66
    CEFBS_None, // G_INTRINSIC_ROUNDEVEN = 67
    CEFBS_None, // G_READCYCLECOUNTER = 68
    CEFBS_None, // G_LOAD = 69
    CEFBS_None, // G_SEXTLOAD = 70
    CEFBS_None, // G_ZEXTLOAD = 71
    CEFBS_None, // G_INDEXED_LOAD = 72
    CEFBS_None, // G_INDEXED_SEXTLOAD = 73
    CEFBS_None, // G_INDEXED_ZEXTLOAD = 74
    CEFBS_None, // G_STORE = 75
    CEFBS_None, // G_INDEXED_STORE = 76
    CEFBS_None, // G_ATOMIC_CMPXCHG_WITH_SUCCESS = 77
    CEFBS_None, // G_ATOMIC_CMPXCHG = 78
    CEFBS_None, // G_ATOMICRMW_XCHG = 79
    CEFBS_None, // G_ATOMICRMW_ADD = 80
    CEFBS_None, // G_ATOMICRMW_SUB = 81
    CEFBS_None, // G_ATOMICRMW_AND = 82
    CEFBS_None, // G_ATOMICRMW_NAND = 83
    CEFBS_None, // G_ATOMICRMW_OR = 84
    CEFBS_None, // G_ATOMICRMW_XOR = 85
    CEFBS_None, // G_ATOMICRMW_MAX = 86
    CEFBS_None, // G_ATOMICRMW_MIN = 87
    CEFBS_None, // G_ATOMICRMW_UMAX = 88
    CEFBS_None, // G_ATOMICRMW_UMIN = 89
    CEFBS_None, // G_ATOMICRMW_FADD = 90
    CEFBS_None, // G_ATOMICRMW_FSUB = 91
    CEFBS_None, // G_FENCE = 92
    CEFBS_None, // G_BRCOND = 93
    CEFBS_None, // G_BRINDIRECT = 94
    CEFBS_None, // G_INTRINSIC = 95
    CEFBS_None, // G_INTRINSIC_W_SIDE_EFFECTS = 96
    CEFBS_None, // G_ANYEXT = 97
    CEFBS_None, // G_TRUNC = 98
    CEFBS_None, // G_CONSTANT = 99
    CEFBS_None, // G_FCONSTANT = 100
    CEFBS_None, // G_VASTART = 101
    CEFBS_None, // G_VAARG = 102
    CEFBS_None, // G_SEXT = 103
    CEFBS_None, // G_SEXT_INREG = 104
    CEFBS_None, // G_ZEXT = 105
    CEFBS_None, // G_SHL = 106
    CEFBS_None, // G_LSHR = 107
    CEFBS_None, // G_ASHR = 108
    CEFBS_None, // G_FSHL = 109
    CEFBS_None, // G_FSHR = 110
    CEFBS_None, // G_ICMP = 111
    CEFBS_None, // G_FCMP = 112
    CEFBS_None, // G_SELECT = 113
    CEFBS_None, // G_UADDO = 114
    CEFBS_None, // G_UADDE = 115
    CEFBS_None, // G_USUBO = 116
    CEFBS_None, // G_USUBE = 117
    CEFBS_None, // G_SADDO = 118
    CEFBS_None, // G_SADDE = 119
    CEFBS_None, // G_SSUBO = 120
    CEFBS_None, // G_SSUBE = 121
    CEFBS_None, // G_UMULO = 122
    CEFBS_None, // G_SMULO = 123
    CEFBS_None, // G_UMULH = 124
    CEFBS_None, // G_SMULH = 125
    CEFBS_None, // G_UADDSAT = 126
    CEFBS_None, // G_SADDSAT = 127
    CEFBS_None, // G_USUBSAT = 128
    CEFBS_None, // G_SSUBSAT = 129
    CEFBS_None, // G_USHLSAT = 130
    CEFBS_None, // G_SSHLSAT = 131
    CEFBS_None, // G_SMULFIX = 132
    CEFBS_None, // G_UMULFIX = 133
    CEFBS_None, // G_SMULFIXSAT = 134
    CEFBS_None, // G_UMULFIXSAT = 135
    CEFBS_None, // G_SDIVFIX = 136
    CEFBS_None, // G_UDIVFIX = 137
    CEFBS_None, // G_SDIVFIXSAT = 138
    CEFBS_None, // G_UDIVFIXSAT = 139
    CEFBS_None, // G_FADD = 140
    CEFBS_None, // G_FSUB = 141
    CEFBS_None, // G_FMUL = 142
    CEFBS_None, // G_FMA = 143
    CEFBS_None, // G_FMAD = 144
    CEFBS_None, // G_FDIV = 145
    CEFBS_None, // G_FREM = 146
    CEFBS_None, // G_FPOW = 147
    CEFBS_None, // G_FPOWI = 148
    CEFBS_None, // G_FEXP = 149
    CEFBS_None, // G_FEXP2 = 150
    CEFBS_None, // G_FLOG = 151
    CEFBS_None, // G_FLOG2 = 152
    CEFBS_None, // G_FLOG10 = 153
    CEFBS_None, // G_FNEG = 154
    CEFBS_None, // G_FPEXT = 155
    CEFBS_None, // G_FPTRUNC = 156
    CEFBS_None, // G_FPTOSI = 157
    CEFBS_None, // G_FPTOUI = 158
    CEFBS_None, // G_SITOFP = 159
    CEFBS_None, // G_UITOFP = 160
    CEFBS_None, // G_FABS = 161
    CEFBS_None, // G_FCOPYSIGN = 162
    CEFBS_None, // G_FCANONICALIZE = 163
    CEFBS_None, // G_FMINNUM = 164
    CEFBS_None, // G_FMAXNUM = 165
    CEFBS_None, // G_FMINNUM_IEEE = 166
    CEFBS_None, // G_FMAXNUM_IEEE = 167
    CEFBS_None, // G_FMINIMUM = 168
    CEFBS_None, // G_FMAXIMUM = 169
    CEFBS_None, // G_PTR_ADD = 170
    CEFBS_None, // G_PTRMASK = 171
    CEFBS_None, // G_SMIN = 172
    CEFBS_None, // G_SMAX = 173
    CEFBS_None, // G_UMIN = 174
    CEFBS_None, // G_UMAX = 175
    CEFBS_None, // G_ABS = 176
    CEFBS_None, // G_BR = 177
    CEFBS_None, // G_BRJT = 178
    CEFBS_None, // G_INSERT_VECTOR_ELT = 179
    CEFBS_None, // G_EXTRACT_VECTOR_ELT = 180
    CEFBS_None, // G_SHUFFLE_VECTOR = 181
    CEFBS_None, // G_CTTZ = 182
    CEFBS_None, // G_CTTZ_ZERO_UNDEF = 183
    CEFBS_None, // G_CTLZ = 184
    CEFBS_None, // G_CTLZ_ZERO_UNDEF = 185
    CEFBS_None, // G_CTPOP = 186
    CEFBS_None, // G_BSWAP = 187
    CEFBS_None, // G_BITREVERSE = 188
    CEFBS_None, // G_FCEIL = 189
    CEFBS_None, // G_FCOS = 190
    CEFBS_None, // G_FSIN = 191
    CEFBS_None, // G_FSQRT = 192
    CEFBS_None, // G_FFLOOR = 193
    CEFBS_None, // G_FRINT = 194
    CEFBS_None, // G_FNEARBYINT = 195
    CEFBS_None, // G_ADDRSPACE_CAST = 196
    CEFBS_None, // G_BLOCK_ADDR = 197
    CEFBS_None, // G_JUMP_TABLE = 198
    CEFBS_None, // G_DYN_STACKALLOC = 199
    CEFBS_None, // G_STRICT_FADD = 200
    CEFBS_None, // G_STRICT_FSUB = 201
    CEFBS_None, // G_STRICT_FMUL = 202
    CEFBS_None, // G_STRICT_FDIV = 203
    CEFBS_None, // G_STRICT_FREM = 204
    CEFBS_None, // G_STRICT_FMA = 205
    CEFBS_None, // G_STRICT_FSQRT = 206
    CEFBS_None, // G_READ_REGISTER = 207
    CEFBS_None, // G_WRITE_REGISTER = 208
    CEFBS_None, // G_MEMCPY = 209
    CEFBS_None, // G_MEMMOVE = 210
    CEFBS_None, // G_MEMSET = 211
    CEFBS_None, // G_VECREDUCE_SEQ_FADD = 212
    CEFBS_None, // G_VECREDUCE_SEQ_FMUL = 213
    CEFBS_None, // G_VECREDUCE_FADD = 214
    CEFBS_None, // G_VECREDUCE_FMUL = 215
    CEFBS_None, // G_VECREDUCE_FMAX = 216
    CEFBS_None, // G_VECREDUCE_FMIN = 217
    CEFBS_None, // G_VECREDUCE_ADD = 218
    CEFBS_None, // G_VECREDUCE_MUL = 219
    CEFBS_None, // G_VECREDUCE_AND = 220
    CEFBS_None, // G_VECREDUCE_OR = 221
    CEFBS_None, // G_VECREDUCE_XOR = 222
    CEFBS_None, // G_VECREDUCE_SMAX = 223
    CEFBS_None, // G_VECREDUCE_SMIN = 224
    CEFBS_None, // G_VECREDUCE_UMAX = 225
    CEFBS_None, // G_VECREDUCE_UMIN = 226
    CEFBS_Ch9_2, // ADJCALLSTACKDOWN = 227
    CEFBS_Ch9_2, // ADJCALLSTACKUP = 228
    CEFBS_Ch12_1, // ATOMIC_CMP_SWAP_I16 = 229
    CEFBS_Ch12_1, // ATOMIC_CMP_SWAP_I32 = 230
    CEFBS_Ch12_1, // ATOMIC_CMP_SWAP_I8 = 231
    CEFBS_Ch12_1, // ATOMIC_LOAD_ADD_I16 = 232
    CEFBS_Ch12_1, // ATOMIC_LOAD_ADD_I32 = 233
    CEFBS_Ch12_1, // ATOMIC_LOAD_ADD_I8 = 234
    CEFBS_Ch12_1, // ATOMIC_LOAD_AND_I16 = 235
    CEFBS_Ch12_1, // ATOMIC_LOAD_AND_I32 = 236
    CEFBS_Ch12_1, // ATOMIC_LOAD_AND_I8 = 237
    CEFBS_Ch12_1, // ATOMIC_LOAD_NAND_I16 = 238
    CEFBS_Ch12_1, // ATOMIC_LOAD_NAND_I32 = 239
    CEFBS_Ch12_1, // ATOMIC_LOAD_NAND_I8 = 240
    CEFBS_Ch12_1, // ATOMIC_LOAD_OR_I16 = 241
    CEFBS_Ch12_1, // ATOMIC_LOAD_OR_I32 = 242
    CEFBS_Ch12_1, // ATOMIC_LOAD_OR_I8 = 243
    CEFBS_Ch12_1, // ATOMIC_LOAD_SUB_I16 = 244
    CEFBS_Ch12_1, // ATOMIC_LOAD_SUB_I32 = 245
    CEFBS_Ch12_1, // ATOMIC_LOAD_SUB_I8 = 246
    CEFBS_Ch12_1, // ATOMIC_LOAD_XOR_I16 = 247
    CEFBS_Ch12_1, // ATOMIC_LOAD_XOR_I32 = 248
    CEFBS_Ch12_1, // ATOMIC_LOAD_XOR_I8 = 249
    CEFBS_Ch12_1, // ATOMIC_SWAP_I16 = 250
    CEFBS_Ch12_1, // ATOMIC_SWAP_I32 = 251
    CEFBS_Ch12_1, // ATOMIC_SWAP_I8 = 252
    CEFBS_Ch9_2, // CPRESTORE = 253
    CEFBS_None, // CPU0eh_return32 = 254
    CEFBS_Ch8_2, // LONG_BRANCH_ADDiu = 255
    CEFBS_Ch8_2, // LONG_BRANCH_LUi = 256
    CEFBS_Ch11_1, // LoadAddr32Imm = 257
    CEFBS_Ch11_1, // LoadAddr32Reg = 258
    CEFBS_Ch11_1, // LoadImm32Reg = 259
    CEFBS_Ch3_4, // RetLR = 260
    CEFBS_Ch9_1, // TAILCALL = 261
    CEFBS_Ch9_1, // TAILCALL_R = 262
    CEFBS_None, // ADD = 263
    CEFBS_None, // ADDiu = 264
    CEFBS_None, // ADDu = 265
    CEFBS_Ch4_2, // AND = 266
    CEFBS_Ch4_2, // ANDi = 267
    CEFBS_None, // BAL = 268
    CEFBS_None, // BEQ = 269
    CEFBS_None, // BNE = 270
    CEFBS_Ch4_1, // C0MOVE = 271
    CEFBS_Ch7_1, // CLO = 272
    CEFBS_Ch7_1, // CLZ = 273
    CEFBS_None, // CMP = 274
    CEFBS_None, // CMPu = 275
    CEFBS_None, // GCD = 276
    CEFBS_Ch9_1, // JALR = 277
    CEFBS_None, // JEQ = 278
    CEFBS_None, // JGE = 279
    CEFBS_None, // JGT = 280
    CEFBS_None, // JLE = 281
    CEFBS_None, // JLT = 282
    CEFBS_Ch8_1, // JMP = 283
    CEFBS_None, // JNE = 284
    CEFBS_None, // JR = 285
    CEFBS_Ch9_1, // JSUB = 286
    CEFBS_Ch7_1, // LB = 287
    CEFBS_Ch7_1, // LBu = 288
    CEFBS_None, // LD = 289
    CEFBS_Ch7_1, // LEA_ADDiu = 290
    CEFBS_Ch7_1, // LH = 291
    CEFBS_Ch7_1, // LHu = 292
    CEFBS_None, // LL = 293
    CEFBS_Ch3_5, // LUi = 294
    CEFBS_Ch4_1, // MFC0 = 295
    CEFBS_Ch4_1, // MFHI = 296
    CEFBS_Ch4_1, // MFLO = 297
    CEFBS_Ch8_2, // MOVN_I_I = 298
    CEFBS_Ch8_2, // MOVZ_I_I = 299
    CEFBS_Ch4_1, // MTC0 = 300
    CEFBS_Ch4_1, // MTHI = 301
    CEFBS_Ch4_1, // MTLO = 302
    CEFBS_Ch4_1, // MUL = 303
    CEFBS_Ch4_1, // MULT = 304
    CEFBS_Ch4_1, // MULTu = 305
    CEFBS_None, // NOP = 306
    CEFBS_Ch4_2, // NOR = 307
    CEFBS_Ch4_2, // OR = 308
    CEFBS_Ch3_5, // ORi = 309
    CEFBS_None, // RET = 310
    CEFBS_Ch4_1, // ROL = 311
    CEFBS_Ch4_1, // ROLV = 312
    CEFBS_Ch4_1, // ROR = 313
    CEFBS_Ch4_1, // RORV = 314
    CEFBS_Ch7_1, // SB = 315
    CEFBS_None, // SC = 316
    CEFBS_Ch4_1, // SDIV = 317
    CEFBS_Ch7_1, // SH = 318
    CEFBS_Ch3_5, // SHL = 319
    CEFBS_Ch4_1, // SHLV = 320
    CEFBS_Ch4_1, // SHR = 321
    CEFBS_Ch4_1, // SHRV = 322
    CEFBS_None, // SLT = 323
    CEFBS_None, // SLTi = 324
    CEFBS_None, // SLTiu = 325
    CEFBS_None, // SLTu = 326
    CEFBS_Ch4_1, // SRA = 327
    CEFBS_Ch4_1, // SRAV = 328
    CEFBS_None, // ST = 329
    CEFBS_None, // SUB = 330
    CEFBS_None, // SUBu = 331
    CEFBS_Ch12_1, // SYNC = 332
    CEFBS_Ch4_1, // UDIV = 333
    CEFBS_Ch4_2, // XOR = 334
    CEFBS_Ch4_2, // XORi = 335
  };

  assert(Inst.getOpcode() < 336);
  const FeatureBitset &RequiredFeatures = FeatureBitsets[RequiredFeaturesRefs[Inst.getOpcode()]];
  FeatureBitset MissingFeatures =
      (AvailableFeatures & RequiredFeatures) ^
      RequiredFeatures;
  if (MissingFeatures.any()) {
    std::ostringstream Msg;
    Msg << "Attempting to emit " << MCII.getName(Inst.getOpcode()).str()
        << " instruction but the ";
    for (unsigned i = 0, e = MissingFeatures.size(); i != e; ++i)
      if (MissingFeatures.test(i))
        Msg << SubtargetFeatureNames[i] << " ";
    Msg << "predicate(s) are not met";
    report_fatal_error(Msg.str());
  }
#else
  // Silence unused variable warning on targets that don't use MCII for other purposes (e.g. BPF).
  (void)MCII;
#endif // NDEBUG
}
#endif
