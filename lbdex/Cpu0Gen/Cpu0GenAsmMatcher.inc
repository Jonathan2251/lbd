/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Assembly Matcher Source Fragment                                           *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


#ifdef GET_ASSEMBLER_HEADER
#undef GET_ASSEMBLER_HEADER
  // This should be included into the middle of the declaration of
  // your subclasses implementation of MCTargetAsmParser.
  FeatureBitset ComputeAvailableFeatures(const FeatureBitset &FB) const;
  void convertToMCInst(unsigned Kind, MCInst &Inst, unsigned Opcode,
                       const OperandVector &Operands);
  void convertToMapAndConstraints(unsigned Kind,
                           const OperandVector &Operands) override;
  unsigned MatchInstructionImpl(const OperandVector &Operands,
                                MCInst &Inst,
                                uint64_t &ErrorInfo,
                                FeatureBitset &MissingFeatures,
                                bool matchingInlineAsm,
                                unsigned VariantID = 0);
  unsigned MatchInstructionImpl(const OperandVector &Operands,
                                MCInst &Inst,
                                uint64_t &ErrorInfo,
                                bool matchingInlineAsm,
                                unsigned VariantID = 0) {
    FeatureBitset MissingFeatures;
    return MatchInstructionImpl(Operands, Inst, ErrorInfo, MissingFeatures,
                                matchingInlineAsm, VariantID);
  }

  OperandMatchResultTy MatchOperandParserImpl(
    OperandVector &Operands,
    StringRef Mnemonic,
    bool ParseForAllFeatures = false);
  OperandMatchResultTy tryCustomParseOperand(
    OperandVector &Operands,
    unsigned MCK);

#endif // GET_ASSEMBLER_HEADER_INFO


#ifdef GET_OPERAND_DIAGNOSTIC_TYPES
#undef GET_OPERAND_DIAGNOSTIC_TYPES

#endif // GET_OPERAND_DIAGNOSTIC_TYPES


#ifdef GET_REGISTER_MATCHER
#undef GET_REGISTER_MATCHER

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

#endif // GET_REGISTER_MATCHER


#ifdef GET_SUBTARGET_FEATURE_NAME
#undef GET_SUBTARGET_FEATURE_NAME

// User-level names for subtarget features that participate in
// instruction matching.
static const char *getSubtargetFeatureName(uint64_t Val) {
  switch(Val) {
  case Feature_Ch3_1Bit: return "";
  case Feature_Ch3_2Bit: return "";
  case Feature_Ch3_3Bit: return "";
  case Feature_Ch3_4Bit: return "";
  case Feature_Ch3_5Bit: return "";
  case Feature_Ch4_1Bit: return "";
  case Feature_Ch4_2Bit: return "";
  case Feature_Ch5_1Bit: return "";
  case Feature_Ch6_1Bit: return "";
  case Feature_Ch7_1Bit: return "";
  case Feature_Ch8_1Bit: return "";
  case Feature_Ch8_2Bit: return "";
  case Feature_Ch9_1Bit: return "";
  case Feature_Ch9_2Bit: return "";
  case Feature_Ch9_3Bit: return "";
  case Feature_Ch10_1Bit: return "";
  case Feature_Ch11_1Bit: return "";
  case Feature_Ch11_2Bit: return "";
  case Feature_Ch12_1Bit: return "";
  case Feature_Ch_allBit: return "";
  default: return "(unknown)";
  }
}

#endif // GET_SUBTARGET_FEATURE_NAME


#ifdef GET_MATCHER_IMPLEMENTATION
#undef GET_MATCHER_IMPLEMENTATION

enum {
  Tie0_1_1,
};

static const uint8_t TiedAsmOperandTable[][3] = {
  /* Tie0_1_1 */ { 0, 1, 1 },
};

namespace {
enum OperatorConversionKind {
  CVT_Done,
  CVT_Reg,
  CVT_Tied,
  CVT_95_Reg,
  CVT_95_addImmOperands,
  CVT_95_addMemOperands,
  CVT_regZERO,
  CVT_imm_95_0,
  CVT_NUM_CONVERTERS
};

enum InstructionConversionKind {
  Convert__Reg1_0__Reg1_1__Reg1_2,
  Convert__Reg1_0__Reg1_1__Imm1_2,
  Convert__Imm1_0,
  Convert__Reg1_0__Reg1_1,
  Convert__Reg1_0,
  Convert__Reg1_0__Imm1_1,
  Convert__Reg1_0__Mem2_1,
  Convert__Reg1_0__Reg1_1__regZERO,
  Convert__Reg1_0__Reg1_1__Reg1_2__Tie0_1_1,
  Convert_NoOperands,
  Convert__Reg1_0__Tie0_1_1__Mem2_1,
  Convert__imm_95_0,
  CVT_NUM_SIGNATURES
};

} // end anonymous namespace

static const uint8_t ConversionTable[CVT_NUM_SIGNATURES][9] = {
  // Convert__Reg1_0__Reg1_1__Reg1_2
  { CVT_95_Reg, 1, CVT_95_Reg, 2, CVT_95_Reg, 3, CVT_Done },
  // Convert__Reg1_0__Reg1_1__Imm1_2
  { CVT_95_Reg, 1, CVT_95_Reg, 2, CVT_95_addImmOperands, 3, CVT_Done },
  // Convert__Imm1_0
  { CVT_95_addImmOperands, 1, CVT_Done },
  // Convert__Reg1_0__Reg1_1
  { CVT_95_Reg, 1, CVT_95_Reg, 2, CVT_Done },
  // Convert__Reg1_0
  { CVT_95_Reg, 1, CVT_Done },
  // Convert__Reg1_0__Imm1_1
  { CVT_95_Reg, 1, CVT_95_addImmOperands, 2, CVT_Done },
  // Convert__Reg1_0__Mem2_1
  { CVT_95_Reg, 1, CVT_95_addMemOperands, 2, CVT_Done },
  // Convert__Reg1_0__Reg1_1__regZERO
  { CVT_95_Reg, 1, CVT_95_Reg, 2, CVT_regZERO, 0, CVT_Done },
  // Convert__Reg1_0__Reg1_1__Reg1_2__Tie0_1_1
  { CVT_95_Reg, 1, CVT_95_Reg, 2, CVT_95_Reg, 3, CVT_Tied, Tie0_1_1, CVT_Done },
  // Convert_NoOperands
  { CVT_Done },
  // Convert__Reg1_0__Tie0_1_1__Mem2_1
  { CVT_95_Reg, 1, CVT_Tied, Tie0_1_1, CVT_95_addMemOperands, 2, CVT_Done },
  // Convert__imm_95_0
  { CVT_imm_95_0, 0, CVT_Done },
};

void Cpu0AsmParser::
convertToMCInst(unsigned Kind, MCInst &Inst, unsigned Opcode,
                const OperandVector &Operands) {
  assert(Kind < CVT_NUM_SIGNATURES && "Invalid signature!");
  const uint8_t *Converter = ConversionTable[Kind];
  unsigned OpIdx;
  Inst.setOpcode(Opcode);
  for (const uint8_t *p = Converter; *p; p += 2) {
    OpIdx = *(p + 1);
    switch (*p) {
    default: llvm_unreachable("invalid conversion entry!");
    case CVT_Reg:
      static_cast<Cpu0Operand &>(*Operands[OpIdx]).addRegOperands(Inst, 1);
      break;
    case CVT_Tied: {
      assert(OpIdx < (size_t)(std::end(TiedAsmOperandTable) -
                              std::begin(TiedAsmOperandTable)) &&
             "Tied operand not found");
      unsigned TiedResOpnd = TiedAsmOperandTable[OpIdx][0];
      if (TiedResOpnd != (uint8_t)-1)
        Inst.addOperand(Inst.getOperand(TiedResOpnd));
      break;
    }
    case CVT_95_Reg:
      static_cast<Cpu0Operand &>(*Operands[OpIdx]).addRegOperands(Inst, 1);
      break;
    case CVT_95_addImmOperands:
      static_cast<Cpu0Operand &>(*Operands[OpIdx]).addImmOperands(Inst, 1);
      break;
    case CVT_95_addMemOperands:
      static_cast<Cpu0Operand &>(*Operands[OpIdx]).addMemOperands(Inst, 2);
      break;
    case CVT_regZERO:
      Inst.addOperand(MCOperand::createReg(Cpu0::ZERO));
      break;
    case CVT_imm_95_0:
      Inst.addOperand(MCOperand::createImm(0));
      break;
    }
  }
}

void Cpu0AsmParser::
convertToMapAndConstraints(unsigned Kind,
                           const OperandVector &Operands) {
  assert(Kind < CVT_NUM_SIGNATURES && "Invalid signature!");
  unsigned NumMCOperands = 0;
  const uint8_t *Converter = ConversionTable[Kind];
  for (const uint8_t *p = Converter; *p; p += 2) {
    switch (*p) {
    default: llvm_unreachable("invalid conversion entry!");
    case CVT_Reg:
      Operands[*(p + 1)]->setMCOperandNum(NumMCOperands);
      Operands[*(p + 1)]->setConstraint("r");
      ++NumMCOperands;
      break;
    case CVT_Tied:
      ++NumMCOperands;
      break;
    case CVT_95_Reg:
      Operands[*(p + 1)]->setMCOperandNum(NumMCOperands);
      Operands[*(p + 1)]->setConstraint("r");
      NumMCOperands += 1;
      break;
    case CVT_95_addImmOperands:
      Operands[*(p + 1)]->setMCOperandNum(NumMCOperands);
      Operands[*(p + 1)]->setConstraint("m");
      NumMCOperands += 1;
      break;
    case CVT_95_addMemOperands:
      Operands[*(p + 1)]->setMCOperandNum(NumMCOperands);
      Operands[*(p + 1)]->setConstraint("m");
      NumMCOperands += 2;
      break;
    case CVT_regZERO:
      Operands[*(p + 1)]->setMCOperandNum(NumMCOperands);
      Operands[*(p + 1)]->setConstraint("m");
      ++NumMCOperands;
      break;
    case CVT_imm_95_0:
      Operands[*(p + 1)]->setMCOperandNum(NumMCOperands);
      Operands[*(p + 1)]->setConstraint("");
      ++NumMCOperands;
      break;
    }
  }
}

namespace {

/// MatchClassKind - The kinds of classes which participate in
/// instruction matching.
enum MatchClassKind {
  InvalidMatchClass = 0,
  OptionalMatchClass = 1,
  MCK_LAST_TOKEN = OptionalMatchClass,
  MCK_SR, // register class 'SR'
  MCK_C0Regs, // register class 'C0Regs'
  MCK_HILO, // register class 'HILO'
  MCK_CPURegs, // register class 'CPURegs,GPROut'
  MCK_LAST_REGISTER = MCK_CPURegs,
  MCK_Mem, // user defined class 'Cpu0MemAsmOperand'
  MCK_Imm, // user defined class 'ImmAsmOperand'
  NumMatchClassKinds
};

} // end anonymous namespace

static unsigned getDiagKindFromRegisterClass(MatchClassKind RegisterClass) {
  return MCTargetAsmParser::Match_InvalidOperand;
}

static MatchClassKind matchTokenString(StringRef Name) {
  return InvalidMatchClass;
}

/// isSubclass - Compute whether \p A is a subclass of \p B.
static bool isSubclass(MatchClassKind A, MatchClassKind B) {
  if (A == B)
    return true;

  switch (A) {
  default:
    return false;

  case MCK_SR:
    return B == MCK_CPURegs;
  }
}

static unsigned validateOperandClass(MCParsedAsmOperand &GOp, MatchClassKind Kind) {
  Cpu0Operand &Operand = (Cpu0Operand &)GOp;
  if (Kind == InvalidMatchClass)
    return MCTargetAsmParser::Match_InvalidOperand;

  if (Operand.isToken() && Kind <= MCK_LAST_TOKEN)
    return isSubclass(matchTokenString(Operand.getToken()), Kind) ?
             MCTargetAsmParser::Match_Success :
             MCTargetAsmParser::Match_InvalidOperand;

  switch (Kind) {
  default: break;
  // 'Mem' class
  case MCK_Mem: {
    DiagnosticPredicate DP(Operand.isMem());
    if (DP.isMatch())
      return MCTargetAsmParser::Match_Success;
    break;
    }
  // 'Imm' class
  case MCK_Imm: {
    DiagnosticPredicate DP(Operand.isImm());
    if (DP.isMatch())
      return MCTargetAsmParser::Match_Success;
    break;
    }
  } // end switch (Kind)

  if (Operand.isReg()) {
    MatchClassKind OpKind;
    switch (Operand.getReg()) {
    default: OpKind = InvalidMatchClass; break;
    case Cpu0::ZERO: OpKind = MCK_CPURegs; break;
    case Cpu0::AT: OpKind = MCK_CPURegs; break;
    case Cpu0::V0: OpKind = MCK_CPURegs; break;
    case Cpu0::V1: OpKind = MCK_CPURegs; break;
    case Cpu0::A0: OpKind = MCK_CPURegs; break;
    case Cpu0::A1: OpKind = MCK_CPURegs; break;
    case Cpu0::T9: OpKind = MCK_CPURegs; break;
    case Cpu0::T0: OpKind = MCK_CPURegs; break;
    case Cpu0::T1: OpKind = MCK_CPURegs; break;
    case Cpu0::S0: OpKind = MCK_CPURegs; break;
    case Cpu0::S1: OpKind = MCK_CPURegs; break;
    case Cpu0::GP: OpKind = MCK_CPURegs; break;
    case Cpu0::FP: OpKind = MCK_CPURegs; break;
    case Cpu0::SP: OpKind = MCK_CPURegs; break;
    case Cpu0::LR: OpKind = MCK_CPURegs; break;
    case Cpu0::SW: OpKind = MCK_SR; break;
    case Cpu0::HI: OpKind = MCK_HILO; break;
    case Cpu0::LO: OpKind = MCK_HILO; break;
    case Cpu0::PC: OpKind = MCK_C0Regs; break;
    case Cpu0::EPC: OpKind = MCK_C0Regs; break;
    }
    return isSubclass(OpKind, Kind) ? (unsigned)MCTargetAsmParser::Match_Success :
                                      getDiagKindFromRegisterClass(Kind);
  }

  if (Kind > MCK_LAST_TOKEN && Kind <= MCK_LAST_REGISTER)
    return getDiagKindFromRegisterClass(Kind);

  return MCTargetAsmParser::Match_InvalidOperand;
}

#ifndef NDEBUG
const char *getMatchClassName(MatchClassKind Kind) {
  switch (Kind) {
  case InvalidMatchClass: return "InvalidMatchClass";
  case OptionalMatchClass: return "OptionalMatchClass";
  case MCK_SR: return "MCK_SR";
  case MCK_C0Regs: return "MCK_C0Regs";
  case MCK_HILO: return "MCK_HILO";
  case MCK_CPURegs: return "MCK_CPURegs";
  case MCK_Mem: return "MCK_Mem";
  case MCK_Imm: return "MCK_Imm";
  case NumMatchClassKinds: return "NumMatchClassKinds";
  }
  llvm_unreachable("unhandled MatchClassKind!");
}

#endif // NDEBUG
FeatureBitset Cpu0AsmParser::
ComputeAvailableFeatures(const FeatureBitset &FB) const {
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

static bool checkAsmTiedOperandConstraints(const Cpu0AsmParser&AsmParser,
                               unsigned Kind,
                               const OperandVector &Operands,
                               uint64_t &ErrorInfo) {
  assert(Kind < CVT_NUM_SIGNATURES && "Invalid signature!");
  const uint8_t *Converter = ConversionTable[Kind];
  for (const uint8_t *p = Converter; *p; p += 2) {
    switch (*p) {
    case CVT_Tied: {
      unsigned OpIdx = *(p + 1);
      assert(OpIdx < (size_t)(std::end(TiedAsmOperandTable) -
                              std::begin(TiedAsmOperandTable)) &&
             "Tied operand not found");
      unsigned OpndNum1 = TiedAsmOperandTable[OpIdx][1];
      unsigned OpndNum2 = TiedAsmOperandTable[OpIdx][2];
      if (OpndNum1 != OpndNum2) {
        auto &SrcOp1 = Operands[OpndNum1];
        auto &SrcOp2 = Operands[OpndNum2];
        if (SrcOp1->isReg() && SrcOp2->isReg()) {
          if (!AsmParser.regsEqual(*SrcOp1, *SrcOp2)) {
            ErrorInfo = OpndNum2;
            return false;
          }
        }
      }
      break;
    }
    default:
      break;
    }
  }
  return true;
}

static const char *const MnemonicTable =
    "\003add\005addiu\004addu\003and\004andi\003bal\003beq\003bne\005c0mov\003"
    "clo\003clz\003cmp\004cmpu\003div\004divu\003gcd\004jalr\003jeq\003jge\003"
    "jgt\003jle\003jlt\003jmp\003jne\002jr\004jsub\002la\002lb\003lbu\002ld\002"
    "lh\003lhu\002li\002ll\003lui\004mfc0\004mfhi\004mflo\004move\004movn\004"
    "movz\004mtc0\004mthi\004mtlo\003mul\004mult\005multu\003nop\003nor\002o"
    "r\003ori\003rol\004rolv\003ror\004rorv\002sb\002sc\002sh\003shl\004shlv"
    "\003shr\004shrv\003slt\004slti\005sltiu\004sltu\003sra\004srav\002st\003"
    "sub\004subu\004sync\003xor\004xori";

// Feature bitsets.
enum : uint8_t {
  AMFBS_None,
  AMFBS_Ch11_1,
  AMFBS_Ch12_1,
  AMFBS_Ch3_5,
  AMFBS_Ch4_1,
  AMFBS_Ch4_2,
  AMFBS_Ch7_1,
  AMFBS_Ch8_1,
  AMFBS_Ch8_2,
  AMFBS_Ch9_1,
};

static constexpr FeatureBitset FeatureBitsets[] = {
  {}, // AMFBS_None
  {Feature_Ch11_1Bit, },
  {Feature_Ch12_1Bit, },
  {Feature_Ch3_5Bit, },
  {Feature_Ch4_1Bit, },
  {Feature_Ch4_2Bit, },
  {Feature_Ch7_1Bit, },
  {Feature_Ch8_1Bit, },
  {Feature_Ch8_2Bit, },
  {Feature_Ch9_1Bit, },
};

namespace {
  struct MatchEntry {
    uint16_t Mnemonic;
    uint16_t Opcode;
    uint8_t ConvertFn;
    uint8_t RequiredFeaturesIdx;
    uint8_t Classes[3];
    StringRef getMnemonic() const {
      return StringRef(MnemonicTable + Mnemonic + 1,
                       MnemonicTable[Mnemonic]);
    }
  };

  // Predicate for searching for an opcode.
  struct LessOpcode {
    bool operator()(const MatchEntry &LHS, StringRef RHS) {
      return LHS.getMnemonic() < RHS;
    }
    bool operator()(StringRef LHS, const MatchEntry &RHS) {
      return LHS < RHS.getMnemonic();
    }
    bool operator()(const MatchEntry &LHS, const MatchEntry &RHS) {
      return LHS.getMnemonic() < RHS.getMnemonic();
    }
  };
} // end anonymous namespace

static const MatchEntry MatchTable0[] = {
  { 0 /* add */, Cpu0::ADD, Convert__Reg1_0__Reg1_1__Reg1_2, AMFBS_None, { MCK_CPURegs, MCK_CPURegs, MCK_CPURegs }, },
  { 4 /* addiu */, Cpu0::ADDiu, Convert__Reg1_0__Reg1_1__Imm1_2, AMFBS_None, { MCK_CPURegs, MCK_CPURegs, MCK_Imm }, },
  { 10 /* addu */, Cpu0::ADDu, Convert__Reg1_0__Reg1_1__Reg1_2, AMFBS_None, { MCK_CPURegs, MCK_CPURegs, MCK_CPURegs }, },
  { 15 /* and */, Cpu0::AND, Convert__Reg1_0__Reg1_1__Reg1_2, AMFBS_Ch4_2, { MCK_CPURegs, MCK_CPURegs, MCK_CPURegs }, },
  { 19 /* andi */, Cpu0::ANDi, Convert__Reg1_0__Reg1_1__Imm1_2, AMFBS_Ch4_2, { MCK_CPURegs, MCK_CPURegs, MCK_Imm }, },
  { 24 /* bal */, Cpu0::BAL, Convert__Imm1_0, AMFBS_None, { MCK_Imm }, },
  { 28 /* beq */, Cpu0::BEQ, Convert__Reg1_0__Reg1_1__Imm1_2, AMFBS_None, { MCK_CPURegs, MCK_CPURegs, MCK_Imm }, },
  { 32 /* bne */, Cpu0::BNE, Convert__Reg1_0__Reg1_1__Imm1_2, AMFBS_None, { MCK_CPURegs, MCK_CPURegs, MCK_Imm }, },
  { 36 /* c0mov */, Cpu0::C0MOVE, Convert__Reg1_0__Reg1_1, AMFBS_Ch4_1, { MCK_C0Regs, MCK_C0Regs }, },
  { 42 /* clo */, Cpu0::CLO, Convert__Reg1_0__Reg1_1, AMFBS_Ch7_1, { MCK_CPURegs, MCK_CPURegs }, },
  { 46 /* clz */, Cpu0::CLZ, Convert__Reg1_0__Reg1_1, AMFBS_Ch7_1, { MCK_CPURegs, MCK_CPURegs }, },
  { 50 /* cmp */, Cpu0::CMP, Convert__Reg1_0__Reg1_1__Reg1_2, AMFBS_None, { MCK_SR, MCK_CPURegs, MCK_CPURegs }, },
  { 54 /* cmpu */, Cpu0::CMPu, Convert__Reg1_0__Reg1_1__Reg1_2, AMFBS_None, { MCK_SR, MCK_CPURegs, MCK_CPURegs }, },
  { 59 /* div */, Cpu0::SDIV, Convert__Reg1_0__Reg1_1, AMFBS_Ch4_1, { MCK_CPURegs, MCK_CPURegs }, },
  { 63 /* divu */, Cpu0::UDIV, Convert__Reg1_0__Reg1_1, AMFBS_Ch4_1, { MCK_CPURegs, MCK_CPURegs }, },
  { 68 /* gcd */, Cpu0::GCD, Convert__Reg1_0__Reg1_1__Reg1_2, AMFBS_None, { MCK_CPURegs, MCK_CPURegs, MCK_CPURegs }, },
  { 72 /* jalr */, Cpu0::JALR, Convert__Reg1_0, AMFBS_Ch9_1, { MCK_CPURegs }, },
  { 77 /* jeq */, Cpu0::JEQ, Convert__Reg1_0__Imm1_1, AMFBS_None, { MCK_SR, MCK_Imm }, },
  { 81 /* jge */, Cpu0::JGE, Convert__Reg1_0__Imm1_1, AMFBS_None, { MCK_SR, MCK_Imm }, },
  { 85 /* jgt */, Cpu0::JGT, Convert__Reg1_0__Imm1_1, AMFBS_None, { MCK_SR, MCK_Imm }, },
  { 89 /* jle */, Cpu0::JLE, Convert__Reg1_0__Imm1_1, AMFBS_None, { MCK_SR, MCK_Imm }, },
  { 93 /* jlt */, Cpu0::JLT, Convert__Reg1_0__Imm1_1, AMFBS_None, { MCK_SR, MCK_Imm }, },
  { 97 /* jmp */, Cpu0::JMP, Convert__Imm1_0, AMFBS_Ch8_1, { MCK_Imm }, },
  { 101 /* jne */, Cpu0::JNE, Convert__Reg1_0__Imm1_1, AMFBS_None, { MCK_SR, MCK_Imm }, },
  { 105 /* jr */, Cpu0::JR, Convert__Reg1_0, AMFBS_None, { MCK_CPURegs }, },
  { 108 /* jsub */, Cpu0::JSUB, Convert__Imm1_0, AMFBS_Ch9_1, { MCK_Imm }, },
  { 113 /* la */, Cpu0::LoadAddr32Reg, Convert__Reg1_0__Mem2_1, AMFBS_Ch11_1, { MCK_CPURegs, MCK_Mem }, },
  { 113 /* la */, Cpu0::LoadAddr32Imm, Convert__Reg1_0__Imm1_1, AMFBS_Ch11_1, { MCK_CPURegs, MCK_Imm }, },
  { 116 /* lb */, Cpu0::LB, Convert__Reg1_0__Mem2_1, AMFBS_Ch7_1, { MCK_CPURegs, MCK_Mem }, },
  { 119 /* lbu */, Cpu0::LBu, Convert__Reg1_0__Mem2_1, AMFBS_Ch7_1, { MCK_CPURegs, MCK_Mem }, },
  { 123 /* ld */, Cpu0::LD, Convert__Reg1_0__Mem2_1, AMFBS_None, { MCK_CPURegs, MCK_Mem }, },
  { 126 /* lh */, Cpu0::LH, Convert__Reg1_0__Mem2_1, AMFBS_Ch7_1, { MCK_CPURegs, MCK_Mem }, },
  { 129 /* lhu */, Cpu0::LHu, Convert__Reg1_0__Mem2_1, AMFBS_Ch7_1, { MCK_CPURegs, MCK_Mem }, },
  { 133 /* li */, Cpu0::LoadImm32Reg, Convert__Reg1_0__Imm1_1, AMFBS_Ch11_1, { MCK_CPURegs, MCK_Imm }, },
  { 136 /* ll */, Cpu0::LL, Convert__Reg1_0__Mem2_1, AMFBS_None, { MCK_CPURegs, MCK_Mem }, },
  { 139 /* lui */, Cpu0::LUi, Convert__Reg1_0__Imm1_1, AMFBS_Ch3_5, { MCK_CPURegs, MCK_Imm }, },
  { 143 /* mfc0 */, Cpu0::MFC0, Convert__Reg1_0__Reg1_1, AMFBS_Ch4_1, { MCK_CPURegs, MCK_C0Regs }, },
  { 148 /* mfhi */, Cpu0::MFHI, Convert__Reg1_0, AMFBS_Ch4_1, { MCK_CPURegs }, },
  { 153 /* mflo */, Cpu0::MFLO, Convert__Reg1_0, AMFBS_Ch4_1, { MCK_CPURegs }, },
  { 158 /* move */, Cpu0::ADDu, Convert__Reg1_0__Reg1_1__regZERO, AMFBS_Ch3_5, { MCK_CPURegs, MCK_CPURegs }, },
  { 163 /* movn */, Cpu0::MOVN_I_I, Convert__Reg1_0__Reg1_1__Reg1_2__Tie0_1_1, AMFBS_Ch8_2, { MCK_CPURegs, MCK_CPURegs, MCK_CPURegs }, },
  { 168 /* movz */, Cpu0::MOVZ_I_I, Convert__Reg1_0__Reg1_1__Reg1_2__Tie0_1_1, AMFBS_Ch8_2, { MCK_CPURegs, MCK_CPURegs, MCK_CPURegs }, },
  { 173 /* mtc0 */, Cpu0::MTC0, Convert__Reg1_0__Reg1_1, AMFBS_Ch4_1, { MCK_C0Regs, MCK_CPURegs }, },
  { 178 /* mthi */, Cpu0::MTHI, Convert__Reg1_0, AMFBS_Ch4_1, { MCK_CPURegs }, },
  { 183 /* mtlo */, Cpu0::MTLO, Convert__Reg1_0, AMFBS_Ch4_1, { MCK_CPURegs }, },
  { 188 /* mul */, Cpu0::MUL, Convert__Reg1_0__Reg1_1__Reg1_2, AMFBS_Ch4_1, { MCK_CPURegs, MCK_CPURegs, MCK_CPURegs }, },
  { 192 /* mult */, Cpu0::MULT, Convert__Reg1_0__Reg1_1, AMFBS_Ch4_1, { MCK_CPURegs, MCK_CPURegs }, },
  { 197 /* multu */, Cpu0::MULTu, Convert__Reg1_0__Reg1_1, AMFBS_Ch4_1, { MCK_CPURegs, MCK_CPURegs }, },
  { 203 /* nop */, Cpu0::NOP, Convert_NoOperands, AMFBS_None, {  }, },
  { 207 /* nor */, Cpu0::NOR, Convert__Reg1_0__Reg1_1__Reg1_2, AMFBS_Ch4_2, { MCK_CPURegs, MCK_CPURegs, MCK_CPURegs }, },
  { 211 /* or */, Cpu0::OR, Convert__Reg1_0__Reg1_1__Reg1_2, AMFBS_Ch4_2, { MCK_CPURegs, MCK_CPURegs, MCK_CPURegs }, },
  { 214 /* ori */, Cpu0::ORi, Convert__Reg1_0__Reg1_1__Imm1_2, AMFBS_Ch3_5, { MCK_CPURegs, MCK_CPURegs, MCK_Imm }, },
  { 218 /* rol */, Cpu0::ROL, Convert__Reg1_0__Reg1_1__Imm1_2, AMFBS_Ch4_1, { MCK_CPURegs, MCK_CPURegs, MCK_Imm }, },
  { 222 /* rolv */, Cpu0::ROLV, Convert__Reg1_0__Reg1_1__Reg1_2, AMFBS_Ch4_1, { MCK_CPURegs, MCK_CPURegs, MCK_CPURegs }, },
  { 227 /* ror */, Cpu0::ROR, Convert__Reg1_0__Reg1_1__Imm1_2, AMFBS_Ch4_1, { MCK_CPURegs, MCK_CPURegs, MCK_Imm }, },
  { 231 /* rorv */, Cpu0::RORV, Convert__Reg1_0__Reg1_1__Reg1_2, AMFBS_Ch4_1, { MCK_CPURegs, MCK_CPURegs, MCK_CPURegs }, },
  { 236 /* sb */, Cpu0::SB, Convert__Reg1_0__Mem2_1, AMFBS_Ch7_1, { MCK_CPURegs, MCK_Mem }, },
  { 239 /* sc */, Cpu0::SC, Convert__Reg1_0__Tie0_1_1__Mem2_1, AMFBS_None, { MCK_CPURegs, MCK_Mem }, },
  { 242 /* sh */, Cpu0::SH, Convert__Reg1_0__Mem2_1, AMFBS_Ch7_1, { MCK_CPURegs, MCK_Mem }, },
  { 245 /* shl */, Cpu0::SHL, Convert__Reg1_0__Reg1_1__Imm1_2, AMFBS_Ch3_5, { MCK_CPURegs, MCK_CPURegs, MCK_Imm }, },
  { 249 /* shlv */, Cpu0::SHLV, Convert__Reg1_0__Reg1_1__Reg1_2, AMFBS_Ch4_1, { MCK_CPURegs, MCK_CPURegs, MCK_CPURegs }, },
  { 254 /* shr */, Cpu0::SHR, Convert__Reg1_0__Reg1_1__Imm1_2, AMFBS_Ch4_1, { MCK_CPURegs, MCK_CPURegs, MCK_Imm }, },
  { 258 /* shrv */, Cpu0::SHRV, Convert__Reg1_0__Reg1_1__Reg1_2, AMFBS_Ch4_1, { MCK_CPURegs, MCK_CPURegs, MCK_CPURegs }, },
  { 263 /* slt */, Cpu0::SLT, Convert__Reg1_0__Reg1_1__Reg1_2, AMFBS_None, { MCK_CPURegs, MCK_CPURegs, MCK_CPURegs }, },
  { 267 /* slti */, Cpu0::SLTi, Convert__Reg1_0__Reg1_1__Imm1_2, AMFBS_None, { MCK_CPURegs, MCK_CPURegs, MCK_Imm }, },
  { 272 /* sltiu */, Cpu0::SLTiu, Convert__Reg1_0__Reg1_1__Imm1_2, AMFBS_None, { MCK_CPURegs, MCK_CPURegs, MCK_Imm }, },
  { 278 /* sltu */, Cpu0::SLTu, Convert__Reg1_0__Reg1_1__Reg1_2, AMFBS_None, { MCK_CPURegs, MCK_CPURegs, MCK_CPURegs }, },
  { 283 /* sra */, Cpu0::SRA, Convert__Reg1_0__Reg1_1__Imm1_2, AMFBS_Ch4_1, { MCK_CPURegs, MCK_CPURegs, MCK_Imm }, },
  { 287 /* srav */, Cpu0::SRAV, Convert__Reg1_0__Reg1_1__Reg1_2, AMFBS_Ch4_1, { MCK_CPURegs, MCK_CPURegs, MCK_CPURegs }, },
  { 292 /* st */, Cpu0::ST, Convert__Reg1_0__Mem2_1, AMFBS_None, { MCK_CPURegs, MCK_Mem }, },
  { 295 /* sub */, Cpu0::SUB, Convert__Reg1_0__Reg1_1__Reg1_2, AMFBS_None, { MCK_CPURegs, MCK_CPURegs, MCK_CPURegs }, },
  { 299 /* subu */, Cpu0::SUBu, Convert__Reg1_0__Reg1_1__Reg1_2, AMFBS_None, { MCK_CPURegs, MCK_CPURegs, MCK_CPURegs }, },
  { 304 /* sync */, Cpu0::SYNC, Convert__imm_95_0, AMFBS_None, {  }, },
  { 304 /* sync */, Cpu0::SYNC, Convert__Imm1_0, AMFBS_Ch12_1, { MCK_Imm }, },
  { 309 /* xor */, Cpu0::XOR, Convert__Reg1_0__Reg1_1__Reg1_2, AMFBS_Ch4_2, { MCK_CPURegs, MCK_CPURegs, MCK_CPURegs }, },
  { 313 /* xori */, Cpu0::XORi, Convert__Reg1_0__Reg1_1__Imm1_2, AMFBS_Ch4_2, { MCK_CPURegs, MCK_CPURegs, MCK_Imm }, },
};

#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

unsigned Cpu0AsmParser::
MatchInstructionImpl(const OperandVector &Operands,
                     MCInst &Inst,
                     uint64_t &ErrorInfo,
                     FeatureBitset &MissingFeatures,
                     bool matchingInlineAsm, unsigned VariantID) {
  // Eliminate obvious mismatches.
  if (Operands.size() > 4) {
    ErrorInfo = 4;
    return Match_InvalidOperand;
  }

  // Get the current feature set.
  const FeatureBitset &AvailableFeatures = getAvailableFeatures();

  // Get the instruction mnemonic, which is the first token.
  StringRef Mnemonic = ((Cpu0Operand &)*Operands[0]).getToken();

  // Some state to try to produce better error messages.
  bool HadMatchOtherThanFeatures = false;
  bool HadMatchOtherThanPredicate = false;
  unsigned RetCode = Match_InvalidOperand;
  MissingFeatures.set();
  // Set ErrorInfo to the operand that mismatches if it is
  // wrong for all instances of the instruction.
  ErrorInfo = ~0ULL;
  // Find the appropriate table for this asm variant.
  const MatchEntry *Start, *End;
  switch (VariantID) {
  default: llvm_unreachable("invalid variant!");
  case 0: Start = std::begin(MatchTable0); End = std::end(MatchTable0); break;
  }
  // Search the table.
  auto MnemonicRange = std::equal_range(Start, End, Mnemonic, LessOpcode());

  DEBUG_WITH_TYPE("asm-matcher", dbgs() << "AsmMatcher: found " <<
  std::distance(MnemonicRange.first, MnemonicRange.second) <<
  " encodings with mnemonic '" << Mnemonic << "'\n");

  // Return a more specific error code if no mnemonics match.
  if (MnemonicRange.first == MnemonicRange.second)
    return Match_MnemonicFail;

  for (const MatchEntry *it = MnemonicRange.first, *ie = MnemonicRange.second;
       it != ie; ++it) {
    const FeatureBitset &RequiredFeatures = FeatureBitsets[it->RequiredFeaturesIdx];
    bool HasRequiredFeatures =
      (AvailableFeatures & RequiredFeatures) == RequiredFeatures;
    DEBUG_WITH_TYPE("asm-matcher", dbgs() << "Trying to match opcode "
                                          << MII.getName(it->Opcode) << "\n");
    // equal_range guarantees that instruction mnemonic matches.
    assert(Mnemonic == it->getMnemonic());
    bool OperandsValid = true;
    for (unsigned FormalIdx = 0, ActualIdx = 1; FormalIdx != 3; ++FormalIdx) {
      auto Formal = static_cast<MatchClassKind>(it->Classes[FormalIdx]);
      DEBUG_WITH_TYPE("asm-matcher",
                      dbgs() << "  Matching formal operand class " << getMatchClassName(Formal)
                             << " against actual operand at index " << ActualIdx);
      if (ActualIdx < Operands.size())
        DEBUG_WITH_TYPE("asm-matcher", dbgs() << " (";
                        Operands[ActualIdx]->print(dbgs()); dbgs() << "): ");
      else
        DEBUG_WITH_TYPE("asm-matcher", dbgs() << ": ");
      if (ActualIdx >= Operands.size()) {
        DEBUG_WITH_TYPE("asm-matcher", dbgs() << "actual operand index out of range ");
        OperandsValid = (Formal == InvalidMatchClass) || isSubclass(Formal, OptionalMatchClass);
        if (!OperandsValid) ErrorInfo = ActualIdx;
        break;
      }
      MCParsedAsmOperand &Actual = *Operands[ActualIdx];
      unsigned Diag = validateOperandClass(Actual, Formal);
      if (Diag == Match_Success) {
        DEBUG_WITH_TYPE("asm-matcher",
                        dbgs() << "match success using generic matcher\n");
        ++ActualIdx;
        continue;
      }
      // If the generic handler indicates an invalid operand
      // failure, check for a special case.
      if (Diag != Match_Success) {
        unsigned TargetDiag = validateTargetOperandClass(Actual, Formal);
        if (TargetDiag == Match_Success) {
          DEBUG_WITH_TYPE("asm-matcher",
                          dbgs() << "match success using target matcher\n");
          ++ActualIdx;
          continue;
        }
        // If the target matcher returned a specific error code use
        // that, else use the one from the generic matcher.
        if (TargetDiag != Match_InvalidOperand && HasRequiredFeatures)
          Diag = TargetDiag;
      }
      // If current formal operand wasn't matched and it is optional
      // then try to match next formal operand
      if (Diag == Match_InvalidOperand && isSubclass(Formal, OptionalMatchClass)) {
        DEBUG_WITH_TYPE("asm-matcher", dbgs() << "ignoring optional operand\n");
        continue;
      }
      // If this operand is broken for all of the instances of this
      // mnemonic, keep track of it so we can report loc info.
      // If we already had a match that only failed due to a
      // target predicate, that diagnostic is preferred.
      if (!HadMatchOtherThanPredicate &&
          (it == MnemonicRange.first || ErrorInfo <= ActualIdx)) {
        if (HasRequiredFeatures && (ErrorInfo != ActualIdx || Diag != Match_InvalidOperand))
          RetCode = Diag;
        ErrorInfo = ActualIdx;
      }
      // Otherwise, just reject this instance of the mnemonic.
      OperandsValid = false;
      break;
    }

    if (!OperandsValid) {
      DEBUG_WITH_TYPE("asm-matcher", dbgs() << "Opcode result: multiple "
                                               "operand mismatches, ignoring "
                                               "this opcode\n");
      continue;
    }
    if (!HasRequiredFeatures) {
      HadMatchOtherThanFeatures = true;
      FeatureBitset NewMissingFeatures = RequiredFeatures & ~AvailableFeatures;
      DEBUG_WITH_TYPE("asm-matcher", dbgs() << "Missing target features:";
                      for (unsigned I = 0, E = NewMissingFeatures.size(); I != E; ++I)
                        if (NewMissingFeatures[I])
                          dbgs() << ' ' << I;
                      dbgs() << "\n");
      if (NewMissingFeatures.count() <=
          MissingFeatures.count())
        MissingFeatures = NewMissingFeatures;
      continue;
    }

    Inst.clear();

    Inst.setOpcode(it->Opcode);
    // We have a potential match but have not rendered the operands.
    // Check the target predicate to handle any context sensitive
    // constraints.
    // For example, Ties that are referenced multiple times must be
    // checked here to ensure the input is the same for each match
    // constraints. If we leave it any later the ties will have been
    // canonicalized
    unsigned MatchResult;
    if ((MatchResult = checkEarlyTargetMatchPredicate(Inst, Operands)) != Match_Success) {
      Inst.clear();
      DEBUG_WITH_TYPE(
          "asm-matcher",
          dbgs() << "Early target match predicate failed with diag code "
                 << MatchResult << "\n");
      RetCode = MatchResult;
      HadMatchOtherThanPredicate = true;
      continue;
    }

    if (matchingInlineAsm) {
      convertToMapAndConstraints(it->ConvertFn, Operands);
      if (!checkAsmTiedOperandConstraints(*this, it->ConvertFn, Operands, ErrorInfo))
        return Match_InvalidTiedOperand;

      return Match_Success;
    }

    // We have selected a definite instruction, convert the parsed
    // operands into the appropriate MCInst.
    convertToMCInst(it->ConvertFn, Inst, it->Opcode, Operands);

    // We have a potential match. Check the target predicate to
    // handle any context sensitive constraints.
    if ((MatchResult = checkTargetMatchPredicate(Inst)) != Match_Success) {
      DEBUG_WITH_TYPE("asm-matcher",
                      dbgs() << "Target match predicate failed with diag code "
                             << MatchResult << "\n");
      Inst.clear();
      RetCode = MatchResult;
      HadMatchOtherThanPredicate = true;
      continue;
    }

    if (!checkAsmTiedOperandConstraints(*this, it->ConvertFn, Operands, ErrorInfo))
      return Match_InvalidTiedOperand;

    DEBUG_WITH_TYPE(
        "asm-matcher",
        dbgs() << "Opcode result: complete match, selecting this opcode\n");
    return Match_Success;
  }

  // Okay, we had no match.  Try to return a useful error code.
  if (HadMatchOtherThanPredicate || !HadMatchOtherThanFeatures)
    return RetCode;

  ErrorInfo = 0;
  return Match_MissingFeature;
}

namespace {
  struct OperandMatchEntry {
    uint16_t Mnemonic;
    uint8_t OperandMask;
    uint8_t Class;
    uint8_t RequiredFeaturesIdx;

    StringRef getMnemonic() const {
      return StringRef(MnemonicTable + Mnemonic + 1,
                       MnemonicTable[Mnemonic]);
    }
  };

  // Predicate for searching for an opcode.
  struct LessOpcodeOperand {
    bool operator()(const OperandMatchEntry &LHS, StringRef RHS) {
      return LHS.getMnemonic()  < RHS;
    }
    bool operator()(StringRef LHS, const OperandMatchEntry &RHS) {
      return LHS < RHS.getMnemonic();
    }
    bool operator()(const OperandMatchEntry &LHS, const OperandMatchEntry &RHS) {
      return LHS.getMnemonic() < RHS.getMnemonic();
    }
  };
} // end anonymous namespace

static const OperandMatchEntry OperandMatchTable[11] = {
  /* Operand List Mnemonic, Mask, Operand Class, Features */
  { 113 /* la */, 2 /* 1 */, MCK_Mem, AMFBS_Ch11_1 },
  { 116 /* lb */, 2 /* 1 */, MCK_Mem, AMFBS_Ch7_1 },
  { 119 /* lbu */, 2 /* 1 */, MCK_Mem, AMFBS_Ch7_1 },
  { 123 /* ld */, 2 /* 1 */, MCK_Mem, AMFBS_None },
  { 126 /* lh */, 2 /* 1 */, MCK_Mem, AMFBS_Ch7_1 },
  { 129 /* lhu */, 2 /* 1 */, MCK_Mem, AMFBS_Ch7_1 },
  { 136 /* ll */, 2 /* 1 */, MCK_Mem, AMFBS_None },
  { 236 /* sb */, 2 /* 1 */, MCK_Mem, AMFBS_Ch7_1 },
  { 239 /* sc */, 2 /* 1 */, MCK_Mem, AMFBS_None },
  { 242 /* sh */, 2 /* 1 */, MCK_Mem, AMFBS_Ch7_1 },
  { 292 /* st */, 2 /* 1 */, MCK_Mem, AMFBS_None },
};

OperandMatchResultTy Cpu0AsmParser::
tryCustomParseOperand(OperandVector &Operands,
                      unsigned MCK) {

  switch(MCK) {
  case MCK_Mem:
    return parseMemOperand(Operands);
  default:
    return MatchOperand_NoMatch;
  }
  return MatchOperand_NoMatch;
}

OperandMatchResultTy Cpu0AsmParser::
MatchOperandParserImpl(OperandVector &Operands,
                       StringRef Mnemonic,
                       bool ParseForAllFeatures) {
  // Get the current feature set.
  const FeatureBitset &AvailableFeatures = getAvailableFeatures();

  // Get the next operand index.
  unsigned NextOpNum = Operands.size() - 1;
  // Search the table.
  auto MnemonicRange =
    std::equal_range(std::begin(OperandMatchTable), std::end(OperandMatchTable),
                     Mnemonic, LessOpcodeOperand());

  if (MnemonicRange.first == MnemonicRange.second)
    return MatchOperand_NoMatch;

  for (const OperandMatchEntry *it = MnemonicRange.first,
       *ie = MnemonicRange.second; it != ie; ++it) {
    // equal_range guarantees that instruction mnemonic matches.
    assert(Mnemonic == it->getMnemonic());

    // check if the available features match
    const FeatureBitset &RequiredFeatures = FeatureBitsets[it->RequiredFeaturesIdx];
    if (!ParseForAllFeatures && (AvailableFeatures & RequiredFeatures) != RequiredFeatures)
      continue;

    // check if the operand in question has a custom parser.
    if (!(it->OperandMask & (1 << NextOpNum)))
      continue;

    // call custom parse method to handle the operand
    OperandMatchResultTy Result = tryCustomParseOperand(Operands, it->Class);
    if (Result != MatchOperand_NoMatch)
      return Result;
  }

  // Okay, we had no match.
  return MatchOperand_NoMatch;
}

#endif // GET_MATCHER_IMPLEMENTATION


#ifdef GET_MNEMONIC_SPELL_CHECKER
#undef GET_MNEMONIC_SPELL_CHECKER

static std::string Cpu0MnemonicSpellCheck(StringRef S, const FeatureBitset &FBS, unsigned VariantID) {
  const unsigned MaxEditDist = 2;
  std::vector<StringRef> Candidates;
  StringRef Prev = "";

  // Find the appropriate table for this asm variant.
  const MatchEntry *Start, *End;
  switch (VariantID) {
  default: llvm_unreachable("invalid variant!");
  case 0: Start = std::begin(MatchTable0); End = std::end(MatchTable0); break;
  }

  for (auto I = Start; I < End; I++) {
    // Ignore unsupported instructions.
    const FeatureBitset &RequiredFeatures = FeatureBitsets[I->RequiredFeaturesIdx];
    if ((FBS & RequiredFeatures) != RequiredFeatures)
      continue;

    StringRef T = I->getMnemonic();
    // Avoid recomputing the edit distance for the same string.
    if (T.equals(Prev))
      continue;

    Prev = T;
    unsigned Dist = S.edit_distance(T, false, MaxEditDist);
    if (Dist <= MaxEditDist)
      Candidates.push_back(T);
  }

  if (Candidates.empty())
    return "";

  std::string Res = ", did you mean: ";
  unsigned i = 0;
  for (; i < Candidates.size() - 1; i++)
    Res += Candidates[i].str() + ", ";
  return Res + Candidates[i].str() + "?";
}

#endif // GET_MNEMONIC_SPELL_CHECKER


#ifdef GET_MNEMONIC_CHECKER
#undef GET_MNEMONIC_CHECKER

static bool Cpu0CheckMnemonic(StringRef Mnemonic,
                                const FeatureBitset &AvailableFeatures,
                                unsigned VariantID) {
  // Find the appropriate table for this asm variant.
  const MatchEntry *Start, *End;
  switch (VariantID) {
  default: llvm_unreachable("invalid variant!");
  case 0: Start = std::begin(MatchTable0); End = std::end(MatchTable0); break;
  }

  // Search the table.
  auto MnemonicRange = std::equal_range(Start, End, Mnemonic, LessOpcode());

  if (MnemonicRange.first == MnemonicRange.second)
    return false;

  for (const MatchEntry *it = MnemonicRange.first, *ie = MnemonicRange.second;
       it != ie; ++it) {
    const FeatureBitset &RequiredFeatures =
      FeatureBitsets[it->RequiredFeaturesIdx];
    if ((AvailableFeatures & RequiredFeatures) == RequiredFeatures)
      return true;
  }
  return false;
}

#endif // GET_MNEMONIC_CHECKER

