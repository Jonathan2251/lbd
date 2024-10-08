/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Assembly Writer Source Fragment                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// getMnemonic - This method is automatically generated by tablegen
/// from the instruction set description.
std::pair<const char *, uint64_t> Cpu0InstPrinter::getMnemonic(const MCInst *MI) {

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif
  static const char AsmStrs[] = {
  /* 0 */ "mfc0\t\0"
  /* 6 */ "mtc0\t\0"
  /* 12 */ "la\t\0"
  /* 16 */ "sra\t\0"
  /* 21 */ "lb\t\0"
  /* 25 */ "sb\t\0"
  /* 29 */ "jsub\t\0"
  /* 35 */ "sc\t\0"
  /* 39 */ "gcd\t\0"
  /* 44 */ "add\t\0"
  /* 49 */ "ld\t\0"
  /* 53 */ "and\t\0"
  /* 58 */ "jge\t\0"
  /* 63 */ "jle\t\0"
  /* 68 */ "bne\t\0"
  /* 73 */ "jne\t\0"
  /* 78 */ ".cprestore\t\0"
  /* 90 */ "lh\t\0"
  /* 94 */ "sh\t\0"
  /* 98 */ "andi\t\0"
  /* 104 */ "mfhi\t\0"
  /* 110 */ "mthi\t\0"
  /* 116 */ "li\t\0"
  /* 120 */ "xori\t\0"
  /* 126 */ "slti\t\0"
  /* 132 */ "lui\t\0"
  /* 137 */ "bal\t\0"
  /* 142 */ "shl\t\0"
  /* 147 */ "ll\t\0"
  /* 151 */ "rol\t\0"
  /* 156 */ "mul\t\0"
  /* 161 */ "movn\t\0"
  /* 167 */ "clo\t\0"
  /* 172 */ "mflo\t\0"
  /* 178 */ "mtlo\t\0"
  /* 184 */ "cmp\t\0"
  /* 189 */ "jmp\t\0"
  /* 194 */ "beq\t\0"
  /* 199 */ "jeq\t\0"
  /* 204 */ "shr\t\0"
  /* 209 */ "jr\t\0"
  /* 213 */ "jalr\t\0"
  /* 219 */ "nor\t\0"
  /* 224 */ "ror\t\0"
  /* 229 */ "xor\t\0"
  /* 234 */ "ret\t\0"
  /* 239 */ "jgt\t\0"
  /* 244 */ "jlt\t\0"
  /* 249 */ "slt\t\0"
  /* 254 */ "mult\t\0"
  /* 260 */ "st\t\0"
  /* 264 */ "lbu\t\0"
  /* 269 */ "subu\t\0"
  /* 275 */ "addu\t\0"
  /* 281 */ "lhu\t\0"
  /* 286 */ "addiu\t\0"
  /* 293 */ "sltiu\t\0"
  /* 300 */ "cmpu\t\0"
  /* 306 */ "sltu\t\0"
  /* 312 */ "multu\t\0"
  /* 319 */ "divu\t\0"
  /* 325 */ "srav\t\0"
  /* 331 */ "div\t\0"
  /* 336 */ "shlv\t\0"
  /* 342 */ "rolv\t\0"
  /* 348 */ "c0mov\t\0"
  /* 355 */ "shrv\t\0"
  /* 361 */ "rorv\t\0"
  /* 367 */ "clz\t\0"
  /* 372 */ "movz\t\0"
  /* 378 */ "!ADJCALLSTACKDOWN \0"
  /* 397 */ "!ADJCALLSTACKUP \0"
  /* 414 */ "sync \0"
  /* 420 */ "# XRay Function Patchable RET.\0"
  /* 451 */ "# XRay Typed Event Log.\0"
  /* 475 */ "# XRay Custom Event Log.\0"
  /* 500 */ "# XRay Function Enter.\0"
  /* 523 */ "# XRay Tail Call Exit.\0"
  /* 546 */ "# XRay Function Exit.\0"
  /* 568 */ "LIFETIME_END\0"
  /* 581 */ "PSEUDO_PROBE\0"
  /* 594 */ "BUNDLE\0"
  /* 601 */ "DBG_VALUE\0"
  /* 611 */ "DBG_INSTR_REF\0"
  /* 625 */ "DBG_LABEL\0"
  /* 635 */ "LIFETIME_START\0"
  /* 650 */ "# FEntry call\0"
  /* 664 */ "nop\0"
};
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

  static const uint16_t OpInfo0[] = {
    0U,	// PHI
    0U,	// INLINEASM
    0U,	// INLINEASM_BR
    0U,	// CFI_INSTRUCTION
    0U,	// EH_LABEL
    0U,	// GC_LABEL
    0U,	// ANNOTATION_LABEL
    0U,	// KILL
    0U,	// EXTRACT_SUBREG
    0U,	// INSERT_SUBREG
    0U,	// IMPLICIT_DEF
    0U,	// SUBREG_TO_REG
    0U,	// COPY_TO_REGCLASS
    602U,	// DBG_VALUE
    612U,	// DBG_INSTR_REF
    626U,	// DBG_LABEL
    0U,	// REG_SEQUENCE
    0U,	// COPY
    595U,	// BUNDLE
    636U,	// LIFETIME_START
    569U,	// LIFETIME_END
    582U,	// PSEUDO_PROBE
    0U,	// STACKMAP
    651U,	// FENTRY_CALL
    0U,	// PATCHPOINT
    0U,	// LOAD_STACK_GUARD
    0U,	// PREALLOCATED_SETUP
    0U,	// PREALLOCATED_ARG
    0U,	// STATEPOINT
    0U,	// LOCAL_ESCAPE
    0U,	// FAULTING_OP
    0U,	// PATCHABLE_OP
    501U,	// PATCHABLE_FUNCTION_ENTER
    421U,	// PATCHABLE_RET
    547U,	// PATCHABLE_FUNCTION_EXIT
    524U,	// PATCHABLE_TAIL_CALL
    476U,	// PATCHABLE_EVENT_CALL
    452U,	// PATCHABLE_TYPED_EVENT_CALL
    0U,	// ICALL_BRANCH_FUNNEL
    0U,	// G_ADD
    0U,	// G_SUB
    0U,	// G_MUL
    0U,	// G_SDIV
    0U,	// G_UDIV
    0U,	// G_SREM
    0U,	// G_UREM
    0U,	// G_AND
    0U,	// G_OR
    0U,	// G_XOR
    0U,	// G_IMPLICIT_DEF
    0U,	// G_PHI
    0U,	// G_FRAME_INDEX
    0U,	// G_GLOBAL_VALUE
    0U,	// G_EXTRACT
    0U,	// G_UNMERGE_VALUES
    0U,	// G_INSERT
    0U,	// G_MERGE_VALUES
    0U,	// G_BUILD_VECTOR
    0U,	// G_BUILD_VECTOR_TRUNC
    0U,	// G_CONCAT_VECTORS
    0U,	// G_PTRTOINT
    0U,	// G_INTTOPTR
    0U,	// G_BITCAST
    0U,	// G_FREEZE
    0U,	// G_INTRINSIC_TRUNC
    0U,	// G_INTRINSIC_ROUND
    0U,	// G_INTRINSIC_LRINT
    0U,	// G_INTRINSIC_ROUNDEVEN
    0U,	// G_READCYCLECOUNTER
    0U,	// G_LOAD
    0U,	// G_SEXTLOAD
    0U,	// G_ZEXTLOAD
    0U,	// G_INDEXED_LOAD
    0U,	// G_INDEXED_SEXTLOAD
    0U,	// G_INDEXED_ZEXTLOAD
    0U,	// G_STORE
    0U,	// G_INDEXED_STORE
    0U,	// G_ATOMIC_CMPXCHG_WITH_SUCCESS
    0U,	// G_ATOMIC_CMPXCHG
    0U,	// G_ATOMICRMW_XCHG
    0U,	// G_ATOMICRMW_ADD
    0U,	// G_ATOMICRMW_SUB
    0U,	// G_ATOMICRMW_AND
    0U,	// G_ATOMICRMW_NAND
    0U,	// G_ATOMICRMW_OR
    0U,	// G_ATOMICRMW_XOR
    0U,	// G_ATOMICRMW_MAX
    0U,	// G_ATOMICRMW_MIN
    0U,	// G_ATOMICRMW_UMAX
    0U,	// G_ATOMICRMW_UMIN
    0U,	// G_ATOMICRMW_FADD
    0U,	// G_ATOMICRMW_FSUB
    0U,	// G_FENCE
    0U,	// G_BRCOND
    0U,	// G_BRINDIRECT
    0U,	// G_INTRINSIC
    0U,	// G_INTRINSIC_W_SIDE_EFFECTS
    0U,	// G_ANYEXT
    0U,	// G_TRUNC
    0U,	// G_CONSTANT
    0U,	// G_FCONSTANT
    0U,	// G_VASTART
    0U,	// G_VAARG
    0U,	// G_SEXT
    0U,	// G_SEXT_INREG
    0U,	// G_ZEXT
    0U,	// G_SHL
    0U,	// G_LSHR
    0U,	// G_ASHR
    0U,	// G_FSHL
    0U,	// G_FSHR
    0U,	// G_ICMP
    0U,	// G_FCMP
    0U,	// G_SELECT
    0U,	// G_UADDO
    0U,	// G_UADDE
    0U,	// G_USUBO
    0U,	// G_USUBE
    0U,	// G_SADDO
    0U,	// G_SADDE
    0U,	// G_SSUBO
    0U,	// G_SSUBE
    0U,	// G_UMULO
    0U,	// G_SMULO
    0U,	// G_UMULH
    0U,	// G_SMULH
    0U,	// G_UADDSAT
    0U,	// G_SADDSAT
    0U,	// G_USUBSAT
    0U,	// G_SSUBSAT
    0U,	// G_USHLSAT
    0U,	// G_SSHLSAT
    0U,	// G_SMULFIX
    0U,	// G_UMULFIX
    0U,	// G_SMULFIXSAT
    0U,	// G_UMULFIXSAT
    0U,	// G_SDIVFIX
    0U,	// G_UDIVFIX
    0U,	// G_SDIVFIXSAT
    0U,	// G_UDIVFIXSAT
    0U,	// G_FADD
    0U,	// G_FSUB
    0U,	// G_FMUL
    0U,	// G_FMA
    0U,	// G_FMAD
    0U,	// G_FDIV
    0U,	// G_FREM
    0U,	// G_FPOW
    0U,	// G_FPOWI
    0U,	// G_FEXP
    0U,	// G_FEXP2
    0U,	// G_FLOG
    0U,	// G_FLOG2
    0U,	// G_FLOG10
    0U,	// G_FNEG
    0U,	// G_FPEXT
    0U,	// G_FPTRUNC
    0U,	// G_FPTOSI
    0U,	// G_FPTOUI
    0U,	// G_SITOFP
    0U,	// G_UITOFP
    0U,	// G_FABS
    0U,	// G_FCOPYSIGN
    0U,	// G_FCANONICALIZE
    0U,	// G_FMINNUM
    0U,	// G_FMAXNUM
    0U,	// G_FMINNUM_IEEE
    0U,	// G_FMAXNUM_IEEE
    0U,	// G_FMINIMUM
    0U,	// G_FMAXIMUM
    0U,	// G_PTR_ADD
    0U,	// G_PTRMASK
    0U,	// G_SMIN
    0U,	// G_SMAX
    0U,	// G_UMIN
    0U,	// G_UMAX
    0U,	// G_ABS
    0U,	// G_BR
    0U,	// G_BRJT
    0U,	// G_INSERT_VECTOR_ELT
    0U,	// G_EXTRACT_VECTOR_ELT
    0U,	// G_SHUFFLE_VECTOR
    0U,	// G_CTTZ
    0U,	// G_CTTZ_ZERO_UNDEF
    0U,	// G_CTLZ
    0U,	// G_CTLZ_ZERO_UNDEF
    0U,	// G_CTPOP
    0U,	// G_BSWAP
    0U,	// G_BITREVERSE
    0U,	// G_FCEIL
    0U,	// G_FCOS
    0U,	// G_FSIN
    0U,	// G_FSQRT
    0U,	// G_FFLOOR
    0U,	// G_FRINT
    0U,	// G_FNEARBYINT
    0U,	// G_ADDRSPACE_CAST
    0U,	// G_BLOCK_ADDR
    0U,	// G_JUMP_TABLE
    0U,	// G_DYN_STACKALLOC
    0U,	// G_STRICT_FADD
    0U,	// G_STRICT_FSUB
    0U,	// G_STRICT_FMUL
    0U,	// G_STRICT_FDIV
    0U,	// G_STRICT_FREM
    0U,	// G_STRICT_FMA
    0U,	// G_STRICT_FSQRT
    0U,	// G_READ_REGISTER
    0U,	// G_WRITE_REGISTER
    0U,	// G_MEMCPY
    0U,	// G_MEMMOVE
    0U,	// G_MEMSET
    0U,	// G_VECREDUCE_SEQ_FADD
    0U,	// G_VECREDUCE_SEQ_FMUL
    0U,	// G_VECREDUCE_FADD
    0U,	// G_VECREDUCE_FMUL
    0U,	// G_VECREDUCE_FMAX
    0U,	// G_VECREDUCE_FMIN
    0U,	// G_VECREDUCE_ADD
    0U,	// G_VECREDUCE_MUL
    0U,	// G_VECREDUCE_AND
    0U,	// G_VECREDUCE_OR
    0U,	// G_VECREDUCE_XOR
    0U,	// G_VECREDUCE_SMAX
    0U,	// G_VECREDUCE_SMIN
    0U,	// G_VECREDUCE_UMAX
    0U,	// G_VECREDUCE_UMIN
    1403U,	// ADJCALLSTACKDOWN
    1422U,	// ADJCALLSTACKUP
    0U,	// ATOMIC_CMP_SWAP_I16
    0U,	// ATOMIC_CMP_SWAP_I32
    0U,	// ATOMIC_CMP_SWAP_I8
    0U,	// ATOMIC_LOAD_ADD_I16
    0U,	// ATOMIC_LOAD_ADD_I32
    0U,	// ATOMIC_LOAD_ADD_I8
    0U,	// ATOMIC_LOAD_AND_I16
    0U,	// ATOMIC_LOAD_AND_I32
    0U,	// ATOMIC_LOAD_AND_I8
    0U,	// ATOMIC_LOAD_NAND_I16
    0U,	// ATOMIC_LOAD_NAND_I32
    0U,	// ATOMIC_LOAD_NAND_I8
    0U,	// ATOMIC_LOAD_OR_I16
    0U,	// ATOMIC_LOAD_OR_I32
    0U,	// ATOMIC_LOAD_OR_I8
    0U,	// ATOMIC_LOAD_SUB_I16
    0U,	// ATOMIC_LOAD_SUB_I32
    0U,	// ATOMIC_LOAD_SUB_I8
    0U,	// ATOMIC_LOAD_XOR_I16
    0U,	// ATOMIC_LOAD_XOR_I32
    0U,	// ATOMIC_LOAD_XOR_I8
    0U,	// ATOMIC_SWAP_I16
    0U,	// ATOMIC_SWAP_I32
    0U,	// ATOMIC_SWAP_I8
    2127U,	// CPRESTORE
    0U,	// CPU0eh_return32
    0U,	// LONG_BRANCH_ADDiu
    0U,	// LONG_BRANCH_LUi
    10253U,	// LoadAddr32Imm
    26637U,	// LoadAddr32Reg
    10357U,	// LoadImm32Reg
    0U,	// RetLR
    0U,	// TAILCALL
    0U,	// TAILCALL_R
    10285U,	// ADD
    10527U,	// ADDiu
    10516U,	// ADDu
    10294U,	// AND
    10339U,	// ANDi
    3210U,	// BAL
    10435U,	// BEQ
    10309U,	// BNE
    10589U,	// C0MOVE
    10408U,	// CLO
    10608U,	// CLZ
    10425U,	// CMP
    10541U,	// CMPu
    10280U,	// GCD
    2262U,	// JALR
    43208U,	// JEQ
    43067U,	// JGE
    43248U,	// JGT
    43072U,	// JLE
    43253U,	// JLT
    3262U,	// JMP
    43082U,	// JNE
    2258U,	// JR
    3102U,	// JSUB
    26646U,	// LB
    26889U,	// LBu
    26674U,	// LD
    59679U,	// LEA_ADDiu
    26715U,	// LH
    26906U,	// LHu
    26772U,	// LL
    10373U,	// LUi
    10241U,	// MFC0
    2153U,	// MFHI
    2221U,	// MFLO
    10402U,	// MOVN_I_I
    10613U,	// MOVZ_I_I
    10247U,	// MTC0
    2159U,	// MTHI
    2227U,	// MTLO
    10397U,	// MUL
    10495U,	// MULT
    10553U,	// MULTu
    665U,	// NOP
    10460U,	// NOR
    10461U,	// OR
    10362U,	// ORi
    2283U,	// RET
    10392U,	// ROL
    10583U,	// ROLV
    10465U,	// ROR
    10602U,	// RORV
    26650U,	// SB
    4132U,	// SC
    10572U,	// SDIV
    26719U,	// SH
    10383U,	// SHL
    10577U,	// SHLV
    10445U,	// SHR
    10596U,	// SHRV
    10490U,	// SLT
    10367U,	// SLTi
    10534U,	// SLTiu
    10547U,	// SLTu
    10257U,	// SRA
    10566U,	// SRAV
    26885U,	// ST
    10271U,	// SUB
    10510U,	// SUBu
    2463U,	// SYNC
    10560U,	// UDIV
    10470U,	// XOR
    10361U,	// XORi
  };

  static const uint8_t OpInfo1[] = {
    0U,	// PHI
    0U,	// INLINEASM
    0U,	// INLINEASM_BR
    0U,	// CFI_INSTRUCTION
    0U,	// EH_LABEL
    0U,	// GC_LABEL
    0U,	// ANNOTATION_LABEL
    0U,	// KILL
    0U,	// EXTRACT_SUBREG
    0U,	// INSERT_SUBREG
    0U,	// IMPLICIT_DEF
    0U,	// SUBREG_TO_REG
    0U,	// COPY_TO_REGCLASS
    0U,	// DBG_VALUE
    0U,	// DBG_INSTR_REF
    0U,	// DBG_LABEL
    0U,	// REG_SEQUENCE
    0U,	// COPY
    0U,	// BUNDLE
    0U,	// LIFETIME_START
    0U,	// LIFETIME_END
    0U,	// PSEUDO_PROBE
    0U,	// STACKMAP
    0U,	// FENTRY_CALL
    0U,	// PATCHPOINT
    0U,	// LOAD_STACK_GUARD
    0U,	// PREALLOCATED_SETUP
    0U,	// PREALLOCATED_ARG
    0U,	// STATEPOINT
    0U,	// LOCAL_ESCAPE
    0U,	// FAULTING_OP
    0U,	// PATCHABLE_OP
    0U,	// PATCHABLE_FUNCTION_ENTER
    0U,	// PATCHABLE_RET
    0U,	// PATCHABLE_FUNCTION_EXIT
    0U,	// PATCHABLE_TAIL_CALL
    0U,	// PATCHABLE_EVENT_CALL
    0U,	// PATCHABLE_TYPED_EVENT_CALL
    0U,	// ICALL_BRANCH_FUNNEL
    0U,	// G_ADD
    0U,	// G_SUB
    0U,	// G_MUL
    0U,	// G_SDIV
    0U,	// G_UDIV
    0U,	// G_SREM
    0U,	// G_UREM
    0U,	// G_AND
    0U,	// G_OR
    0U,	// G_XOR
    0U,	// G_IMPLICIT_DEF
    0U,	// G_PHI
    0U,	// G_FRAME_INDEX
    0U,	// G_GLOBAL_VALUE
    0U,	// G_EXTRACT
    0U,	// G_UNMERGE_VALUES
    0U,	// G_INSERT
    0U,	// G_MERGE_VALUES
    0U,	// G_BUILD_VECTOR
    0U,	// G_BUILD_VECTOR_TRUNC
    0U,	// G_CONCAT_VECTORS
    0U,	// G_PTRTOINT
    0U,	// G_INTTOPTR
    0U,	// G_BITCAST
    0U,	// G_FREEZE
    0U,	// G_INTRINSIC_TRUNC
    0U,	// G_INTRINSIC_ROUND
    0U,	// G_INTRINSIC_LRINT
    0U,	// G_INTRINSIC_ROUNDEVEN
    0U,	// G_READCYCLECOUNTER
    0U,	// G_LOAD
    0U,	// G_SEXTLOAD
    0U,	// G_ZEXTLOAD
    0U,	// G_INDEXED_LOAD
    0U,	// G_INDEXED_SEXTLOAD
    0U,	// G_INDEXED_ZEXTLOAD
    0U,	// G_STORE
    0U,	// G_INDEXED_STORE
    0U,	// G_ATOMIC_CMPXCHG_WITH_SUCCESS
    0U,	// G_ATOMIC_CMPXCHG
    0U,	// G_ATOMICRMW_XCHG
    0U,	// G_ATOMICRMW_ADD
    0U,	// G_ATOMICRMW_SUB
    0U,	// G_ATOMICRMW_AND
    0U,	// G_ATOMICRMW_NAND
    0U,	// G_ATOMICRMW_OR
    0U,	// G_ATOMICRMW_XOR
    0U,	// G_ATOMICRMW_MAX
    0U,	// G_ATOMICRMW_MIN
    0U,	// G_ATOMICRMW_UMAX
    0U,	// G_ATOMICRMW_UMIN
    0U,	// G_ATOMICRMW_FADD
    0U,	// G_ATOMICRMW_FSUB
    0U,	// G_FENCE
    0U,	// G_BRCOND
    0U,	// G_BRINDIRECT
    0U,	// G_INTRINSIC
    0U,	// G_INTRINSIC_W_SIDE_EFFECTS
    0U,	// G_ANYEXT
    0U,	// G_TRUNC
    0U,	// G_CONSTANT
    0U,	// G_FCONSTANT
    0U,	// G_VASTART
    0U,	// G_VAARG
    0U,	// G_SEXT
    0U,	// G_SEXT_INREG
    0U,	// G_ZEXT
    0U,	// G_SHL
    0U,	// G_LSHR
    0U,	// G_ASHR
    0U,	// G_FSHL
    0U,	// G_FSHR
    0U,	// G_ICMP
    0U,	// G_FCMP
    0U,	// G_SELECT
    0U,	// G_UADDO
    0U,	// G_UADDE
    0U,	// G_USUBO
    0U,	// G_USUBE
    0U,	// G_SADDO
    0U,	// G_SADDE
    0U,	// G_SSUBO
    0U,	// G_SSUBE
    0U,	// G_UMULO
    0U,	// G_SMULO
    0U,	// G_UMULH
    0U,	// G_SMULH
    0U,	// G_UADDSAT
    0U,	// G_SADDSAT
    0U,	// G_USUBSAT
    0U,	// G_SSUBSAT
    0U,	// G_USHLSAT
    0U,	// G_SSHLSAT
    0U,	// G_SMULFIX
    0U,	// G_UMULFIX
    0U,	// G_SMULFIXSAT
    0U,	// G_UMULFIXSAT
    0U,	// G_SDIVFIX
    0U,	// G_UDIVFIX
    0U,	// G_SDIVFIXSAT
    0U,	// G_UDIVFIXSAT
    0U,	// G_FADD
    0U,	// G_FSUB
    0U,	// G_FMUL
    0U,	// G_FMA
    0U,	// G_FMAD
    0U,	// G_FDIV
    0U,	// G_FREM
    0U,	// G_FPOW
    0U,	// G_FPOWI
    0U,	// G_FEXP
    0U,	// G_FEXP2
    0U,	// G_FLOG
    0U,	// G_FLOG2
    0U,	// G_FLOG10
    0U,	// G_FNEG
    0U,	// G_FPEXT
    0U,	// G_FPTRUNC
    0U,	// G_FPTOSI
    0U,	// G_FPTOUI
    0U,	// G_SITOFP
    0U,	// G_UITOFP
    0U,	// G_FABS
    0U,	// G_FCOPYSIGN
    0U,	// G_FCANONICALIZE
    0U,	// G_FMINNUM
    0U,	// G_FMAXNUM
    0U,	// G_FMINNUM_IEEE
    0U,	// G_FMAXNUM_IEEE
    0U,	// G_FMINIMUM
    0U,	// G_FMAXIMUM
    0U,	// G_PTR_ADD
    0U,	// G_PTRMASK
    0U,	// G_SMIN
    0U,	// G_SMAX
    0U,	// G_UMIN
    0U,	// G_UMAX
    0U,	// G_ABS
    0U,	// G_BR
    0U,	// G_BRJT
    0U,	// G_INSERT_VECTOR_ELT
    0U,	// G_EXTRACT_VECTOR_ELT
    0U,	// G_SHUFFLE_VECTOR
    0U,	// G_CTTZ
    0U,	// G_CTTZ_ZERO_UNDEF
    0U,	// G_CTLZ
    0U,	// G_CTLZ_ZERO_UNDEF
    0U,	// G_CTPOP
    0U,	// G_BSWAP
    0U,	// G_BITREVERSE
    0U,	// G_FCEIL
    0U,	// G_FCOS
    0U,	// G_FSIN
    0U,	// G_FSQRT
    0U,	// G_FFLOOR
    0U,	// G_FRINT
    0U,	// G_FNEARBYINT
    0U,	// G_ADDRSPACE_CAST
    0U,	// G_BLOCK_ADDR
    0U,	// G_JUMP_TABLE
    0U,	// G_DYN_STACKALLOC
    0U,	// G_STRICT_FADD
    0U,	// G_STRICT_FSUB
    0U,	// G_STRICT_FMUL
    0U,	// G_STRICT_FDIV
    0U,	// G_STRICT_FREM
    0U,	// G_STRICT_FMA
    0U,	// G_STRICT_FSQRT
    0U,	// G_READ_REGISTER
    0U,	// G_WRITE_REGISTER
    0U,	// G_MEMCPY
    0U,	// G_MEMMOVE
    0U,	// G_MEMSET
    0U,	// G_VECREDUCE_SEQ_FADD
    0U,	// G_VECREDUCE_SEQ_FMUL
    0U,	// G_VECREDUCE_FADD
    0U,	// G_VECREDUCE_FMUL
    0U,	// G_VECREDUCE_FMAX
    0U,	// G_VECREDUCE_FMIN
    0U,	// G_VECREDUCE_ADD
    0U,	// G_VECREDUCE_MUL
    0U,	// G_VECREDUCE_AND
    0U,	// G_VECREDUCE_OR
    0U,	// G_VECREDUCE_XOR
    0U,	// G_VECREDUCE_SMAX
    0U,	// G_VECREDUCE_SMIN
    0U,	// G_VECREDUCE_UMAX
    0U,	// G_VECREDUCE_UMIN
    0U,	// ADJCALLSTACKDOWN
    0U,	// ADJCALLSTACKUP
    0U,	// ATOMIC_CMP_SWAP_I16
    0U,	// ATOMIC_CMP_SWAP_I32
    0U,	// ATOMIC_CMP_SWAP_I8
    0U,	// ATOMIC_LOAD_ADD_I16
    0U,	// ATOMIC_LOAD_ADD_I32
    0U,	// ATOMIC_LOAD_ADD_I8
    0U,	// ATOMIC_LOAD_AND_I16
    0U,	// ATOMIC_LOAD_AND_I32
    0U,	// ATOMIC_LOAD_AND_I8
    0U,	// ATOMIC_LOAD_NAND_I16
    0U,	// ATOMIC_LOAD_NAND_I32
    0U,	// ATOMIC_LOAD_NAND_I8
    0U,	// ATOMIC_LOAD_OR_I16
    0U,	// ATOMIC_LOAD_OR_I32
    0U,	// ATOMIC_LOAD_OR_I8
    0U,	// ATOMIC_LOAD_SUB_I16
    0U,	// ATOMIC_LOAD_SUB_I32
    0U,	// ATOMIC_LOAD_SUB_I8
    0U,	// ATOMIC_LOAD_XOR_I16
    0U,	// ATOMIC_LOAD_XOR_I32
    0U,	// ATOMIC_LOAD_XOR_I8
    0U,	// ATOMIC_SWAP_I16
    0U,	// ATOMIC_SWAP_I32
    0U,	// ATOMIC_SWAP_I8
    0U,	// CPRESTORE
    0U,	// CPU0eh_return32
    0U,	// LONG_BRANCH_ADDiu
    0U,	// LONG_BRANCH_LUi
    0U,	// LoadAddr32Imm
    0U,	// LoadAddr32Reg
    0U,	// LoadImm32Reg
    0U,	// RetLR
    0U,	// TAILCALL
    0U,	// TAILCALL_R
    2U,	// ADD
    2U,	// ADDiu
    2U,	// ADDu
    2U,	// AND
    6U,	// ANDi
    0U,	// BAL
    10U,	// BEQ
    10U,	// BNE
    0U,	// C0MOVE
    0U,	// CLO
    0U,	// CLZ
    2U,	// CMP
    2U,	// CMPu
    2U,	// GCD
    0U,	// JALR
    0U,	// JEQ
    0U,	// JGE
    0U,	// JGT
    0U,	// JLE
    0U,	// JLT
    0U,	// JMP
    0U,	// JNE
    0U,	// JR
    0U,	// JSUB
    0U,	// LB
    0U,	// LBu
    0U,	// LD
    0U,	// LEA_ADDiu
    0U,	// LH
    0U,	// LHu
    0U,	// LL
    1U,	// LUi
    0U,	// MFC0
    0U,	// MFHI
    0U,	// MFLO
    2U,	// MOVN_I_I
    2U,	// MOVZ_I_I
    0U,	// MTC0
    0U,	// MTHI
    0U,	// MTLO
    2U,	// MUL
    0U,	// MULT
    0U,	// MULTu
    0U,	// NOP
    2U,	// NOR
    2U,	// OR
    6U,	// ORi
    0U,	// RET
    2U,	// ROL
    2U,	// ROLV
    2U,	// ROR
    2U,	// RORV
    0U,	// SB
    0U,	// SC
    0U,	// SDIV
    0U,	// SH
    2U,	// SHL
    2U,	// SHLV
    2U,	// SHR
    2U,	// SHRV
    2U,	// SLT
    2U,	// SLTi
    2U,	// SLTiu
    2U,	// SLTu
    2U,	// SRA
    2U,	// SRAV
    0U,	// ST
    2U,	// SUB
    2U,	// SUBu
    0U,	// SYNC
    0U,	// UDIV
    2U,	// XOR
    6U,	// XORi
  };

  // Emit the opcode for the instruction.
  uint32_t Bits = 0;
  Bits |= OpInfo0[MI->getOpcode()] << 0;
  Bits |= OpInfo1[MI->getOpcode()] << 16;
  return {AsmStrs+(Bits & 1023)-1, Bits};

}
/// printInstruction - This method is automatically generated by tablegen
/// from the instruction set description.
void Cpu0InstPrinter::printInstruction(const MCInst *MI, uint64_t Address, raw_ostream &O) {
  O << "\t";

  auto MnemonicInfo = getMnemonic(MI);

  O << MnemonicInfo.first;

  uint32_t Bits = MnemonicInfo.second;
  assert(Bits != 0 && "Cannot print this instruction.");

  // Fragment 0 encoded into 3 bits for 5 unique commands.
  switch ((Bits >> 10) & 7) {
  default: llvm_unreachable("Invalid command number.");
  case 0:
    // DBG_VALUE, DBG_INSTR_REF, DBG_LABEL, BUNDLE, LIFETIME_START, LIFETIME_...
    return;
    break;
  case 1:
    // ADJCALLSTACKDOWN, ADJCALLSTACKUP
    printUnsignedImm(MI, 0, O);
    return;
    break;
  case 2:
    // CPRESTORE, LoadAddr32Imm, LoadAddr32Reg, LoadImm32Reg, ADD, ADDiu, ADD...
    printOperand(MI, 0, O);
    break;
  case 3:
    // BAL, JMP, JSUB
    printOperand(MI, Address, 0, O);
    return;
    break;
  case 4:
    // SC
    printOperand(MI, 1, O);
    O << ", ";
    printMemOperand(MI, 2, O);
    return;
    break;
  }


  // Fragment 1 encoded into 1 bits for 2 unique commands.
  if ((Bits >> 13) & 1) {
    // LoadAddr32Imm, LoadAddr32Reg, LoadImm32Reg, ADD, ADDiu, ADDu, AND, AND...
    O << ", ";
  } else {
    // CPRESTORE, JALR, JR, MFHI, MFLO, MTHI, MTLO, RET, SYNC
    return;
  }


  // Fragment 2 encoded into 3 bits for 5 unique commands.
  switch ((Bits >> 14) & 7) {
  default: llvm_unreachable("Invalid command number.");
  case 0:
    // LoadAddr32Imm, LoadImm32Reg, ADD, ADDiu, ADDu, AND, ANDi, BEQ, BNE, C0...
    printOperand(MI, 1, O);
    break;
  case 1:
    // LoadAddr32Reg, LB, LBu, LD, LH, LHu, LL, SB, SH, ST
    printMemOperand(MI, 1, O);
    return;
    break;
  case 2:
    // JEQ, JGE, JGT, JLE, JLT, JNE
    printOperand(MI, Address, 1, O);
    return;
    break;
  case 3:
    // LEA_ADDiu
    printMemOperandEA(MI, 1, O);
    return;
    break;
  case 4:
    // LUi
    printUnsignedImm(MI, 1, O);
    return;
    break;
  }


  // Fragment 3 encoded into 1 bits for 2 unique commands.
  if ((Bits >> 17) & 1) {
    // ADD, ADDiu, ADDu, AND, ANDi, BEQ, BNE, CMP, CMPu, GCD, MOVN_I_I, MOVZ_...
    O << ", ";
  } else {
    // LoadAddr32Imm, LoadImm32Reg, C0MOVE, CLO, CLZ, MFC0, MTC0, MULT, MULTu...
    return;
  }


  // Fragment 4 encoded into 2 bits for 3 unique commands.
  switch ((Bits >> 18) & 3) {
  default: llvm_unreachable("Invalid command number.");
  case 0:
    // ADD, ADDiu, ADDu, AND, CMP, CMPu, GCD, MOVN_I_I, MOVZ_I_I, MUL, NOR, O...
    printOperand(MI, 2, O);
    return;
    break;
  case 1:
    // ANDi, ORi, XORi
    printUnsignedImm(MI, 2, O);
    return;
    break;
  case 2:
    // BEQ, BNE
    printOperand(MI, Address, 2, O);
    return;
    break;
  }

}


/// getRegisterName - This method is automatically generated by tblgen
/// from the register set description.  This returns the assembler name
/// for the specified register.
const char *Cpu0InstPrinter::getRegisterName(unsigned RegNo) {
  assert(RegNo && RegNo < 21 && "Invalid register number!");


#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif
  static const char AsmStrs[] = {
  /* 0 */ "10\0"
  /* 3 */ "ac0\0"
  /* 7 */ "1\0"
  /* 9 */ "2\0"
  /* 11 */ "3\0"
  /* 13 */ "4\0"
  /* 15 */ "5\0"
  /* 17 */ "7\0"
  /* 19 */ "8\0"
  /* 21 */ "t9\0"
  /* 24 */ "epc\0"
  /* 28 */ "zero\0"
  /* 33 */ "fp\0"
  /* 36 */ "gp\0"
  /* 39 */ "sp\0"
  /* 42 */ "lr\0"
  /* 45 */ "sw\0"
};
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

  static const uint8_t RegAsmOffset[] = {
    7, 24, 33, 36, 3, 3, 42, 25, 39, 45, 28, 13, 15, 22, 
    0, 17, 19, 21, 9, 11, 
  };

  assert (*(AsmStrs+RegAsmOffset[RegNo-1]) &&
          "Invalid alt name index for register!");
  return AsmStrs+RegAsmOffset[RegNo-1];
}

#ifdef PRINT_ALIAS_INSTR
#undef PRINT_ALIAS_INSTR

bool Cpu0InstPrinter::printAliasInstr(const MCInst *MI, uint64_t Address, raw_ostream &OS) {
  static const PatternsForOpcode OpToPatterns[] = {
    {Cpu0::ADDu, 0, 1 },
    {Cpu0::SYNC, 1, 1 },
  };

  static const AliasPattern Patterns[] = {
    // Cpu0::ADDu - 0
    {0, 0, 3, 3 },
    // Cpu0::SYNC - 1
    {12, 3, 1, 1 },
  };

  static const AliasPatternCond Conds[] = {
    // (ADDu GPROut:$dst, GPROut:$src, ZERO) - 0
    {AliasPatternCond::K_RegClass, Cpu0::GPROutRegClassID},
    {AliasPatternCond::K_RegClass, Cpu0::GPROutRegClassID},
    {AliasPatternCond::K_Reg, Cpu0::ZERO},
    // (SYNC 0) - 3
    {AliasPatternCond::K_Imm, uint32_t(0)},
  };

  static const char AsmStrings[] =
    /* 0 */ "move $\x01, $\x02\0"
    /* 12 */ "sync\0"
  ;

#ifndef NDEBUG
  static struct SortCheck {
    SortCheck(ArrayRef<PatternsForOpcode> OpToPatterns) {
      assert(std::is_sorted(
                 OpToPatterns.begin(), OpToPatterns.end(),
                 [](const PatternsForOpcode &L, const PatternsForOpcode &R) {
                   return L.Opcode < R.Opcode;
                 }) &&
             "tablegen failed to sort opcode patterns");
    }
  } sortCheckVar(OpToPatterns);
#endif

  AliasMatchingData M {
    makeArrayRef(OpToPatterns),
    makeArrayRef(Patterns),
    makeArrayRef(Conds),
    StringRef(AsmStrings, array_lengthof(AsmStrings)),
    nullptr,
  };
  const char *AsmString = matchAliasPatterns(MI, nullptr, M);
  if (!AsmString) return false;

  unsigned I = 0;
  while (AsmString[I] != ' ' && AsmString[I] != '\t' &&
         AsmString[I] != '$' && AsmString[I] != '\0')
    ++I;
  OS << '\t' << StringRef(AsmString, I);
  if (AsmString[I] != '\0') {
    if (AsmString[I] == ' ' || AsmString[I] == '\t') {
      OS << '\t';
      ++I;
    }
    do {
      if (AsmString[I] == '$') {
        ++I;
        if (AsmString[I] == (char)0xff) {
          ++I;
          int OpIdx = AsmString[I++] - 1;
          int PrintMethodIdx = AsmString[I++] - 1;
          printCustomAliasOperand(MI, Address, OpIdx, PrintMethodIdx, OS);
        } else
          printOperand(MI, unsigned(AsmString[I++]) - 1, OS);
      } else {
        OS << AsmString[I++];
      }
    } while (AsmString[I] != '\0');
  }

  return true;
}

void Cpu0InstPrinter::printCustomAliasOperand(
         const MCInst *MI, uint64_t Address, unsigned OpIdx,
         unsigned PrintMethodIdx,
         raw_ostream &OS) {
  llvm_unreachable("Unknown PrintMethod kind");
}

#endif // PRINT_ALIAS_INSTR
