/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|*  * Cpu0 Disassembler                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include "llvm/MC/MCInst.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/raw_ostream.h"
#include <assert.h>

namespace llvm {

// Helper functions for extracting fields from encoded instructions.
// InsnType must either be integral or an APInt-like object that must:
// * Have a static const max_size_in_bits equal to the number of bits in the
//   encoding.
// * be default-constructible and copy-constructible
// * be constructible from a uint64_t
// * be constructible from an APInt (this can be private)
// * Support getBitsSet(loBit, hiBit)
// * be convertible to uint64_t
// * Support the ~, &, ==, !=, and |= operators with other objects of the same type
// * Support shift (<<, >>) with signed and unsigned integers on the RHS
// * Support put (<<) to raw_ostream&
template <typename InsnType>
#if defined(_MSC_VER) && !defined(__clang__)
__declspec(noinline)
#endif
static InsnType fieldFromInstruction(InsnType insn, unsigned startBit,
                                     unsigned numBits, std::true_type) {
  assert(startBit + numBits <= 64 && "Cannot support >64-bit extractions!");
  assert(startBit + numBits <= (sizeof(InsnType) * 8) &&
         "Instruction field out of bounds!");
  InsnType fieldMask;
  if (numBits == sizeof(InsnType) * 8)
    fieldMask = (InsnType)(-1LL);
  else
    fieldMask = (((InsnType)1 << numBits) - 1) << startBit;
  return (insn & fieldMask) >> startBit;
}

template <typename InsnType>
static InsnType fieldFromInstruction(InsnType insn, unsigned startBit,
                                     unsigned numBits, std::false_type) {
  assert(startBit + numBits <= InsnType::max_size_in_bits && "Instruction field out of bounds!");
  InsnType fieldMask = InsnType::getBitsSet(0, numBits);
  return (insn >> startBit) & fieldMask;
}

template <typename InsnType>
static InsnType fieldFromInstruction(InsnType insn, unsigned startBit,
                                     unsigned numBits) {
  return fieldFromInstruction(insn, startBit, numBits, std::is_integral<InsnType>());
}

static const uint8_t DecoderTableCpu032[] = {
/* 0 */       MCD::OPC_ExtractField, 24, 8,  // Inst{31-24} ...
/* 3 */       MCD::OPC_FilterValue, 0, 11, 0, 0, // Skip to: 19
/* 8 */       MCD::OPC_CheckField, 0, 24, 0, 149, 4, 0, // Skip to: 1188
/* 15 */      MCD::OPC_Decode, 178, 2, 0, // Opcode: NOP
/* 19 */      MCD::OPC_FilterValue, 1, 4, 0, 0, // Skip to: 28
/* 24 */      MCD::OPC_Decode, 161, 2, 1, // Opcode: LD
/* 28 */      MCD::OPC_FilterValue, 2, 4, 0, 0, // Skip to: 37
/* 33 */      MCD::OPC_Decode, 201, 2, 1, // Opcode: ST
/* 37 */      MCD::OPC_FilterValue, 3, 9, 0, 0, // Skip to: 51
/* 42 */      MCD::OPC_CheckPredicate, 0, 117, 4, 0, // Skip to: 1188
/* 47 */      MCD::OPC_Decode, 159, 2, 1, // Opcode: LB
/* 51 */      MCD::OPC_FilterValue, 4, 9, 0, 0, // Skip to: 65
/* 56 */      MCD::OPC_CheckPredicate, 0, 103, 4, 0, // Skip to: 1188
/* 61 */      MCD::OPC_Decode, 160, 2, 1, // Opcode: LBu
/* 65 */      MCD::OPC_FilterValue, 5, 9, 0, 0, // Skip to: 79
/* 70 */      MCD::OPC_CheckPredicate, 0, 89, 4, 0, // Skip to: 1188
/* 75 */      MCD::OPC_Decode, 187, 2, 1, // Opcode: SB
/* 79 */      MCD::OPC_FilterValue, 6, 9, 0, 0, // Skip to: 93
/* 84 */      MCD::OPC_CheckPredicate, 0, 75, 4, 0, // Skip to: 1188
/* 89 */      MCD::OPC_Decode, 163, 2, 1, // Opcode: LH
/* 93 */      MCD::OPC_FilterValue, 7, 9, 0, 0, // Skip to: 107
/* 98 */      MCD::OPC_CheckPredicate, 0, 61, 4, 0, // Skip to: 1188
/* 103 */     MCD::OPC_Decode, 164, 2, 1, // Opcode: LHu
/* 107 */     MCD::OPC_FilterValue, 8, 9, 0, 0, // Skip to: 121
/* 112 */     MCD::OPC_CheckPredicate, 0, 47, 4, 0, // Skip to: 1188
/* 117 */     MCD::OPC_Decode, 190, 2, 1, // Opcode: SH
/* 121 */     MCD::OPC_FilterValue, 9, 4, 0, 0, // Skip to: 130
/* 126 */     MCD::OPC_Decode, 136, 2, 2, // Opcode: ADDiu
/* 130 */     MCD::OPC_FilterValue, 10, 16, 0, 0, // Skip to: 151
/* 135 */     MCD::OPC_CheckPredicate, 1, 24, 4, 0, // Skip to: 1188
/* 140 */     MCD::OPC_CheckField, 0, 12, 0, 17, 4, 0, // Skip to: 1188
/* 147 */     MCD::OPC_Decode, 171, 2, 3, // Opcode: MOVZ_I_I
/* 151 */     MCD::OPC_FilterValue, 11, 16, 0, 0, // Skip to: 172
/* 156 */     MCD::OPC_CheckPredicate, 1, 3, 4, 0, // Skip to: 1188
/* 161 */     MCD::OPC_CheckField, 0, 12, 0, 252, 3, 0, // Skip to: 1188
/* 168 */     MCD::OPC_Decode, 170, 2, 3, // Opcode: MOVN_I_I
/* 172 */     MCD::OPC_FilterValue, 12, 9, 0, 0, // Skip to: 186
/* 177 */     MCD::OPC_CheckPredicate, 2, 238, 3, 0, // Skip to: 1188
/* 182 */     MCD::OPC_Decode, 139, 2, 4, // Opcode: ANDi
/* 186 */     MCD::OPC_FilterValue, 13, 9, 0, 0, // Skip to: 200
/* 191 */     MCD::OPC_CheckPredicate, 3, 224, 3, 0, // Skip to: 1188
/* 196 */     MCD::OPC_Decode, 181, 2, 4, // Opcode: ORi
/* 200 */     MCD::OPC_FilterValue, 14, 9, 0, 0, // Skip to: 214
/* 205 */     MCD::OPC_CheckPredicate, 2, 210, 3, 0, // Skip to: 1188
/* 210 */     MCD::OPC_Decode, 207, 2, 4, // Opcode: XORi
/* 214 */     MCD::OPC_FilterValue, 15, 16, 0, 0, // Skip to: 235
/* 219 */     MCD::OPC_CheckPredicate, 3, 196, 3, 0, // Skip to: 1188
/* 224 */     MCD::OPC_CheckField, 16, 4, 0, 189, 3, 0, // Skip to: 1188
/* 231 */     MCD::OPC_Decode, 166, 2, 5, // Opcode: LUi
/* 235 */     MCD::OPC_FilterValue, 17, 11, 0, 0, // Skip to: 251
/* 240 */     MCD::OPC_CheckField, 0, 12, 0, 173, 3, 0, // Skip to: 1188
/* 247 */     MCD::OPC_Decode, 137, 2, 6, // Opcode: ADDu
/* 251 */     MCD::OPC_FilterValue, 18, 11, 0, 0, // Skip to: 267
/* 256 */     MCD::OPC_CheckField, 0, 12, 0, 157, 3, 0, // Skip to: 1188
/* 263 */     MCD::OPC_Decode, 203, 2, 6, // Opcode: SUBu
/* 267 */     MCD::OPC_FilterValue, 19, 11, 0, 0, // Skip to: 283
/* 272 */     MCD::OPC_CheckField, 0, 12, 0, 141, 3, 0, // Skip to: 1188
/* 279 */     MCD::OPC_Decode, 135, 2, 6, // Opcode: ADD
/* 283 */     MCD::OPC_FilterValue, 20, 11, 0, 0, // Skip to: 299
/* 288 */     MCD::OPC_CheckField, 0, 12, 0, 125, 3, 0, // Skip to: 1188
/* 295 */     MCD::OPC_Decode, 202, 2, 6, // Opcode: SUB
/* 299 */     MCD::OPC_FilterValue, 21, 16, 0, 0, // Skip to: 320
/* 304 */     MCD::OPC_CheckPredicate, 0, 111, 3, 0, // Skip to: 1188
/* 309 */     MCD::OPC_CheckField, 0, 16, 0, 104, 3, 0, // Skip to: 1188
/* 316 */     MCD::OPC_Decode, 145, 2, 7, // Opcode: CLZ
/* 320 */     MCD::OPC_FilterValue, 22, 16, 0, 0, // Skip to: 341
/* 325 */     MCD::OPC_CheckPredicate, 0, 90, 3, 0, // Skip to: 1188
/* 330 */     MCD::OPC_CheckField, 0, 16, 0, 83, 3, 0, // Skip to: 1188
/* 337 */     MCD::OPC_Decode, 144, 2, 7, // Opcode: CLO
/* 341 */     MCD::OPC_FilterValue, 23, 16, 0, 0, // Skip to: 362
/* 346 */     MCD::OPC_CheckPredicate, 4, 69, 3, 0, // Skip to: 1188
/* 351 */     MCD::OPC_CheckField, 0, 12, 0, 62, 3, 0, // Skip to: 1188
/* 358 */     MCD::OPC_Decode, 175, 2, 6, // Opcode: MUL
/* 362 */     MCD::OPC_FilterValue, 24, 16, 0, 0, // Skip to: 383
/* 367 */     MCD::OPC_CheckPredicate, 2, 48, 3, 0, // Skip to: 1188
/* 372 */     MCD::OPC_CheckField, 0, 12, 0, 41, 3, 0, // Skip to: 1188
/* 379 */     MCD::OPC_Decode, 138, 2, 6, // Opcode: AND
/* 383 */     MCD::OPC_FilterValue, 25, 16, 0, 0, // Skip to: 404
/* 388 */     MCD::OPC_CheckPredicate, 2, 27, 3, 0, // Skip to: 1188
/* 393 */     MCD::OPC_CheckField, 0, 12, 0, 20, 3, 0, // Skip to: 1188
/* 400 */     MCD::OPC_Decode, 180, 2, 6, // Opcode: OR
/* 404 */     MCD::OPC_FilterValue, 26, 16, 0, 0, // Skip to: 425
/* 409 */     MCD::OPC_CheckPredicate, 2, 6, 3, 0, // Skip to: 1188
/* 414 */     MCD::OPC_CheckField, 0, 12, 0, 255, 2, 0, // Skip to: 1188
/* 421 */     MCD::OPC_Decode, 206, 2, 6, // Opcode: XOR
/* 425 */     MCD::OPC_FilterValue, 27, 16, 0, 0, // Skip to: 446
/* 430 */     MCD::OPC_CheckPredicate, 2, 241, 2, 0, // Skip to: 1188
/* 435 */     MCD::OPC_CheckField, 0, 12, 0, 234, 2, 0, // Skip to: 1188
/* 442 */     MCD::OPC_Decode, 179, 2, 8, // Opcode: NOR
/* 446 */     MCD::OPC_FilterValue, 28, 16, 0, 0, // Skip to: 467
/* 451 */     MCD::OPC_CheckPredicate, 4, 220, 2, 0, // Skip to: 1188
/* 456 */     MCD::OPC_CheckField, 12, 4, 0, 213, 2, 0, // Skip to: 1188
/* 463 */     MCD::OPC_Decode, 183, 2, 9, // Opcode: ROL
/* 467 */     MCD::OPC_FilterValue, 29, 16, 0, 0, // Skip to: 488
/* 472 */     MCD::OPC_CheckPredicate, 4, 199, 2, 0, // Skip to: 1188
/* 477 */     MCD::OPC_CheckField, 12, 4, 0, 192, 2, 0, // Skip to: 1188
/* 484 */     MCD::OPC_Decode, 185, 2, 9, // Opcode: ROR
/* 488 */     MCD::OPC_FilterValue, 30, 16, 0, 0, // Skip to: 509
/* 493 */     MCD::OPC_CheckPredicate, 3, 178, 2, 0, // Skip to: 1188
/* 498 */     MCD::OPC_CheckField, 12, 4, 0, 171, 2, 0, // Skip to: 1188
/* 505 */     MCD::OPC_Decode, 191, 2, 9, // Opcode: SHL
/* 509 */     MCD::OPC_FilterValue, 31, 16, 0, 0, // Skip to: 530
/* 514 */     MCD::OPC_CheckPredicate, 4, 157, 2, 0, // Skip to: 1188
/* 519 */     MCD::OPC_CheckField, 12, 4, 0, 150, 2, 0, // Skip to: 1188
/* 526 */     MCD::OPC_Decode, 193, 2, 9, // Opcode: SHR
/* 530 */     MCD::OPC_FilterValue, 32, 16, 0, 0, // Skip to: 551
/* 535 */     MCD::OPC_CheckPredicate, 4, 136, 2, 0, // Skip to: 1188
/* 540 */     MCD::OPC_CheckField, 12, 4, 0, 129, 2, 0, // Skip to: 1188
/* 547 */     MCD::OPC_Decode, 199, 2, 9, // Opcode: SRA
/* 551 */     MCD::OPC_FilterValue, 33, 16, 0, 0, // Skip to: 572
/* 556 */     MCD::OPC_CheckPredicate, 4, 115, 2, 0, // Skip to: 1188
/* 561 */     MCD::OPC_CheckField, 0, 12, 0, 108, 2, 0, // Skip to: 1188
/* 568 */     MCD::OPC_Decode, 200, 2, 6, // Opcode: SRAV
/* 572 */     MCD::OPC_FilterValue, 34, 16, 0, 0, // Skip to: 593
/* 577 */     MCD::OPC_CheckPredicate, 4, 94, 2, 0, // Skip to: 1188
/* 582 */     MCD::OPC_CheckField, 0, 12, 0, 87, 2, 0, // Skip to: 1188
/* 589 */     MCD::OPC_Decode, 192, 2, 6, // Opcode: SHLV
/* 593 */     MCD::OPC_FilterValue, 35, 16, 0, 0, // Skip to: 614
/* 598 */     MCD::OPC_CheckPredicate, 4, 73, 2, 0, // Skip to: 1188
/* 603 */     MCD::OPC_CheckField, 0, 12, 0, 66, 2, 0, // Skip to: 1188
/* 610 */     MCD::OPC_Decode, 194, 2, 6, // Opcode: SHRV
/* 614 */     MCD::OPC_FilterValue, 36, 16, 0, 0, // Skip to: 635
/* 619 */     MCD::OPC_CheckPredicate, 4, 52, 2, 0, // Skip to: 1188
/* 624 */     MCD::OPC_CheckField, 0, 12, 0, 45, 2, 0, // Skip to: 1188
/* 631 */     MCD::OPC_Decode, 184, 2, 6, // Opcode: ROLV
/* 635 */     MCD::OPC_FilterValue, 37, 16, 0, 0, // Skip to: 656
/* 640 */     MCD::OPC_CheckPredicate, 4, 31, 2, 0, // Skip to: 1188
/* 645 */     MCD::OPC_CheckField, 0, 12, 0, 24, 2, 0, // Skip to: 1188
/* 652 */     MCD::OPC_Decode, 186, 2, 6, // Opcode: RORV
/* 656 */     MCD::OPC_FilterValue, 38, 4, 0, 0, // Skip to: 665
/* 661 */     MCD::OPC_Decode, 196, 2, 2, // Opcode: SLTi
/* 665 */     MCD::OPC_FilterValue, 39, 4, 0, 0, // Skip to: 674
/* 670 */     MCD::OPC_Decode, 197, 2, 2, // Opcode: SLTiu
/* 674 */     MCD::OPC_FilterValue, 40, 11, 0, 0, // Skip to: 690
/* 679 */     MCD::OPC_CheckField, 0, 12, 0, 246, 1, 0, // Skip to: 1188
/* 686 */     MCD::OPC_Decode, 195, 2, 6, // Opcode: SLT
/* 690 */     MCD::OPC_FilterValue, 41, 11, 0, 0, // Skip to: 706
/* 695 */     MCD::OPC_CheckField, 0, 12, 0, 230, 1, 0, // Skip to: 1188
/* 702 */     MCD::OPC_Decode, 198, 2, 6, // Opcode: SLTu
/* 706 */     MCD::OPC_FilterValue, 42, 11, 0, 0, // Skip to: 722
/* 711 */     MCD::OPC_CheckField, 0, 12, 0, 214, 1, 0, // Skip to: 1188
/* 718 */     MCD::OPC_Decode, 146, 2, 10, // Opcode: CMP
/* 722 */     MCD::OPC_FilterValue, 43, 11, 0, 0, // Skip to: 738
/* 727 */     MCD::OPC_CheckField, 0, 12, 0, 198, 1, 0, // Skip to: 1188
/* 734 */     MCD::OPC_Decode, 147, 2, 10, // Opcode: CMPu
/* 738 */     MCD::OPC_FilterValue, 48, 4, 0, 0, // Skip to: 747
/* 743 */     MCD::OPC_Decode, 150, 2, 11, // Opcode: JEQ
/* 747 */     MCD::OPC_FilterValue, 49, 4, 0, 0, // Skip to: 756
/* 752 */     MCD::OPC_Decode, 156, 2, 11, // Opcode: JNE
/* 756 */     MCD::OPC_FilterValue, 50, 4, 0, 0, // Skip to: 765
/* 761 */     MCD::OPC_Decode, 154, 2, 11, // Opcode: JLT
/* 765 */     MCD::OPC_FilterValue, 51, 4, 0, 0, // Skip to: 774
/* 770 */     MCD::OPC_Decode, 152, 2, 11, // Opcode: JGT
/* 774 */     MCD::OPC_FilterValue, 52, 4, 0, 0, // Skip to: 783
/* 779 */     MCD::OPC_Decode, 153, 2, 11, // Opcode: JLE
/* 783 */     MCD::OPC_FilterValue, 53, 4, 0, 0, // Skip to: 792
/* 788 */     MCD::OPC_Decode, 151, 2, 11, // Opcode: JGE
/* 792 */     MCD::OPC_FilterValue, 54, 9, 0, 0, // Skip to: 806
/* 797 */     MCD::OPC_CheckPredicate, 5, 130, 1, 0, // Skip to: 1188
/* 802 */     MCD::OPC_Decode, 155, 2, 12, // Opcode: JMP
/* 806 */     MCD::OPC_FilterValue, 55, 4, 0, 0, // Skip to: 815
/* 811 */     MCD::OPC_Decode, 141, 2, 13, // Opcode: BEQ
/* 815 */     MCD::OPC_FilterValue, 56, 4, 0, 0, // Skip to: 824
/* 820 */     MCD::OPC_Decode, 142, 2, 13, // Opcode: BNE
/* 824 */     MCD::OPC_FilterValue, 57, 23, 0, 0, // Skip to: 852
/* 829 */     MCD::OPC_CheckPredicate, 6, 98, 1, 0, // Skip to: 1188
/* 834 */     MCD::OPC_CheckField, 20, 4, 14, 91, 1, 0, // Skip to: 1188
/* 841 */     MCD::OPC_CheckField, 0, 16, 0, 84, 1, 0, // Skip to: 1188
/* 848 */     MCD::OPC_Decode, 149, 2, 14, // Opcode: JALR
/* 852 */     MCD::OPC_FilterValue, 58, 4, 0, 0, // Skip to: 861
/* 857 */     MCD::OPC_Decode, 140, 2, 12, // Opcode: BAL
/* 861 */     MCD::OPC_FilterValue, 59, 9, 0, 0, // Skip to: 875
/* 866 */     MCD::OPC_CheckPredicate, 6, 61, 1, 0, // Skip to: 1188
/* 871 */     MCD::OPC_Decode, 158, 2, 15, // Opcode: JSUB
/* 875 */     MCD::OPC_FilterValue, 60, 11, 0, 0, // Skip to: 891
/* 880 */     MCD::OPC_CheckField, 0, 20, 0, 45, 1, 0, // Skip to: 1188
/* 887 */     MCD::OPC_Decode, 157, 2, 16, // Opcode: JR
/* 891 */     MCD::OPC_FilterValue, 65, 16, 0, 0, // Skip to: 912
/* 896 */     MCD::OPC_CheckPredicate, 4, 31, 1, 0, // Skip to: 1188
/* 901 */     MCD::OPC_CheckField, 0, 16, 0, 24, 1, 0, // Skip to: 1188
/* 908 */     MCD::OPC_Decode, 176, 2, 17, // Opcode: MULT
/* 912 */     MCD::OPC_FilterValue, 66, 16, 0, 0, // Skip to: 933
/* 917 */     MCD::OPC_CheckPredicate, 4, 10, 1, 0, // Skip to: 1188
/* 922 */     MCD::OPC_CheckField, 0, 16, 0, 3, 1, 0, // Skip to: 1188
/* 929 */     MCD::OPC_Decode, 177, 2, 17, // Opcode: MULTu
/* 933 */     MCD::OPC_FilterValue, 67, 16, 0, 0, // Skip to: 954
/* 938 */     MCD::OPC_CheckPredicate, 4, 245, 0, 0, // Skip to: 1188
/* 943 */     MCD::OPC_CheckField, 0, 16, 0, 238, 0, 0, // Skip to: 1188
/* 950 */     MCD::OPC_Decode, 189, 2, 17, // Opcode: SDIV
/* 954 */     MCD::OPC_FilterValue, 68, 16, 0, 0, // Skip to: 975
/* 959 */     MCD::OPC_CheckPredicate, 4, 224, 0, 0, // Skip to: 1188
/* 964 */     MCD::OPC_CheckField, 0, 16, 0, 217, 0, 0, // Skip to: 1188
/* 971 */     MCD::OPC_Decode, 205, 2, 17, // Opcode: UDIV
/* 975 */     MCD::OPC_FilterValue, 70, 16, 0, 0, // Skip to: 996
/* 980 */     MCD::OPC_CheckPredicate, 4, 203, 0, 0, // Skip to: 1188
/* 985 */     MCD::OPC_CheckField, 0, 20, 0, 196, 0, 0, // Skip to: 1188
/* 992 */     MCD::OPC_Decode, 168, 2, 18, // Opcode: MFHI
/* 996 */     MCD::OPC_FilterValue, 71, 16, 0, 0, // Skip to: 1017
/* 1001 */    MCD::OPC_CheckPredicate, 4, 182, 0, 0, // Skip to: 1188
/* 1006 */    MCD::OPC_CheckField, 0, 20, 0, 175, 0, 0, // Skip to: 1188
/* 1013 */    MCD::OPC_Decode, 169, 2, 18, // Opcode: MFLO
/* 1017 */    MCD::OPC_FilterValue, 72, 16, 0, 0, // Skip to: 1038
/* 1022 */    MCD::OPC_CheckPredicate, 4, 161, 0, 0, // Skip to: 1188
/* 1027 */    MCD::OPC_CheckField, 0, 20, 0, 154, 0, 0, // Skip to: 1188
/* 1034 */    MCD::OPC_Decode, 173, 2, 18, // Opcode: MTHI
/* 1038 */    MCD::OPC_FilterValue, 73, 16, 0, 0, // Skip to: 1059
/* 1043 */    MCD::OPC_CheckPredicate, 4, 140, 0, 0, // Skip to: 1188
/* 1048 */    MCD::OPC_CheckField, 0, 20, 0, 133, 0, 0, // Skip to: 1188
/* 1055 */    MCD::OPC_Decode, 174, 2, 18, // Opcode: MTLO
/* 1059 */    MCD::OPC_FilterValue, 80, 16, 0, 0, // Skip to: 1080
/* 1064 */    MCD::OPC_CheckPredicate, 4, 119, 0, 0, // Skip to: 1188
/* 1069 */    MCD::OPC_CheckField, 0, 16, 0, 112, 0, 0, // Skip to: 1188
/* 1076 */    MCD::OPC_Decode, 167, 2, 19, // Opcode: MFC0
/* 1080 */    MCD::OPC_FilterValue, 81, 16, 0, 0, // Skip to: 1101
/* 1085 */    MCD::OPC_CheckPredicate, 4, 98, 0, 0, // Skip to: 1188
/* 1090 */    MCD::OPC_CheckField, 0, 16, 0, 91, 0, 0, // Skip to: 1188
/* 1097 */    MCD::OPC_Decode, 172, 2, 20, // Opcode: MTC0
/* 1101 */    MCD::OPC_FilterValue, 82, 16, 0, 0, // Skip to: 1122
/* 1106 */    MCD::OPC_CheckPredicate, 4, 77, 0, 0, // Skip to: 1188
/* 1111 */    MCD::OPC_CheckField, 0, 16, 0, 70, 0, 0, // Skip to: 1188
/* 1118 */    MCD::OPC_Decode, 143, 2, 21, // Opcode: C0MOVE
/* 1122 */    MCD::OPC_FilterValue, 96, 43, 0, 0, // Skip to: 1170
/* 1127 */    MCD::OPC_ExtractField, 0, 6,  // Inst{5-0} ...
/* 1130 */    MCD::OPC_FilterValue, 0, 53, 0, 0, // Skip to: 1188
/* 1135 */    MCD::OPC_ExtractField, 11, 1,  // Inst{11} ...
/* 1138 */    MCD::OPC_FilterValue, 0, 45, 0, 0, // Skip to: 1188
/* 1143 */    MCD::OPC_CheckField, 6, 5, 0, 4, 0, 0, // Skip to: 1154
/* 1150 */    MCD::OPC_Decode, 148, 2, 6, // Opcode: GCD
/* 1154 */    MCD::OPC_CheckPredicate, 7, 29, 0, 0, // Skip to: 1188
/* 1159 */    MCD::OPC_CheckField, 12, 12, 0, 22, 0, 0, // Skip to: 1188
/* 1166 */    MCD::OPC_Decode, 204, 2, 22, // Opcode: SYNC
/* 1170 */    MCD::OPC_FilterValue, 97, 4, 0, 0, // Skip to: 1179
/* 1175 */    MCD::OPC_Decode, 165, 2, 1, // Opcode: LL
/* 1179 */    MCD::OPC_FilterValue, 98, 4, 0, 0, // Skip to: 1188
/* 1184 */    MCD::OPC_Decode, 188, 2, 1, // Opcode: SC
/* 1188 */    MCD::OPC_Fail,
  0
};

static bool checkDecoderPredicate(unsigned Idx, const FeatureBitset &Bits) {
  switch (Idx) {
  default: llvm_unreachable("Invalid index!");
  case 0:
    return (Bits[Cpu0::FeatureChapter7_1]);
  case 1:
    return (Bits[Cpu0::FeatureChapter8_2]);
  case 2:
    return (Bits[Cpu0::FeatureChapter4_2]);
  case 3:
    return (Bits[Cpu0::FeatureChapter3_5]);
  case 4:
    return (Bits[Cpu0::FeatureChapter4_1]);
  case 5:
    return (Bits[Cpu0::FeatureChapter8_1]);
  case 6:
    return (Bits[Cpu0::FeatureChapter9_1]);
  case 7:
    return (Bits[Cpu0::FeatureChapter12_1]);
  }
}

template <typename InsnType>
static DecodeStatus decodeToMCInst(DecodeStatus S, unsigned Idx, InsnType insn, MCInst &MI,
                                   uint64_t Address, const void *Decoder, bool &DecodeComplete) {
  DecodeComplete = true;
  InsnType tmp;
  switch (Idx) {
  default: llvm_unreachable("Invalid index!");
  case 0:
    return S;
  case 1:
    if (DecodeMem(MI, insn, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    return S;
  case 2:
    tmp = fieldFromInstruction(insn, 20, 4);
    if (DecodeGPROutRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 16, 4);
    if (DecodeCPURegsRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 0, 16);
    if (DecodeSimm16(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    return S;
  case 3:
    tmp = fieldFromInstruction(insn, 20, 4);
    if (DecodeCPURegsRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 16, 4);
    if (DecodeCPURegsRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 4);
    if (DecodeCPURegsRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 20, 4);
    if (DecodeCPURegsRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    return S;
  case 4:
    tmp = fieldFromInstruction(insn, 20, 4);
    if (DecodeGPROutRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 16, 4);
    if (DecodeCPURegsRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 0, 16);
    MI.addOperand(MCOperand::createImm(tmp));
    return S;
  case 5:
    tmp = fieldFromInstruction(insn, 20, 4);
    if (DecodeGPROutRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 0, 16);
    MI.addOperand(MCOperand::createImm(tmp));
    return S;
  case 6:
    tmp = fieldFromInstruction(insn, 20, 4);
    if (DecodeGPROutRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 16, 4);
    if (DecodeCPURegsRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 4);
    if (DecodeCPURegsRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    return S;
  case 7:
    tmp = fieldFromInstruction(insn, 20, 4);
    if (DecodeGPROutRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 16, 4);
    if (DecodeCPURegsRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    return S;
  case 8:
    tmp = fieldFromInstruction(insn, 20, 4);
    if (DecodeCPURegsRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 16, 4);
    if (DecodeCPURegsRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 4);
    if (DecodeCPURegsRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    return S;
  case 9:
    tmp = fieldFromInstruction(insn, 20, 4);
    if (DecodeGPROutRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 16, 4);
    if (DecodeCPURegsRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 0, 12);
    MI.addOperand(MCOperand::createImm(tmp));
    return S;
  case 10:
    tmp = fieldFromInstruction(insn, 20, 4);
    if (DecodeSRRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 16, 4);
    if (DecodeCPURegsRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 12, 4);
    if (DecodeCPURegsRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    return S;
  case 11:
    tmp = fieldFromInstruction(insn, 0, 24);
    if (DecodeBranch24Target(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    return S;
  case 12:
    tmp = fieldFromInstruction(insn, 0, 24);
    MI.addOperand(MCOperand::createImm(tmp));
    return S;
  case 13:
    tmp = fieldFromInstruction(insn, 20, 4);
    if (DecodeGPROutRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 16, 4);
    if (DecodeGPROutRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 0, 16);
    if (DecodeBranch16Target(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    return S;
  case 14:
    tmp = fieldFromInstruction(insn, 16, 4);
    if (DecodeGPROutRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    return S;
  case 15:
    if (DecodeJumpTarget(MI, insn, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    return S;
  case 16:
    if (DecodeJumpFR(MI, insn, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    return S;
  case 17:
    tmp = fieldFromInstruction(insn, 20, 4);
    if (DecodeCPURegsRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 16, 4);
    if (DecodeCPURegsRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    return S;
  case 18:
    tmp = fieldFromInstruction(insn, 20, 4);
    if (DecodeCPURegsRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    return S;
  case 19:
    tmp = fieldFromInstruction(insn, 20, 4);
    if (DecodeCPURegsRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 16, 4);
    if (DecodeC0RegsRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    return S;
  case 20:
    tmp = fieldFromInstruction(insn, 20, 4);
    if (DecodeC0RegsRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 16, 4);
    if (DecodeCPURegsRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    return S;
  case 21:
    tmp = fieldFromInstruction(insn, 20, 4);
    if (DecodeC0RegsRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    tmp = fieldFromInstruction(insn, 16, 4);
    if (DecodeC0RegsRegisterClass(MI, tmp, Address, Decoder) == MCDisassembler::Fail) { return MCDisassembler::Fail; }
    return S;
  case 22:
    tmp = fieldFromInstruction(insn, 6, 5);
    MI.addOperand(MCOperand::createImm(tmp));
    return S;
  }
}

template <typename InsnType>
static DecodeStatus decodeInstruction(const uint8_t DecodeTable[], MCInst &MI,
                                      InsnType insn, uint64_t Address,
                                      const void *DisAsm,
                                      const MCSubtargetInfo &STI) {
  const FeatureBitset &Bits = STI.getFeatureBits();

  const uint8_t *Ptr = DecodeTable;
  InsnType CurFieldValue = 0;
  DecodeStatus S = MCDisassembler::Success;
  while (true) {
    ptrdiff_t Loc = Ptr - DecodeTable;
    switch (*Ptr) {
    default:
      errs() << Loc << ": Unexpected decode table opcode!\n";
      return MCDisassembler::Fail;
    case MCD::OPC_ExtractField: {
      unsigned Start = *++Ptr;
      unsigned Len = *++Ptr;
      ++Ptr;
      CurFieldValue = fieldFromInstruction(insn, Start, Len);
      LLVM_DEBUG(dbgs() << Loc << ": OPC_ExtractField(" << Start << ", "
                   << Len << "): " << CurFieldValue << "\n");
      break;
    }
    case MCD::OPC_FilterValue: {
      // Decode the field value.
      unsigned Len;
      InsnType Val = decodeULEB128(++Ptr, &Len);
      Ptr += Len;
      // NumToSkip is a plain 24-bit integer.
      unsigned NumToSkip = *Ptr++;
      NumToSkip |= (*Ptr++) << 8;
      NumToSkip |= (*Ptr++) << 16;

      // Perform the filter operation.
      if (Val != CurFieldValue)
        Ptr += NumToSkip;
      LLVM_DEBUG(dbgs() << Loc << ": OPC_FilterValue(" << Val << ", " << NumToSkip
                   << "): " << ((Val != CurFieldValue) ? "FAIL:" : "PASS:")
                   << " continuing at " << (Ptr - DecodeTable) << "\n");

      break;
    }
    case MCD::OPC_CheckField: {
      unsigned Start = *++Ptr;
      unsigned Len = *++Ptr;
      InsnType FieldValue = fieldFromInstruction(insn, Start, Len);
      // Decode the field value.
      InsnType ExpectedValue = decodeULEB128(++Ptr, &Len);
      Ptr += Len;
      // NumToSkip is a plain 24-bit integer.
      unsigned NumToSkip = *Ptr++;
      NumToSkip |= (*Ptr++) << 8;
      NumToSkip |= (*Ptr++) << 16;

      // If the actual and expected values don't match, skip.
      if (ExpectedValue != FieldValue)
        Ptr += NumToSkip;
      LLVM_DEBUG(dbgs() << Loc << ": OPC_CheckField(" << Start << ", "
                   << Len << ", " << ExpectedValue << ", " << NumToSkip
                   << "): FieldValue = " << FieldValue << ", ExpectedValue = "
                   << ExpectedValue << ": "
                   << ((ExpectedValue == FieldValue) ? "PASS\n" : "FAIL\n"));
      break;
    }
    case MCD::OPC_CheckPredicate: {
      unsigned Len;
      // Decode the Predicate Index value.
      unsigned PIdx = decodeULEB128(++Ptr, &Len);
      Ptr += Len;
      // NumToSkip is a plain 24-bit integer.
      unsigned NumToSkip = *Ptr++;
      NumToSkip |= (*Ptr++) << 8;
      NumToSkip |= (*Ptr++) << 16;
      // Check the predicate.
      bool Pred;
      if (!(Pred = checkDecoderPredicate(PIdx, Bits)))
        Ptr += NumToSkip;
      (void)Pred;
      LLVM_DEBUG(dbgs() << Loc << ": OPC_CheckPredicate(" << PIdx << "): "
            << (Pred ? "PASS\n" : "FAIL\n"));

      break;
    }
    case MCD::OPC_Decode: {
      unsigned Len;
      // Decode the Opcode value.
      unsigned Opc = decodeULEB128(++Ptr, &Len);
      Ptr += Len;
      unsigned DecodeIdx = decodeULEB128(Ptr, &Len);
      Ptr += Len;

      MI.clear();
      MI.setOpcode(Opc);
      bool DecodeComplete;
      S = decodeToMCInst(S, DecodeIdx, insn, MI, Address, DisAsm, DecodeComplete);
      assert(DecodeComplete);

      LLVM_DEBUG(dbgs() << Loc << ": OPC_Decode: opcode " << Opc
                   << ", using decoder " << DecodeIdx << ": "
                   << (S != MCDisassembler::Fail ? "PASS" : "FAIL") << "\n");
      return S;
    }
    case MCD::OPC_TryDecode: {
      unsigned Len;
      // Decode the Opcode value.
      unsigned Opc = decodeULEB128(++Ptr, &Len);
      Ptr += Len;
      unsigned DecodeIdx = decodeULEB128(Ptr, &Len);
      Ptr += Len;
      // NumToSkip is a plain 24-bit integer.
      unsigned NumToSkip = *Ptr++;
      NumToSkip |= (*Ptr++) << 8;
      NumToSkip |= (*Ptr++) << 16;

      // Perform the decode operation.
      MCInst TmpMI;
      TmpMI.setOpcode(Opc);
      bool DecodeComplete;
      S = decodeToMCInst(S, DecodeIdx, insn, TmpMI, Address, DisAsm, DecodeComplete);
      LLVM_DEBUG(dbgs() << Loc << ": OPC_TryDecode: opcode " << Opc
                   << ", using decoder " << DecodeIdx << ": ");

      if (DecodeComplete) {
        // Decoding complete.
        LLVM_DEBUG(dbgs() << (S != MCDisassembler::Fail ? "PASS" : "FAIL") << "\n");
        MI = TmpMI;
        return S;
      } else {
        assert(S == MCDisassembler::Fail);
        // If the decoding was incomplete, skip.
        Ptr += NumToSkip;
        LLVM_DEBUG(dbgs() << "FAIL: continuing at " << (Ptr - DecodeTable) << "\n");
        // Reset decode status. This also drops a SoftFail status that could be
        // set before the decode attempt.
        S = MCDisassembler::Success;
      }
      break;
    }
    case MCD::OPC_SoftFail: {
      // Decode the mask values.
      unsigned Len;
      InsnType PositiveMask = decodeULEB128(++Ptr, &Len);
      Ptr += Len;
      InsnType NegativeMask = decodeULEB128(Ptr, &Len);
      Ptr += Len;
      bool Fail = (insn & PositiveMask) || (~insn & NegativeMask);
      if (Fail)
        S = MCDisassembler::SoftFail;
      LLVM_DEBUG(dbgs() << Loc << ": OPC_SoftFail: " << (Fail ? "FAIL\n" : "PASS\n"));
      break;
    }
    case MCD::OPC_Fail: {
      LLVM_DEBUG(dbgs() << Loc << ": OPC_Fail\n");
      return MCDisassembler::Fail;
    }
    }
  }
  llvm_unreachable("bogosity detected in disassembler state machine!");
}


} // end namespace llvm
