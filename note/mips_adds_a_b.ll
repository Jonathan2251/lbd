; ~/llvm/3.9.0/release/cmake_debug_build/bin/llc -debug -print-after-all -march=mips -mattr=+msa,+fp64 -relocation-model=pic mips_adds_a_b.ll -o -

@llvm_mips_adds_a_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_adds_a_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_adds_a_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_adds_a_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_adds_a_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_adds_a_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.adds.a.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_adds_a_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.adds.a.b(<16 x i8>, <16 x i8>) nounwind

