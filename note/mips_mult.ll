; ~/llvm/3.9.0/release/cmake_debug_build/bin/llc -debug -print-after-all -march=mipsel -mcpu=mips32 -mattr=+dsp -verify-machineinstrs 1.ll -o -

define i64 @test__builtin_mips_mult1(i32 %i0, i32 %a0, i32 %a1) nounwind readnone {
entry:
; CHECK: mult $ac{{[0-9]}}

  %0 = tail call i64 @llvm.mips.mult(i32 %a0, i32 %a1)
  ret i64 %0
}

declare i64 @llvm.mips.mult(i32, i32) nounwind readnone
