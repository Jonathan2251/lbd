; ~/llvm/test/build/bin/llc -march=cpu0 -mcpu=cpu032II -relocation-model=static test_memcpy.ll -O3 -o -

; IntrWriteMem: then no optimized out for -O3 in this instruction.
; Reference Intrinsics.td for depscription and more Intr*Mem usage and examples.
; From llvm-project/llvm/test/CodeGen/ARM/memfunc.ll
define i32 @llvm_memset_test(i8* %dest, i8* %src) nounwind {
entry:
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 1, i32 500, i1 false);
  ret i32 0
}

define i32 @llvm_memcpy_test(i8* %dest, i8* %src) nounwind {
entry:
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 500, i1 false);
  ret i32 0
}

define i32 @llvm_cpu0_gcd_test2(i32 %a, i32 %b) nounwind {
entry:
  %res = tail call i32 @llvm.cpu0.gcd(i32 %a, i32 %b)
  ret i32 0
}

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i1) nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind
declare i32 @llvm.cpu0.gcd(i32, i32) nounwind
