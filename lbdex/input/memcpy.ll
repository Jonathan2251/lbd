; ~/llvm/test/cmake_debug_build/bin/Debug/llc -march=cpu0 -relocation-model=static < memcpy.ll

; /// start
; ModuleID = 'memcpy.bc'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32-S64"
target triple = "mips-unknown-linux-gnu"

; Function Attrs: nounwind
define void @memcpy(i8* %dest, i8* %source, i32 %size) #0 {
entry:
  %dest.addr = alloca i8*, align 4
  %source.addr = alloca i8*, align 4
  %size.addr = alloca i32, align 4
  store i8* %dest, i8** %dest.addr, align 4
  store i8* %source, i8** %source.addr, align 4
  store i32 %size, i32* %size.addr, align 4
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
