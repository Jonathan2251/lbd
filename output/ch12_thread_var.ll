; ModuleID = 'ch12_thread_var.bc'
target datalayout = "E-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64"
target triple = "mips-unknown-linux-gnu"

@a = thread_local global i32 0, align 4
@b = thread_local global i32 0, align 4

; Function Attrs: nounwind
define i32 @_Z15test_thread_varv() #0 {
  store i32 2, i32* @a, align 4
  %1 = load i32, i32* @a, align 4
  ret i32 %1
}

define i32 @_Z17test_thread_var_2v() #1 {
  %1 = call i32* @_ZTW1b()
  store i32 3, i32* %1, align 4
  %2 = call i32* @_ZTW1b()
  %3 = load i32, i32* %2, align 4
  ret i32 %3
}

define weak_odr hidden i32* @_ZTW1b() {
  ret i32* @b
}

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="fa
lse" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp
-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "targ
et-cpu"="mips32r2" "target-features"="+mips32r2" "unsafe-fp-math"="false" "use-s
oft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-
frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="f
alse" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="
mips32r2" "target-features"="+mips32r2" "unsafe-fp-math"="false" "use-soft-float
"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.7.0 (tags/RELEASE_370/final)"}
