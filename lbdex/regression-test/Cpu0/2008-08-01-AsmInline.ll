; RUN: llc -march=cpu0 < %s | FileCheck %s

%struct.DWstruct = type { i32, i32 }

define i32 @A0(i32 %u, i32 %v) nounwind  {
entry:
; CHECK: multu 
; CHECK: mflo
; CHECK: mfhi
  %asmtmp = tail call %struct.DWstruct asm "multu $2,$3", "={lo},={hi},r,r"( i32 %u, i32 %v ) nounwind
  %asmresult = extractvalue %struct.DWstruct %asmtmp, 0
  %asmresult1 = extractvalue %struct.DWstruct %asmtmp, 1    ; <i32> [#uses=1]
  %res = add i32 %asmresult, %asmresult1
  ret i32 %res
}

@gi2 = external global i32
@gi1 = external global i32
@gi0 = external global i32

define void @foo0() nounwind {
entry:
; CHECK: addu
  %0 = load i32, i32* @gi1, align 4
  %1 = load i32, i32* @gi0, align 4
  %2 = tail call i32 asm "addu $0, $1, $2", "=r,r,r"(i32 %0, i32 %1) nounwind
  store i32 %2, i32* @gi2, align 4
  ret void
}

