; Positive test for inline register constraints
;
; RUN: llc -march=cpu0 -no-integrated-as < %s | FileCheck %s

define i32 @main() nounwind {
entry:

; r with char
;CHECK:	#APP
;CHECK:	addiu ${{[0-9]+|t9}},${{[0-9]+|t9}},23
;CHECK:	#NO_APP
  tail call i8 asm sideeffect "addiu $0,$1,$2", "=r,r,n"(i8 27, i8 23) nounwind

; r with short
;CHECK:	#APP
;CHECK:	addiu ${{[0-9]+|t9}},${{[0-9]+|t9}},13
;CHECK:	#NO_APP
  tail call i16 asm sideeffect "addiu $0,$1,$2", "=r,r,n"(i16 17, i16 13) nounwind

; r with int
;CHECK:	#APP
;CHECK:	addiu ${{[0-9]+|t9}},${{[0-9]+|t9}},3
;CHECK:	#NO_APP
  tail call i32 asm sideeffect "addiu $0,$1,$2", "=r,r,n"(i32 7, i32 3) nounwind

; Now c with 1024: make sure register $25 is picked
; CHECK: #APP
; CHECK: addiu $t9,$t9,1024
; CHECK: #NO_APP	
   tail call i32 asm sideeffect "addiu $0,$1,$2", "=c,c,I"(i32 4194304, i32 1024) nounwind

  ret i32 0
}
