; RUN: llc -march=cpu0 -no-integrated-as < %s | FileCheck %s

define i32 @main() nounwind {
entry:

; First I with short
; CHECK: #APP
; CHECK: addiu ${{[0-9]+|t9}},${{[0-9]+|t9}},4096
; CHECK: #NO_APP
  tail call i16 asm sideeffect "addiu $0,$1,$2", "=r,r,I"(i16 7, i16 4096) nounwind

; Then I with int
; CHECK: #APP
; CHECK: addiu ${{[0-9]+|t9}},${{[0-9]+|t9}},-3
; CHECK: #NO_APP
   tail call i32 asm sideeffect "addiu $0,$1,$2", "=r,r,I"(i32 7, i32 -3) nounwind

; Now J with 0
; CHECK: #APP
; CHECK: addiu ${{[0-9]+|t9}},${{[0-9]+|t9}},0
; CHECK: #NO_APP
  tail call i32 asm sideeffect "addiu $0,$1,$2\0A\09 ", "=r,r,J"(i32 7, i16 0) nounwind

; Now K with 64
; CHECK: #APP
; CHECK: addiu ${{[0-9]+|t9}},${{[0-9]+|t9}},64
; CHECK: #NO_APP	
  tail call i16 asm sideeffect "addiu $0,$1,$2\0A\09 ", "=r,r,K"(i16 7, i16 64) nounwind

; Now L with 0x00100000
; CHECK: #APP
; CHECK: ori ${{[0-9]+|t9}},${{[0-9]+|t9}},1048576
; CHECK: #NO_APP	
  tail call i32 asm sideeffect "ori $0,$1,$2\0A\09", "=r,r,L"(i32 7, i32 1048576) nounwind

; Now N with -3
; CHECK: #APP
; CHECK: addiu ${{[0-9]+|t9}},${{[0-9]+|t9}},-3
; CHECK: #NO_APP	
  tail call i32 asm sideeffect "addiu $0,$1,$2", "=r,r,N"(i32 7, i32 -3) nounwind

; Now O with -3
; CHECK: #APP
; CHECK: addiu ${{[0-9]+|t9}},${{[0-9]+|t9}},-3
; CHECK: #NO_APP	
  tail call i32 asm sideeffect "addiu $0,$1,$2", "=r,r,O"(i32 7, i16 -3) nounwind

; Now P with 65535
; CHECK: #APP
; CHECK: addiu ${{[0-9]+|t9}},${{[0-9]+|t9}},65535
; CHECK: #NO_APP	
  tail call i32 asm sideeffect "addiu $0,$1,$2", "=r,r,P"(i32 7, i32 65535) nounwind

  ret i32 0
}
