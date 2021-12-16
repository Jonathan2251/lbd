; Positive test for inline register constraints
;
; RUN: llc -march=cpu0el -no-integrated-as < %s  | FileCheck -check-prefix=CHECK_LITTLE_32 %s

%union.u_tag = type { i64 }
%struct.anon = type { i32, i32 }
@uval = common global %union.u_tag zeroinitializer, align 8

; X with -3
define i32 @constraint_X() nounwind {
entry:
;CHECK_LITTLE_32-LABEL:   constraint_X:
;CHECK_LITTLE_32: #APP
;CHECK_LITTLE_32: addiu ${{[0-9]+|t9}},${{[0-9]+|t9}},0xfffffffffffffffd
;CHECK_LITTLE_32: #NO_APP
  tail call i32 asm sideeffect "addiu $0,$1,${2:X}", "=r,r,I"(i32 7, i32 -3) ;
  ret i32 0
}

; x with -3
define i32 @constraint_x() nounwind {
entry:
;CHECK_LITTLE_32-LABEL:   constraint_x:
;CHECK_LITTLE_32: #APP
;CHECK_LITTLE_32: addiu ${{[0-9]+|t9}},${{[0-9]+|t9}},0xfffd
;CHECK_LITTLE_32: #NO_APP
  tail call i32 asm sideeffect "addiu $0,$1,${2:x}", "=r,r,I"(i32 7, i32 -3) ;
  ret i32 0
}

; d with -3
define i32 @constraint_d() nounwind {
entry:
;CHECK_LITTLE_32-LABEL:   constraint_d:
;CHECK_LITTLE_32:   #APP
;CHECK_LITTLE_32:   addiu ${{[0-9]+|t9}},${{[0-9]+|t9}},-3
;CHECK_LITTLE_32:   #NO_APP
  tail call i32 asm sideeffect "addiu $0,$1,${2:d}", "=r,r,I"(i32 7, i32 -3) ;
  ret i32 0
}

; m with -3
define i32 @constraint_m() nounwind {
entry:
;CHECK_LITTLE_32-LABEL:   constraint_m:
;CHECK_LITTLE_32:   #APP
;CHECK_LITTLE_32:   addiu ${{[0-9]+|t9}},${{[0-9]+|t9}},-4
;CHECK_LITTLE_32:   #NO_APP
  tail call i32 asm sideeffect "addiu $0,$1,${2:m}", "=r,r,I"(i32 7, i32 -3) ;
  ret i32 0
}

; z with -3
define i32 @constraint_z() nounwind {
entry:
;CHECK_LITTLE_32-LABEL: constraint_z:
;CHECK_LITTLE_32:    #APP
;CHECK_LITTLE_32:    addiu ${{[0-9]+|t9}},${{[0-9]+|t9}},-3
;CHECK_LITTLE_32:    #NO_APP
  tail call i32 asm sideeffect "addiu $0,$1,${2:z}", "=r,r,I"(i32 7, i32 -3) ;

; z with 0
;CHECK_LITTLE_32:    #APP
;CHECK_LITTLE_32:    addiu ${{[0-9]+|t9}},${{[0-9]+|t9}},$0
;CHECK_LITTLE_32:    #NO_APP
  tail call i32 asm sideeffect "addiu $0,$1,${2:z}", "=r,r,I"(i32 7, i32 0) nounwind
  ret i32 0
}

