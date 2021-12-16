; RUN: llc -march=cpu0 -relocation-model=pic -no-integrated-as < %s | FileCheck %s

; Simple memory
@g1 = external global i32

define i32 @f1(i32 %x) nounwind {
entry:
; CHECK: addiu $[[T0:[0-9]+|t9]], $sp
; CHECK: #APP
; CHECK: st ${{[0-9]+|t9}}, 0($[[T0]])
; CHECK: #NO_APP
; CHECK: #APP
; CHECK: ld $[[T3:[0-9]+|t9]], 0($[[T0]])
; CHECK: #NO_APP
; CHECK: ld  $[[T1:[0-9]+|t9]], %got_lo(g1)
; CHECK: st  $[[T3]], 0($[[T1]])

  %l1 = alloca i32, align 4
  call void asm "st $1, $0", "=*m,r"(i32* %l1, i32 %x) nounwind
  %0 = call i32 asm "ld $0, $1", "=r,*m"(i32* %l1) nounwind
  store i32 %0, i32* @g1, align 4
  ret i32 %0
}

; CHECK: #APP
; CHECK-NEXT: ld ${{[0-9]+|t9}},0(${{[0-9]+|t9}});
; CHECK-NEXT: #NO_APP

;int b[8] = {0,1,2,3,4,5,6,7};
;int main()
;{
;  int i;
; 
;  // The first word. Notice, no 'D'
;  { asm (
;    "lw    %0,%1;\n"
;    : "=r" (i) : "m" (*(b+4)));}
; 
;  // The second word
;  { asm (
;    "lw    %0,%D1;\n"
;    : "=r" (i) "m" (*(b+4)));}
;}

@b = common global [20 x i32] zeroinitializer, align 4

define void @main() {
entry:
  tail call void asm sideeffect "    ld    $0,${1};", "r,*m,~{$11}"(i32 undef, i32* getelementptr inbounds ([20 x i32], [20 x i32]* @b, i32 0, i32 3))
  ret void
}

attributes #0 = { nounwind }

