; RUN: llc  < %s -march=cpu0el -relocation-model=pic | FileCheck %s

@caller.sf1 = internal unnamed_addr global void (...)* null, align 4
@gf1 = external global void (...)*
@.str = private unnamed_addr constant [3 x i8] c"f2\00"

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind {
entry:
; CHECK: ld	$[[R1:[0-9]+|t9]], %got(f2)($gp)
; CHECK: ori $t9, $[[R1]], %lo(f2)
  tail call fastcc void @f2()
  ret i32 0
}

define void @caller(i32 %a0, i32 %a1) nounwind {
entry:
  %tobool = icmp eq i32 %a1, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:         
; CHECK: ld  $[[R1:[0-9]+|t9]], %got(caller.sf1)
; CHECK: ori  $[[R2:[0-9]+|t9]], $[[R1]], %lo(caller.sf1)
; CHECK: ld  $t9, {{[0-9]+|t9}}($[[R2]])
  %tmp1 = load void (...)*, void (...)** @caller.sf1, align 4
  tail call void (...) %tmp1() nounwind
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %tobool3 = icmp ne i32 %a0, 0
  %tmp4 = load void (...)*, void (...)** @gf1, align 4
;  %cond = select i1 %tobool3, void (...)* %tmp4, void (...)* bitcast (void ()* @sf2 to void (...)*)
;  store void (...)* %cond, void (...)** @caller.sf1, align 4
  ret void
}

define internal void @sf2() nounwind {
entry:
; CHECK: ld  $[[R2:[0-9]+|t9]], %got($.str)
; CHECK: ori ${{[0-9]+|t9}}, $[[R2]], %lo($.str)
; CHECK: ld	$t9, %call16(printf)($gp)
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0)) nounwind
  ret void
}

declare i32 @printf(i8* nocapture, ...) nounwind

define internal fastcc void @f2() nounwind noinline {
entry:
; CHECK: ld  $[[R2:[0-9]+|t9]], %got($.str)
; CHECK: ori ${{[0-9]+|t9}}, $[[R2]], %lo($.str)
; CHECK: ld	$t9, %call16(printf)($gp)
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0)) nounwind
  ret void
}

