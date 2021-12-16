; RUN: llc -march=cpu0el -mcpu=cpu032II -relocation-model=pic < %s | FileCheck %s

%0 = type { i8, i16, i32, i64, double, i32, [4 x i8] }
%struct.S1 = type { i8, i16, i32, i64, double, i32 }
%struct.S2 = type { [4 x i32] }
%struct.S3 = type { i8 }

@f1.s1 = internal unnamed_addr constant %0 { i8 1, i16 2, i32 3, i64 4, double 5.000000e+00, i32 6, [4 x i8] undef }, align 8
@f1.s2 = internal unnamed_addr constant %struct.S2 { [4 x i32] [i32 7, i32 8, i32 9, i32 10] }, align 4

define void @f1() nounwind {
entry:
; CHECK: ld  $[[R1:[0-9]+|t9]], %got(f1.s1)
; CHECK: ori $[[R0:[0-9]+|t9]], $[[R1]], %lo(f1.s1)
; CHECK: ld  $[[R2:[0-9]+|t9]], 28($[[R0]])
; CHECK: st  $[[R2]], 36($sp)
; CHECK: ld  $[[R2:[0-9]+|t9]], 24($[[R0]])
; CHECK: st  $[[R2]], 32($sp)
; CHECK: ld  $[[R2:[0-9]+|t9]], 20($[[R0]])
; CHECK: st  $[[R2]], 28($sp)
; CHECK: ld  $[[R2:[0-9]+|t9]], 16($[[R0]])
; CHECK: st  $[[R2]], 24($sp)
; CHECK: ld  $[[R2:[0-9]+|t9]], 12($[[R0]])
; CHECK: st  $[[R2]], 20($sp)
; CHECK: ld  $[[R2:[0-9]+|t9]], 8($[[R0]])
; CHECK: st  $[[R2]], 16($sp)
; CHECK: ld  $[[R2:[0-9]+|t9]], 4($[[R0]])
; CHECK: st  $[[R2]], 12($sp)
; CHECK: ld  $[[R2:[0-9]+|t9]], 0($[[R0]])
; CHECK: st  $[[R2]], 8($sp)
; CHECK: ld  $t9, %call16(callee1)($gp)
; CHECK: jalr $t9
  %agg.tmp10 = alloca %struct.S3, align 4
  call void @callee1(float 2.000000e+01, %struct.S1* byval(%struct.S1) bitcast (%0* @f1.s1 to %struct.S1*)) nounwind
  call void @callee2(%struct.S2* byval(%struct.S2) @f1.s2) nounwind
  %tmp11 = getelementptr inbounds %struct.S3, %struct.S3* %agg.tmp10, i32 0, i32 0
  store i8 11, i8* %tmp11, align 4
  call void @callee3(float 2.100000e+01, %struct.S3* byval(%struct.S3) %agg.tmp10, %struct.S1* byval(%struct.S1) bitcast (%0* @f1.s1 to %struct.S1*)) nounwind
  ret void
}

declare void @callee1(float, %struct.S1* byval(%struct.S1))

declare void @callee2(%struct.S2* byval(%struct.S2))

declare void @callee3(float, %struct.S3* byval(%struct.S3), %struct.S1* byval(%struct.S1))

