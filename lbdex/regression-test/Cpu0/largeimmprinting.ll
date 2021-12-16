; RUN: llc -march=cpu0 -relocation-model=static -filetype=asm < %s | FileCheck %s -check-prefix=STATIC
; RUN: llc -march=cpu0 -relocation-model=pic -filetype=asm < %s | FileCheck %s -check-prefix=PIC

%struct.S1 = type { [65536 x i8] }

@s1 = external global %struct.S1

define void @f() nounwind {
entry:
; STATIC:  lui $[[R0:[0-9]+|t9]], 65535
; STATIC:  addiu $[[R0]], $[[R0]], -{{[0-9]+|t9}}
; STATIC:  addu $sp, $sp, $[[R0]]
; STATIC:  lui $[[R1:[0-9]+|t9]], %hi(s1)
; STATIC:  ori $[[R2:[0-9]+|t9]], $[[R1]], %lo(s1)

; PIC:  lui $[[R0:[0-9]+|t9]], 65535
; PIC:  addiu $[[R0]], $[[R0]], -{{[0-9]+|t9}}
; PIC:  addu $sp, $sp, $[[R0]]
; PIC:  lui $[[R1:[0-9]+|t9]], %got_hi(s1)
; PIC:  addu	$[[R2:[0-9]+|t9]], $[[R1]], $gp
; PIC:  ld $[[R3:[0-9]+|t9]], %got_lo(s1)($[[R2]])

  %agg.tmp = alloca %struct.S1, align 1
  %tmp = getelementptr inbounds %struct.S1, %struct.S1* %agg.tmp, i32 0, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %tmp, i8* getelementptr inbounds (%struct.S1, %struct.S1* @s1, i32 0, i32 0, i32 0), i32 65536, i32 1, i1 false)
  call void @f2(%struct.S1* byval(%struct.S1) %agg.tmp) nounwind
  ret void
}

declare void @f2(%struct.S1* byval(%struct.S1))

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
