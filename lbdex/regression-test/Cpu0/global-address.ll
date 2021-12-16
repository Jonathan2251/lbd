; RUN: llc -march=cpu0el -relocation-model=pic -cpu0-use-small-section=false < %s | FileCheck %s -check-prefix=PIC-0
; RUN: llc -march=cpu0el -relocation-model=pic -cpu0-use-small-section=true < %s | FileCheck %s -check-prefix=PIC-1
; RUN: llc -march=cpu0el -relocation-model=static  -cpu0-use-small-section=false < %s | FileCheck %s -check-prefix=STATIC-0
; RUN: llc -march=cpu0el -relocation-model=static  -cpu0-use-small-section=true < %s | FileCheck %s -check-prefix=STATIC-1

@s1 = internal unnamed_addr global i32 8, align 4
@g1 = external global i32
@gV = global i32 100, align 4

define void @foo() nounwind {
entry:
; PIC-0: ld  $[[R0:[0-9]+|t9]], %got(s1)($gp)
; PIC-0: ori  $[[R1:[0-9]+|t9]], $[[R0]], %lo(s1)
; PIC-0: ld  ${{[0-9]+|t9}}, 0($[[R1]])
; PIC-0: ld  $gp
; PIC-0: lui $[[R0:[0-9]+|t9]], %got_hi(g1)
; PIC-0: addu  $[[R1:[0-9]+|t9]], $[[R0]], $gp
; PIC-0: ld  $[[R2:[0-9]+|t9]], %got_lo(g1)($[[R1]])
; PIC-0: ld  ${{[0-9]+|t9}}, 0($[[R2]])
; PIC-0: lui $[[R3:[0-9]+|t9]], %got_hi(gV)
; PIC-0: addu  $[[R4:[0-9]+|t9]], $[[R3]], $gp
; PIC-0: ld  $[[R5:[0-9]+|t9]], %got_lo(gV)($[[R4]])
; PIC-0: ld  ${{[0-9]+|t9}}, 0($[[R5]])
; PIC-1: ld  $[[R0:[0-9]+|t9]], %got(s1)
; PIC-1: ori  $[[R1:[0-9]+|t9]], $[[R0]], %lo(s1)
; PIC-1: ld  ${{[0-9]+|t9}}, 0($[[R1]])
; PIC-1: ld  $gp
; PIC-1: ld $[[R0:[0-9]+|t9]], %got(g1)
; PIC-1: ld  ${{[0-9]+|t9}}, 0($[[R0]])
; PIC-1: ld $[[R0:[0-9]+|t9]], %got(gV)($gp)
; PIC-1: ld  ${{[0-9]+|t9}}, 0($[[R0]])
; STATIC-0: lui $[[R0:[0-9]+|t9]], %hi(s1)
; STATIC-0: ori  $[[R1:[0-9]+|t9]], $[[R0]], %lo(s1)
; STATIC-0: ld  ${{[0-9]+|t9}}, 0($[[R1]])
; STATIC-0: lui $[[R2:[0-9]+|t9]], %hi(g1)
; STATIC-0: ori  $[[R3:[0-9]+|t9]], $[[R2]], %lo(g1)
; STATIC-0: ld  ${{[0-9]+|t9}}, 0($[[R3]])
; STATIC-1: ori  $[[R0:[0-9]+|t9]], $gp, %gp_rel(s1)
; STATIC-1: ld  ${{[0-9]+|t9}}, 0($[[R0]])
; STATIC-1: ori  $[[R1:[0-9]+|t9]], $gp, %gp_rel(g1)
; STATIC-1: ld  ${{[0-9]+|t9}}, 0($[[R1]])
; STATIC-1: ori  $[[R0:[0-9]+|t9]], $gp, %gp_rel(gV)
; STATIC-1: ld  ${{[0-9]+|t9}}, 0($[[R0]])

  %0 = load i32, i32* @s1, align 4
  tail call void @foo1(i32 %0) nounwind
  %1 = load i32, i32* @g1, align 4
  store i32 %1, i32* @s1, align 4
  %add = add nsw i32 %1, 2
  store i32 %add, i32* @g1, align 4
  %c = alloca i32, align 4
  %2 = load i32, i32* @gV, align 4
  store i32 %2, i32* %c, align 4
  ret void
}

declare void @foo1(i32)

