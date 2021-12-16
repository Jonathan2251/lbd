; RUN: llc  < %s -march=cpu0 -mcpu=cpu032II -no-integrated-as | FileCheck %s

; CHECK: ld	$2, 8($sp)
; CHECK: st	$0, 4($sp)
; CHECK: addiu $3,	$ZERO, 0
; CHECK: slti $4, $3, 1
; CHECK: sltiu $4, $3, 0
; CHECK: add $3, $1, $2
; CHECK: sub $3, $2, $3
; CHECK: mul $2, $1, $3
; CHECK: div $3, $2
; CHECK: divu $2, $3
; CHECK: and $2, $1, $3
; CHECK: or $3, $1, $2
; CHECK: xor $1, $2, $3
; CHECK: mult $4, $3
; CHECK: multu $3, $2
; CHECK: mfhi $3
; CHECK: mflo $2
; CHECK: mthi $2
; CHECK: mtlo $2
; CHECK: sra $2, $2, 2
; CHECK: rol $2, $1, 3
; CHECK: ror $3, $3, 4
; CHECK: shl $2, $2, 2
; CHECK: shr $2, $3, 5
; CHECK: beq $2, $4, 20
; CHECK: slt $4, $2, $3
; CHECK: sltu $2, $3, $1
; CHECK: bne $2, $3, -16
; CHECK: swi 0x00000400
; CHECK: jsub 0x000010000
; CHECK: ret $lr
; CHECK: jalr $t9
; CHECK: li $3, 0x00700000
; CHECK: la $3, 0x00800000($6)
; CHECK: la $3, 0x00900000

  module asm "ld\09$2, 8($sp)"
  module asm "st\09$0, 4($sp)"
  module asm "addiu $3,\09$ZERO, 0"
  module asm "slti $4, $3, 1"
  module asm "sltiu $4, $3, 0"
  module asm "add $3, $1, $2"
  module asm "sub $3, $2, $3"
  module asm "mul $2, $1, $3"
  module asm "div $3, $2"
  module asm "divu $2, $3"
  module asm "and $2, $1, $3"
  module asm "or $3, $1, $2"
  module asm "xor $1, $2, $3"
  module asm "mult $4, $3"
  module asm "multu $3, $2"
  module asm "mfhi $3"
  module asm "mflo $2"
  module asm "mthi $2"
  module asm "mtlo $2"
  module asm "sra $2, $2, 2"
  module asm "rol $2, $1, 3"
  module asm "ror $3, $3, 4"
  module asm "shl $2, $2, 2"
  module asm "shr $2, $3, 5"
  module asm "beq $2, $4, 20"
  module asm "slt $4, $2, $3"
  module asm "sltu $2, $3, $1"
  module asm "bne $2, $3, -16"
  module asm "swi 0x00000400"
  module asm "jsub 0x000010000"
  module asm "ret $lr"
  module asm "jalr $t9"
  module asm "li $3, 0x00700000"
  module asm "la $3, 0x00800000($6)"
  module asm "la $3, 0x00900000"

