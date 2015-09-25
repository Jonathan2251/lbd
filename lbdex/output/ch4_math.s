	.text
	.section .mdebug.abi32
	.previous
	.file	"ch4_math.ll"
	.globl	_Z9test_mathv
	.align	2
	.type	_Z9test_mathv,@function
	.ent	_Z9test_mathv           # @_Z9test_mathv
_Z9test_mathv:
	.cfi_startproc
	.frame	$sp,8,$lr
	.mask 	0x00000000,0
	.set	noreorder
	.set	nomacro
# BB#0:
	addiu	$sp, $sp, -8
$tmp0:
	.cfi_def_cfa_offset 8
	ld	$2, 0($sp)
	ld	$3, 4($sp)
	subu	$4, $3, $2
	addu	$5, $3, $2
	addu	$4, $5, $4
	mul	$5, $3, $2
	addu	$4, $4, $5
	shl	$5, $3, 2
	addu	$4, $4, $5
	sra	$5, $3, 2
	addu	$4, $4, $5
	addiu	$5, $zero, 128
	shrv	$5, $5, $2
	addiu	$t9, $zero, 1
	shlv	$t9, $t9, $2
	srav	$2, $3, $2
	shr	$3, $3, 30
	addu	$3, $4, $3
	addu	$3, $3, $t9
	addu	$3, $3, $5
	addu	$2, $3, $2
	addiu	$sp, $sp, 8
	ret	$lr
	.set	macro
	.set	reorder
	.end	_Z9test_mathv
$tmp1:
	.size	_Z9test_mathv, ($tmp1)-_Z9test_mathv
	.cfi_endproc


