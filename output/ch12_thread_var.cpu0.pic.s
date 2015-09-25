  .text
  .section .mdebug.abiO32
  .previous
  .file  "ch12_thread_var.bc"
  .globl  _Z15test_thread_varv
  .align  2
  .type  _Z15test_thread_varv,@function
  .ent  _Z15test_thread_varv    # @_Z15test_thread_varv
_Z15test_thread_varv:
  .frame  $fp,16,$lr
  .mask   0x00005000,-4
  .set  noreorder
  .cpload  $t9
  .set  nomacro
# BB#0:
  addiu  $sp, $sp, -16
  st  $lr, 12($sp)            # 4-byte Folded Spill
  st  $fp, 8($sp)             # 4-byte Folded Spill
  move   $fp, $sp
  .cprestore  8
  ld  $t9, %call16(__tls_get_addr)($gp)
  ori  $4, $gp, %tlsgd(a)
  jalr  $t9
  nop
  ld  $gp, 8($fp)
  addiu  $3, $zero, 2
  st  $3, 0($2)
  addu  $2, $zero, $3
  move   $sp, $fp
  ld  $fp, 8($sp)             # 4-byte Folded Reload
  ld  $lr, 12($sp)            # 4-byte Folded Reload
  addiu  $sp, $sp, 16
  ret  $lr
  nop
  .set  macro
  .set  reorder
  .end  _Z15test_thread_varv
$func_end0:
  .size  _Z15test_thread_varv, ($func_end0)-_Z15test_thread_varv

  .globl  _Z17test_thread_var_2v
  .align  2
  .type  _Z17test_thread_var_2v,@function
  .ent  _Z17test_thread_var_2v  # @_Z17test_thread_var_2v
_Z17test_thread_var_2v:
  .cfi_startproc
  .frame  $fp,16,$lr
  .mask   0x00005000,-4
  .set  noreorder
  .cpload  $t9
  .set  nomacro
# BB#0:
  addiu  $sp, $sp, -16
$tmp0:
  .cfi_def_cfa_offset 16
  st  $lr, 12($sp)            # 4-byte Folded Spill
  st  $fp, 8($sp)             # 4-byte Folded Spill
$tmp1:
  .cfi_offset 14, -4
$tmp2:
  .cfi_offset 12, -8
  move   $fp, $sp
$tmp3:
  .cfi_def_cfa_register 12
  .cprestore  8
  ld  $t9, %call16(_ZTW1b)($gp)
  jalr  $t9
  nop
  ld  $gp, 8($fp)
  addiu  $3, $zero, 3
  st  $3, 0($2)
  ld  $t9, %call16(_ZTW1b)($gp)
  jalr  $t9
  nop
  ld  $gp, 8($fp)
  ld  $2, 0($2)
  move   $sp, $fp
  ld  $fp, 8($sp)             # 4-byte Folded Reload
  ld  $lr, 12($sp)            # 4-byte Folded Reload
  addiu  $sp, $sp, 16
  ret  $lr
  nop
  .set  macro
  .set  reorder
  .end  _Z17test_thread_var_2v
$func_end1:
  .size  _Z17test_thread_var_2v, ($func_end1)-_Z17test_thread_var_2v
  .cfi_endproc

  .hidden  _ZTW1b
  .weak  _ZTW1b
  .align  2
  .type  _ZTW1b,@function
  .ent  _ZTW1b                  # @_ZTW1b
_ZTW1b:
  .cfi_startproc
  .frame  $sp,16,$lr
  .mask   0x00004000,-4
  .set  noreorder
  .cpload  $t9
  .set  nomacro
# BB#0:
  addiu  $sp, $sp, -16
$tmp4:
  .cfi_def_cfa_offset 16
  st  $lr, 12($sp)            # 4-byte Folded Spill
$tmp5:
  .cfi_offset 14, -4
  .cprestore  8
  ld  $t9, %call16(__tls_get_addr)($gp)
  ori  $4, $gp, %tlsgd(b)
  jalr  $t9
  nop
  ld  $gp, 8($sp)
  ld  $lr, 12($sp)            # 4-byte Folded Reload
  addiu  $sp, $sp, 16
  ret  $lr
  nop
  .set  macro
  .set  reorder
  .end  _ZTW1b
$func_end2:
  .size  _ZTW1b, ($func_end2)-_ZTW1b
  .cfi_endproc

  .type  a,@object               # @a
  .section  .tbss,"awT",@nobits
  .globl  a
  .align  2
a:
  .4byte  0                       # 0x0
  .size  a, 4

  .type  b,@object               # @b
  .globl  b
  .align  2
b:
  .4byte  0                       # 0x0
  .size  b, 4


