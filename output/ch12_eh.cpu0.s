  .text
  .section .mdebug.abiO32
  .previous
  .file  "ch12_eh.bc"
  .globl  _Z15throw_exceptionii
  .align  2
  .type  _Z15throw_exceptionii,@function
  .ent  _Z15throw_exceptionii   # @_Z15throw_exceptionii
_Z15throw_exceptionii:
  .cfi_startproc
  .frame  $fp,40,$lr
  .mask   0x00005000,-4
  .set  noreorder
  .set  nomacro
# BB#0:
  addiu  $sp, $sp, -40
$tmp0:
  .cfi_def_cfa_offset 40
  st  $lr, 36($sp)            # 4-byte Folded Spill
  st  $fp, 32($sp)            # 4-byte Folded Spill
$tmp1:
  .cfi_offset 14, -4
$tmp2:
  .cfi_offset 12, -8
  move   $fp, $sp
$tmp3:
  .cfi_def_cfa_register 12
  st  $4, 28($fp)
  st  $5, 24($fp)
  ld  $2, 28($fp)
  cmp  $sw, $2, $5
  jle  $sw, LBB0_2
  nop
  jmp  LBB0_1
  nop
LBB0_2:
  move   $sp, $fp
  ld  $fp, 32($sp)            # 4-byte Folded Reload
  ld  $lr, 36($sp)            # 4-byte Folded Reload
  addiu  $sp, $sp, 40
  ret  $lr
  nop
LBB0_1:
  addiu  $4, $zero, 1
  jsub  __cxa_allocate_exception
  nop
  addiu  $3, $zero, 0
  st  $3, 8($sp)
  lui  $3, %hi(_ZTI3Ex1)
  ori  $5, $3, %lo(_ZTI3Ex1)
  addu  $4, $zero, $2
  jsub  __cxa_throw
  nop
  .set  macro
  .set  reorder
  .end  _Z15throw_exceptionii
$func_end0:
  .size  _Z15throw_exceptionii, ($func_end0)-_Z15throw_exceptionii
  .cfi_endproc

  .globl  _Z14test_try_catchv
  .align  2
  .type  _Z14test_try_catchv,@function
  .ent  _Z14test_try_catchv     # @_Z14test_try_catchv
_Z14test_try_catchv:
$tmp7:
$func_begin0 = ($tmp7)
  .cfi_startproc
  .cfi_personality 0, __gxx_personality_v0
  .cfi_lsda 0, $exception0
  .frame  $fp,32,$lr
  .mask   0x00005200,-4
  .set  noreorder
  .set  nomacro
# BB#0:
  addiu  $sp, $sp, -32
$tmp8:
  .cfi_def_cfa_offset 32
  st  $lr, 28($sp)            # 4-byte Folded Spill
  st  $fp, 24($sp)            # 4-byte Folded Spill
  st  $9, 20($sp)             # 4-byte Folded Spill
$tmp9:
  .cfi_offset 14, -4
$tmp10:
  .cfi_offset 12, -8
$tmp11:
  .cfi_offset 9, -12
  move   $fp, $sp
$tmp12:
  .cfi_def_cfa_register 12
$tmp4:
  addiu  $4, $zero, 2
  addiu  $9, $zero, 1
  addu  $5, $zero, $9
  jsub  _Z15throw_exceptionii
  nop
$tmp5:
# BB#2:
  addiu  $2, $zero, 0
  st  $2, 16($fp)
LBB1_3:
  ld  $2, 16($fp)
  move   $sp, $fp
  ld  $9, 20($sp)             # 4-byte Folded Reload
  ld  $fp, 24($sp)            # 4-byte Folded Reload
  ld  $lr, 28($sp)            # 4-byte Folded Reload
  addiu  $sp, $sp, 32
  ret  $lr
  nop
LBB1_1:
$tmp6:
  st  $4, 12($fp)
  st  $5, 8($fp)
  ld  $4, 12($fp)
  jsub  __cxa_begin_catch
  nop
  st  $9, 16($fp)
  jsub  __cxa_end_catch
  nop
  jmp  LBB1_3
  nop
  .set  macro
  .set  reorder
  .end  _Z14test_try_catchv
$func_end1:
  .size  _Z14test_try_catchv, ($func_end1)-_Z14test_try_catchv
  .cfi_endproc
  .section  .gcc_except_table,"a",@progbits
  .align  2
GCC_except_table1:
$exception0:
  .byte  255                     # @LPStart Encoding = omit
  .byte  0                       # @TType Encoding = absptr
  .asciz  "\242\200\200"          # @TType base offset
  .byte  3                       # Call site Encoding = udata4
  .byte  26                      # Call site table length
  .4byte  ($tmp4)-($func_begin0)  # >> Call Site 1 <<
  .4byte  ($tmp5)-($tmp4)         #   Call between $tmp4 and $tmp5
  .4byte  ($tmp6)-($func_begin0)  #     jumps to $tmp6
  .byte  1                       #   On action: 1
  .4byte  ($tmp5)-($func_begin0)  # >> Call Site 2 <<
  .4byte  ($func_end1)-($tmp5)    #   Call between $tmp5 and $func_end1
  .4byte  0                       #     has no landing pad
  .byte  0                       #   On action: cleanup
  .byte  1                       # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
  .byte  0                       #   No further actions
                                        # >> Catch TypeInfos <<
  .4byte  0                       # TypeInfo 1
  .align  2

  .type  _ZTS3Ex1,@object        # @_ZTS3Ex1
  .section  .rodata._ZTS3Ex1,"aG",@progbits,_ZTS3Ex1,comdat
  .weak  _ZTS3Ex1
  .align  2
_ZTS3Ex1:
  .asciz  "3Ex1"
  .size  _ZTS3Ex1, 5

  .type  _ZTI3Ex1,@object        # @_ZTI3Ex1
  .section  .rodata._ZTI3Ex1,"aG",@progbits,_ZTI3Ex1,comdat
  .weak  _ZTI3Ex1
  .align  3
_ZTI3Ex1:
  .4byte  _ZTVN10__cxxabiv117__class_type_infoE+8
  .4byte  _ZTS3Ex1
  .size  _ZTI3Ex1, 8


