.. _sec-elf:

ELF Support
===========

.. contents::
   :local:
   :depth: 4

Cpu0 backend generated the ELF format of obj. 
The ELF (Executable and Linkable Format) is a common standard file format for 
executables, object code, shared libraries and core dumps. 
First published in the System V Application Binary Interface specification, 
and later in the Tool Interface Standard, it was quickly accepted among 
different vendors of Unixsystems. 
In 1999 it was chosen as the standard binary file format for Unix and 
Unix-like systems on x86 by the x86open project. 
Please reference [#wiki-elf]_.

The binary encode of Cpu0 instruction set in obj has been checked in the 
previous chapters. 
But we didn't dig into the ELF file format like elf header and relocation 
record at that time. 
You will learn the llvm-objdump, llvm-readelf, ..., tools and understand the 
ELF file 
format itself through using these tools to analyze the cpu0 generated obj in 
this chapter. 

This chapter introduces the tool to readers since we think it is a valuable 
knowledge in this popular ELF format and the ELF binutils analysis tool. 
An LLVM compiler engineer has the responsibility to make sure that his backend 
generate a correct obj. 
With this tool, you can verify your generated ELF format.
 
The cpu0 author has published a “System Software” book which introduces the 
topics 
of assembler, linker, loader, compiler and OS in concept, and at same time 
demonstrates how to use binutils and gcc to analysis ELF through the example 
code in his book. 
It's a Chinese book of “System Software” in concept and practice. 
This book does the real analysis through binutils. 
The “System Software” [#beck]_ written by Beck is a famous book in concept  
telling readers what the compiler output about, what the linker output about, 
what the loader output about, and how they work together in concept. 
You can reference it to understand how the **“Relocation Record”** works if you 
need to refresh or learning this knowledge for this chapter.

[#lk-out]_, [#lk-obj]_, [#lk-elf]_ are the Chinese documents available from the 
cpu0 author on web site.


ELF format
-----------

ELF is a format used in both obj and executable file. 
So, there are two views in it as :numref:`elf-f1`.

.. _elf-f1:
.. figure:: ../Fig/elf/1.png
    :height: 320 px
    :width: 213 px
    :scale: 100 %
    :align: center

    ELF file format overview

As :numref:`elf-f1`, the “Section header table” include sections .text, 
.rodata, ..., .data which are sections layout for code, read only data, ..., 
and read/write data, respectively. 
“Program header table” include segments for run time code and data. 
The definition of segments is the run time layout for code and data while 
sections is the link time layout for code and data.

ELF header and Section header table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's run Chapter9_3/ with ch6_1.cpp, and dump ELF header information by 
``llvm-readelf -h`` to see what information the ELF header contains.

.. code-block:: console

  [Gamma@localhost input]$ ~/llvm/test/build/bin/llc -march=cpu0 
  -relocation-model=pic -filetype=obj ch6_1.bc -o ch6_1.cpu0.o
  
  [Gamma@localhost input]$ llvm-readelf -h ch6_1.cpu0.o 
    Magic:   7f 45 4c 46 01 02 01 03 00 00 00 00 00 00 00 00 
    Class:                             ELF32
    Data:                              2's complement, big endian
    Version:                           1 (current)
    OS/ABI:                            UNIX - GNU
    ABI Version:                       0
    Type:                              REL (Relocatable file)
    Machine:                           <unknown>: 0xc9
    Version:                           0x1
    Entry point address:               0x0
    Start of program headers:          0 (bytes into file)
    Start of section headers:          176 (bytes into file)
    Flags:                             0x0
    Size of this header:               52 (bytes)
    Size of program headers:           0 (bytes)
    Number of program headers:         0
    Size of section headers:           40 (bytes)
    Number of section headers:         8
    Section header string table index: 5
  [Gamma@localhost input]$ 

  [Gamma@localhost input]$ ~/llvm/test/build/bin/llc 
  -march=mips -relocation-model=pic -filetype=obj ch6_1.bc -o ch6_1.mips.o
  
  [Gamma@localhost input]$ llvm-readelf -h ch6_1.mips.o 
  ELF Header:
    Magic:   7f 45 4c 46 01 02 01 03 00 00 00 00 00 00 00 00 
    Class:                             ELF32
    Data:                              2's complement, big endian
    Version:                           1 (current)
    OS/ABI:                            UNIX - GNU
    ABI Version:                       0
    Type:                              REL (Relocatable file)
    Machine:                           MIPS R3000
    Version:                           0x1
    Entry point address:               0x0
    Start of program headers:          0 (bytes into file)
    Start of section headers:          200 (bytes into file)
    Flags:                             0x50001007, noreorder, pic, cpic, o32, mips32
    Size of this header:               52 (bytes)
    Size of program headers:           0 (bytes)
    Number of program headers:         0
    Size of section headers:           40 (bytes)
    Number of section headers:         9
    Section header string table index: 6
  [Gamma@localhost input]$ 


As above ELF header display, it contains information of magic number, version, 
ABI, ..., . The Machine field of cpu0 is unknown while mips is known as 
MIPSR3000. 
It is unknown because cpu0 is not a popular CPU recognized by utility llvm-readelf. 
Let's check ELF segments information as follows,

.. code-block:: console

  [Gamma@localhost input]$ llvm-readelf -l ch6_1.cpu0.o 
  
  There are no program headers in this file.
  [Gamma@localhost input]$ 


The result is in expectation because cpu0 obj is for link only, not for 
execution. 
So, the segments is empty. 
Check ELF sections information as follows. 
Every section contains offset and size information.

.. code-block:: console

  [Gamma@localhost input]$ llvm-readelf -S ch6_1.cpu0.o 
  There are 10 section headers, starting at offset 0xd4:
  
  Section Headers:
    [Nr] Name              Type            Addr     Off    Size   ES Flg Lk Inf Al
    [ 0]                   NULL            00000000 000000 000000 00      0   0  0
    [ 1] .text             PROGBITS        00000000 000034 000034 00  AX  0   0  4
    [ 2] .rel.text         REL             00000000 000310 000018 08      8   1  4
    [ 3] .data             PROGBITS        00000000 000068 000004 00  WA  0   0  4
    [ 4] .bss              NOBITS          00000000 00006c 000000 00  WA  0   0  4
    [ 5] .eh_frame         PROGBITS        00000000 00006c 000028 00   A  0   0  4
    [ 6] .rel.eh_frame     REL             00000000 000328 000008 08      8   5  4
    [ 7] .shstrtab         STRTAB          00000000 000094 00003e 00      0   0  1
    [ 8] .symtab           SYMTAB          00000000 000264 000090 10      9   6  4
    [ 9] .strtab           STRTAB          00000000 0002f4 00001b 00      0   0  1
  Key to Flags:
    W (write), A (alloc), X (execute), M (merge), S (strings)
    I (info), L (link order), G (group), T (TLS), E (exclude), x (unknown)
    O (extra OS processing required) o (OS specific), p (processor specific)
  [Gamma@localhost input]$ 



Relocation Record
~~~~~~~~~~~~~~~~~

Cpu0 backend translate global variable as follows,

.. code-block:: console

  [Gamma@localhost input]$ clang -target mips-unknown-linux-gnu -c ch6_1.cpp 
  -emit-llvm -o ch6_1.bc
  [Gamma@localhost input]$ ~/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch6_1.bc -o ch6_1.cpu0.s
  [Gamma@localhost input]$ cat ch6_1.cpu0.s 
    .section .mdebug.abi32
    .previous
    .file "ch6_1.bc"
    .text
    ...
    .cfi_startproc
    .frame  $sp,8,$lr
    .mask   0x00000000,0
    .set  noreorder
    .cpload $t9
    ...
    lui $2, %got_hi(gI)
    addu $2, $2, $gp
    ld $2, %got_lo(gI)($2)
    ...
    .type gI,@object              # @gI
    .data
    .globl  gI
    .align  2
  gI:
    .4byte  100                     # 0x64
    .size gI, 4
  
  
  [Gamma@localhost input]$ ~/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=pic -filetype=obj ch6_1.bc -o ch6_1.cpu0.o
  [Gamma@localhost input]$ llvm-objdump -s ch6_1.cpu0.o
  
  ch6_1.cpu0.o:     file format elf32-big
  
  Contents of section .text:
  // .cpload machine instruction
   0000 0fa00000 0daa0000 13aa6000 ........  ..............`.
   ...
   0020 002a0000 00220000 012d0000 0ddd0008  .*..."...-......
   ...
  [Gamma@localhost input]$ Jonathan$ 
  
  [Gamma@localhost input]$ llvm-readelf -tr ch6_1.cpu0.o 
  There are 8 section headers, starting at offset 0xb0:

  Section Headers:
    [Nr] Name
         Type            Addr     Off    Size   ES   Lk Inf Al
         Flags
    [ 0] 
         NULL            00000000 000000 000000 00   0   0  0
         [00000000]: 
    [ 1] .text
         PROGBITS        00000000 000034 000044 00   0   0  4
         [00000006]: ALLOC, EXEC
    [ 2] .rel.text
         REL             00000000 0002a8 000020 08   6   1  4
         [00000000]: 
    [ 3] .data
         PROGBITS        00000000 000078 000008 00   0   0  4
         [00000003]: WRITE, ALLOC
    [ 4] .bss
         NOBITS          00000000 000080 000000 00   0   0  4
         [00000003]: WRITE, ALLOC
    [ 5] .shstrtab
         STRTAB          00000000 000080 000030 00   0   0  1
         [00000000]: 
    [ 6] .symtab
         SYMTAB          00000000 0001f0 000090 10   7   5  4
         [00000000]: 
    [ 7] .strtab
         STRTAB          00000000 000280 000025 00   0   0  1
         [00000000]: 

  Relocation section '.rel.text' at offset 0x2a8 contains 4 entries:
   Offset     Info    Type            Sym.Value  Sym. Name
  00000000  00000805 unrecognized: 5       00000000   _gp_disp
  00000004  00000806 unrecognized: 6       00000000   _gp_disp
  00000020  00000616 unrecognized: 16      00000004   gI
  00000028  00000617 unrecognized: 17      00000004   gI

  
  [Gamma@localhost input]$ llvm-readelf -tr ch6_1.mips.o 
  There are 9 section headers, starting at offset 0xc8:

  Section Headers:
    [Nr] Name
         Type            Addr     Off    Size   ES   Lk Inf Al
         Flags
    [ 0] 
         NULL            00000000 000000 000000 00   0   0  0
         [00000000]: 
    [ 1] .text
         PROGBITS        00000000 000034 000038 00   0   0  4
         [00000006]: ALLOC, EXEC
    [ 2] .rel.text
         REL             00000000 0002f8 000018 08   7   1  4
         [00000000]: 
    [ 3] .data
         PROGBITS        00000000 00006c 000008 00   0   0  4
         [00000003]: WRITE, ALLOC
    [ 4] .bss
         NOBITS          00000000 000074 000000 00   0   0  4
         [00000003]: WRITE, ALLOC
    [ 5] .reginfo
         MIPS_REGINFO    00000000 000074 000018 00   0   0  1
         [00000002]: ALLOC
    [ 6] .shstrtab
         STRTAB          00000000 00008c 000039 00   0   0  1
         [00000000]: 
    [ 7] .symtab
         SYMTAB          00000000 000230 0000a0 10   8   6  4
         [00000000]: 
    [ 8] .strtab
         STRTAB          00000000 0002d0 000025 00   0   0  1
         [00000000]: 

  Relocation section '.rel.text' at offset 0x2f8 contains 3 entries:
   Offset     Info    Type            Sym.Value  Sym. Name
  00000000  00000905 R_MIPS_HI16       00000000   _gp_disp
  00000004  00000906 R_MIPS_LO16       00000000   _gp_disp
  0000001c  00000709 R_MIPS_GOT16      00000004   gI


As depicted in `section Handle $gp register in PIC addressing mode`_, it 
translates **“.cpload %reg”** into the following.

.. code-block:: c++

  // Lower ".cpload $reg" to
  //  "lui   $gp, %hi(_gp_disp)"
  //  "ori $gp, $gp, %lo(_gp_disp)"
  //  "addu  $gp, $gp, $t9"

The _gp_disp value is determined by loader. So, it's undefined in obj. 
You can find both the Relocation Records for offset 0 and 4 of .text section 
refer to _gp_disp value. 
The offset 0 and 4 of .text section are instructions "lui $gp, %hi(_gp_disp)"
and "ori $gp, $gp, %lo(_gp_disp)" which their corresponding obj 
encode are 0fa00000 and  0daa0000, respectively. 
The obj translates the %hi(_gp_disp) and %lo(_gp_disp) into 0 since when loader 
loads this obj into memory, loader will know the _gp_disp value at run time and 
will update these two offset relocation records to the correct offset value. 
You can check if the cpu0 of %hi(_gp_disp) and %lo(_gp_disp) are correct by 
above mips Relocation Records of R_MIPS_HI(_gp_disp) and  R_MIPS_LO(_gp_disp) 
even though the cpu0 is not a CPU recognized by llvm-readelf utilitly. 
The instruction **“ld $2, %got(gI)($gp)”** is same since we don't know what the 
address of .data section variable will load to. 
So, Cpu0 translate the address to 0 and made a relocation record on 0x00000020 
of .text section. 
Linker or Loader will change this address when this program is 
linked or loaded depends on the program is static link or dynamic link.


Cpu0 ELF related files
~~~~~~~~~~~~~~~~~~~~~~

Files Cpu0ELFObjectWrite.cpp and Cpu0MC*.cpp are the files take care the obj 
format. 
Most obj code translation about specific instructions are defined by 
Cpu0InstrInfo.td and Cpu0RegisterInfo.td. 
With these td description, LLVM translate Cpu0 instructions into obj format 
automatically.


llvm-objdump
-------------

llvm-objdump -t -r
~~~~~~~~~~~~~~~~~~

``llvm-objdump -tr`` can display the information of relocation records 
like ``llvm-readelf -tr``. 
Let's run llvm-objdump with and without Cpu0 backend commands as follows to 
see the differences. 

.. code-block:: console

  118-165-83-12:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch9_3.cpp -emit-llvm -o ch9_3.bc
  118-165-83-10:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=pic -filetype=obj ch9_3.bc -o 
  ch9_3.cpu0.o

  118-165-78-12:input Jonathan$ objdump -t -r ch9_3.cpu0.o
  
  ch9_3.cpu0.o:     file format elf32-big

  SYMBOL TABLE:
  00000000 l    df *ABS*	00000000 ch9_3.bc
  00000000 l    d  .text	00000000 .text
  00000000 l    d  .data	00000000 .data
  00000000 l    d  .bss	00000000 .bss
  00000000 g     F .text	00000084 _Z5sum_iiz
  00000084 g     F .text	00000080 main
  00000000         *UND*	00000000 _gp_disp


  RELOCATION RECORDS FOR [.text]:
  OFFSET   TYPE              VALUE 
  00000084 UNKNOWN           _gp_disp
  00000088 UNKNOWN           _gp_disp
  000000e0 UNKNOWN           _Z5sum_iiz


  118-165-83-10:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llvm-objdump -t -r ch9_3.cpu0.o
  
  ch9_3.cpu0.o:	file format ELF32-CPU0

  RELOCATION RECORDS FOR [.text]:
  132 R_CPU0_HI16 _gp_disp
  136 R_CPU0_LO16 _gp_disp
  224 R_CPU0_CALL16 _Z5sum_iiz

  SYMBOL TABLE:
  00000000 l    df *ABS*	00000000 ch9_3.bc
  00000000 l    d  .text	00000000 .text
  00000000 l    d  .data	00000000 .data
  00000000 l    d  .bss	00000000 .bss
  00000000 g     F .text	00000084 _Z5sum_iiz
  00000084 g     F .text	00000080 main
  00000000         *UND*	00000000 _gp_disp


The llvm-objdump can display the file format and relocation records information 
well while the objdump cannot since we add the relocation records information 
in ELF.h as follows, 

.. rubric:: include/llvm/support/ELF.h
.. code-block:: c++

  // Machine architectures
  enum {
    ...
    EM_CPU0          = 998, // Document LLVM Backend Tutorial Cpu0
    EM_CPU0_LE       = 999  // EM_CPU0_LE: little endian; EM_CPU0: big endian
  }
  

.. rubric:: lib/object/ELF.cpp
.. code-block:: c++

  ...

  StringRef getELFRelocationTypeName(uint32_t Machine, uint32_t Type) {
    switch (Machine) {
    ...
    case ELF::EM_CPU0:
      switch (Type) {
  #include "llvm/Support/ELFRelocs/Cpu0.def"
      default:
        break;
      }
      break;
    ...
    }
  

.. rubric:: include/llvm/Support/ELFRelocs/Cpu0.def
.. literalinclude:: ../lbdex/llvm/modify/llvm/include/llvm/BinaryFormat/ELFRelocs/Cpu0.def

.. rubric:: include/llvm/Object/ELFObjectFile.h
.. code-block:: c++
  
  template<support::endianness target_endianness, bool is64Bits>
  error_code ELFObjectFile<target_endianness, is64Bits>
              ::getRelocationValueString(DataRefImpl Rel,
                        SmallVectorImpl<char> &Result) const {
    ...
    case ELF::EM_CPU0:  // llvm-objdump -t -r
    res = symname;
    break;
    ...
  }
  
  template<support::endianness target_endianness, bool is64Bits>
  StringRef ELFObjectFile<target_endianness, is64Bits>
               ::getFileFormatName() const {
    switch(Header->e_ident[ELF::EI_CLASS]) {
    case ELF::ELFCLASS32:
    switch(Header->e_machine) {
    ...
    case ELF::EM_CPU0:  // llvm-objdump -t -r
      return "ELF32-CPU0";
    ...
  }
  
  template<support::endianness target_endianness, bool is64Bits>
  unsigned ELFObjectFile<target_endianness, is64Bits>::getArch() const {
    switch(Header->e_machine) {
    ...
    case ELF::EM_CPU0:  // llvm-objdump -t -r
    return (target_endianness == support::little) ?
         Triple::cpu0el : Triple::cpu0;
    ...
  }

In addition to ``llvm-objdump -t -r``, the ``llvm-readobj -h`` can display the 
Cpu0 elf header information with EM_CPU0 defined above.


llvm-objdump -d
~~~~~~~~~~~~~~~~

Run the last Chapter example code with command ``llvm-objdump -d`` for dumping 
file from elf to hex as follows, 

.. code-block:: console

  JonathantekiiMac:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch8_1_1.cpp -emit-llvm -o ch8_1_1.bc
  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=pic -filetype=obj ch8_1_1.bc 
  -o ch8_1_1.cpu0.o
  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llvm-objdump -d ch8_1_1.cpu0.o
  
  ch8_1_1.cpu0.o: file format ELF32-unknown
  
  Disassembly of section .text:error: no disassembler for target cpu0-unknown-
  unknown

To support llvm-objdump, the following code added to Chapter10_1/ 
(the DecoderMethod for brtarget24 has been added in previous chapter).

.. rubric:: lbdex/chapters/Chapter10_1/CMakeLists.txt
.. literalinclude:: ../lbdex/Cpu0/CMakeLists.txt
    :start-after: #if CH >= CH10_1 1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/CMakeLists.txt
    :start-after: #if CH >= CH10_1 2
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/CMakeLists.txt
    :start-after: #if CH >= CH10_1 3
    :end-before: #endif
  
.. rubric:: lbdex/chapters/Chapter10_1/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //@JumpFR {
    :end-before: //@JumpFR }
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //@JumpLink {
    :end-before: //@JumpLink }
  
.. rubric:: lbdex/chapters/Chapter10_1/Disassembler/CMakeLists.txt
.. literalinclude:: ../lbdex/Cpu0/Disassembler/CMakeLists.txt
  
.. rubric:: lbdex/chapters/Chapter10_1/Disassembler/Cpu0Disassembler.cpp
.. literalinclude:: ../lbdex/chapters/Chapter10_1/Disassembler/Cpu0Disassembler.cpp
  

As above code, it adds directory Disassembler to handle the reverse translation 
from obj to assembly. So, add Disassembler/Cpu0Disassembler.cpp and modify 
the CMakeList.txt to build directory Disassembler, and 
enable the disassembler table generated by "has_disassembler = 1". 
Most of code is handled by the table defined in \*.td files. 
Not every instruction in \*.td can be disassembled without trouble even though 
they can be translated into assembly and obj successfully. 
For those cannot be disassembled, LLVM supply the **"let DecoderMethod"** 
keyword to allow programmers implement their decode function. 
For example in Cpu0, we define functions DecodeBranch24Target(), 
DecodeJumpTarget() and DecodeJumpFR() in 
Cpu0Disassembler.cpp and tell the llvm-tblgen by writing 
**"let DecoderMethod = ..."** in the corresponding instruction definitions or 
ISD node of Cpu0InstrInfo.td. 
LLVM will call these DecodeMethod when user uses Disassembler tools, such  
as ``llvm-objdump -d``.

Finally cpu032II includes all cpu032I instruction set and adds some instrucitons. 
When ``llvm-objdump -d`` is invoked, function selectCpu0ArchFeature() as 
the following will be called through createCpu0MCSubtargetInfo(). 
The llvm-objdump cannot set cpu option like llc as ``llc -mcpu=cpu032I``,
so the varaible CPU in selectCpu0ArchFeature() is empty when invoked by 
``llvm-objdump -d``. Set Cpu0ArchFeature to "+cpu032II" than it can disassemble 
all instructions (cpu032II include all cpu032I instructions and add some new
instructions).

.. rubric:: lbdex/chapters/Chapter10_1/MCTargetDesc/Cpu0MCTargetDesc.cpp
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0MCTargetDesc.cpp
    :start-after: //@1 {
    :end-before: //@1 }


Now, run Chapter10_1/ with command ``llvm-objdump -d ch8_1_1.cpu0.o`` will get 
the following result.

.. code-block:: console

  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=pic -filetype=obj 
  ch8_1_1.bc -o ch8_1_1.cpu0.o
  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llvm-objdump -d ch8_1_1.cpu0.o
  
  ch8_1_1.cpu0.o:	file format ELF32-CPU0

  Disassembly of section .text:
  _Z13test_control1v:
         0: 09 dd ff d8                                   addiu $sp, $sp, -40
         4: 09 30 00 00                                   addiu $3, $zero, 0
         8: 02 3d 00 24                                   st  $3, 36($sp)
         c: 09 20 00 01                                   addiu $2, $zero, 1
        10: 02 2d 00 20                                   st  $2, 32($sp)
        14: 09 40 00 02                                   addiu $4, $zero, 2
        18: 02 4d 00 1c                                   st  $4, 28($sp)
        ...


.. _section Handle $gp register in PIC addressing mode:
	http://jonathan2251.github.io/lbd/funccall.html#handle-gp-register-in-pic-addressing-mode


.. [#wiki-elf] http://en.wikipedia.org/wiki/Executable_and_Linkable_Format

.. [#beck] Leland Beck, System Software: An Introduction to Systems Programming. 

.. [#lk-out] http://ccckmit.wikidot.com/lk:aout

.. [#lk-obj] http://ccckmit.wikidot.com/lk:objfile

.. [#lk-elf] http://ccckmit.wikidot.com/lk:elffile

