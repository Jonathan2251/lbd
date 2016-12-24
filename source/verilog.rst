.. _sec-verilog:

Verify backend on Verilog simulator
===================================

.. contents::
   :local:
   :depth: 4

Until now, we have an llvm backend to compile C or assembly as the blue part of 
:numref:`runbackend-f1`. If without global variable, the elf obj can be 
dumped to hex file via ``llvm-objdump -d`` which finished in Chapter ELF Support.

.. _runbackend-f1: 
.. figure:: ../Fig/verilog/1.png
  :scale: 50 %
  :align: center

  Cpu0 backend without linker


This chapter will implement Cpu0 instructions by Verilog language as the red 
part of :numref:`runbackend-f1`.
With this Verilog machine, we can run this hex program on the Cpu0 Verilog 
machine on PC and see the Cpu0 instructions execution result.


Create verilog simulator of Cpu0
--------------------------------

Verilog language is an IEEE standard in IC design. There are a lot of books and 
documents for this language. Free documents exist in Web sites [#free-doc1]_ 
[#free-doc2]_ [#free-doc3]_ [#free-doc4]_ [#free-doc5]_. 
Verilog also called as Verilog HDL but not VHDL. 
VHDL is the same purpose language which compete against Verilog.
About VHDL reference here [#vhdl]_.
Example code, lbdex/verilog/cpu0.v, is the Cpu0 design in Verilog. 
In Appendix A, we have downloaded and installed Icarus Verilog tool both on 
iMac and Linux. The cpu0.v is a simple design 
with only few hundreds lines of code totally. 
This implementation hasn't the pipeline features, but through implement
the delay slot simulation (SIMULATE_DELAY_SLOT part of code), the exact pipeline
machine cycles can be calculated.

Verilog is a C like language in syntex and 
this book is a compiler book, so we list the cpu0.v as well as the building 
command without explanation as below. 
We expect readers can understand the Verilog code just with a little patience in
reading it. 
There are two type of I/O according computer architecture. 
One is memory mapped I/O, the other is instruction I/O. 
Cpu0 uses memory mapped I/O where memory address 0x80000 as the output port. 
When meet the instruction **"st $ra, cx($rb)"**, where cx($rb) is 
0x80000, Cpu0 displays the content as follows,

.. code-block:: verilog

      ST : begin
        ...
        if (R[b]+c16 == `IOADDR) begin
          outw(R[a]);

.. rubric:: lbdex/verilog/cpu0.v
.. literalinclude:: ../lbdex/verilog/cpu0.v

.. rubric:: lbdex/verilog/Makefile
.. literalinclude:: ../lbdex/verilog/Makefile


Since Cpu0 Verilog machine supports both big and little endian, the memory 
and cpu module both have a wire connectting each other. 
The endian information stored in ROM of memory module, and memory module send 
the information when it is up according the following code,

.. rubric:: lbdex/verilog/cpu0.v
.. code-block:: console

    assign cfg = mconfig[0][0:0];
    ...
    wire cfg;

    cpu0 cpu(.clock(clock), .itype(itype), .pc(pc), .tick(tick), .ir(ir),
    .mar(mar), .mdr(mdr), .dbus(dbus), .m_en(m_en), .m_rw(m_rw), .m_size(m_size),
    .cfg(cfg));

    memory0 mem(.clock(clock), .reset(reset), .en(m_en), .rw(m_rw), 
    .m_size(m_size), .abus(mar), .dbus_in(mdr), .dbus_out(dbus), .cfg(cfg));


Instead of setting endian tranfer in memory module, the endian transfer can 
also be set in CPU module, and memory moudle always return with big endian.
I am not an professional engineer in FPGA/CPU hardware design. 
But according book "Computer 
Architecture: A Quantitative Approach", some operations may have no tolerance 
in time of execution stage. Any endian swap will make the clock cycle time 
longer and affect the CPU performance. So, I set the endian transfer in memory
module. In system with bus, it will be set in bus system I think.


Verify backend
--------------

Now let's compile ch_run_backend.cpp as below. Since code size grows up from 
low to high address and stack grows up from high to low address. $sp is set 
at 0x7fffc because assuming cpu0.v uses 0x80000 bytes of memory.

.. rubric:: lbdex/input/start.h
.. literalinclude:: ../lbdex/input/start.h
    :start-after: /// start

.. rubric:: lbdex/input/boot.cpp
.. literalinclude:: ../lbdex/input/boot.cpp
    :start-after: /// start

.. rubric:: lbdex/input/print.h
.. literalinclude:: ../lbdex/input/print.h
    :start-after: /// start

.. rubric:: lbdex/input/print.cpp
.. literalinclude:: ../lbdex/input/print.cpp
    :start-after: /// start

.. rubric:: lbdex/input/ch_nolld.h
.. literalinclude:: ../lbdex/input/ch_nolld.h
    :start-after: /// start

.. rubric:: lbdex/input/ch_nolld.cpp
.. literalinclude:: ../lbdex/input/ch_nolld.cpp
    :start-after: /// start

.. rubric:: lbdex/input/ch_run_backend.cpp
.. literalinclude:: ../lbdex/input/ch_run_backend.cpp
    :start-after: /// start

.. rubric:: lbdex/input/functions.sh
.. literalinclude:: ../lbdex/input/functions.sh

.. rubric:: lbdex/input/build-run_backend.sh
.. literalinclude:: ../lbdex/input/build-run_backend.sh

To run program without linker implementation at this point, the boot.cpp must be 
set at the beginning of code, and the main() of ch_run_backend.cpp comes 
immediately after it.
Let's run Chapter11_2/ with ``llvm-objdump -d`` for input file 
ch_run_backend.cpp to generate the hex file via build-run_bacekend.sh, then 
feed hex file to cpu0Is Verilog simulator to get the output result as below. 
Remind ch_run_backend.cpp have to be compiled with option 
``clang -target mips-unknown-linux-gnu`` since the example code 
ch9_3_vararg.cpp which uses the vararg needs to be compiled with this option. 
Other example codes have no differences between this option and default option. 


.. code-block:: console

  JonathantekiiMac:input Jonathan$ pwd
  /Users/Jonathan/llvm/test/lbdex/input
  JonathantekiiMac:input Jonathan$ bash build-run_backend.sh cpu032I be
  JonathantekiiMac:input Jonathan$ cd ../verilog cd ../verilog
  JonathantekiiMac:input Jonathan$ pwd
  /Users/Jonathan/llvm/test/lbdex/verilog
  JonathantekiiMac:verilog Jonathan$ make
  JonathantekiiMac:verilog Jonathan$ ./cpu0Is
  WARNING: cpu0Is.v:386: $readmemh(cpu0.hex): Not enough words in the file for the 
  taskInterrupt(001)
  74
  7
  0
  0
  253
  3
  1
  14
  3
  -126
  130
  -32766
  32770
  393307
  16777222
  2
  4
  51
  2
  2147483647
  -2147483648
  7
  120
  15
  5
  0
  31
  49
  total cpu cycles = 50645               
  RET to PC < 0, finished!

  JonathantekiiMac:input Jonathan$ bash build-run_backend.sh cpu032II be
  JonathantekiiMac:input Jonathan$ cd ../verilog
  JonathantekiiMac:verilog Jonathan$ ./cpu0IIs
  ...
  total cpu cycles = 48335               
  RET to PC < 0, finished!

The "total cpu cycles" is calculated in this verilog simualtor so that the 
backend compiler and CPU performance can be reviewed.
Only the CPU cycles are counted in this implemenation since I/O 
cycles time is unknown.
As explained in chapter "Control flow statements", cpu032II which uses 
instructions slt and beq
has better performance than cmp and jeq in cpu032I.
Instructions "jmp" has no delay slot so it is better used in dynamic linker 
implementation.

You can trace the memory binary code and destination
register changed at every instruction execution by unmark TRACE in Makefile as 
below,

.. rubric:: lbdex/verilog/Makefile

.. code-block:: c++

      TRACE=-D TRACE

.. code-block:: console

  JonathantekiiMac:raw Jonathan$ ./cpu0Is
  WARNING: cpu0.v:386: $readmemh(cpu0.hex): Not enough words in the file for the 
  requested range [0:28671].
  00000000: 2600000c
  00000004: 26000004
  00000008: 26000004
  0000000c: 26fffffc
  00000010: 09100000
  00000014: 09200000
  ...
  taskInterrupt(001)
  1530ns 00000054 : 02ed002c m[28620+44  ]=-1          SW=00000000
  1610ns 00000058 : 02bd0028 m[28620+40  ]=0           SW=00000000
  ...                     
  RET to PC < 0, finished!


As above result, cpu0.v dumps the memory first after reading input file cpu0.hex. 
Next, it runs instructions from address 0 and print each destination 
register value in the fourth column. 
The first column is the nano seconds of timing. The second 
is instruction address. The third is instruction content. 
Now, most example codes depicted in the previous chapters are verified by 
print the variable with print_integer().

Since the cpu0.v machine is created by Verilog language, suppose it can run on
real FPGA device (but I never do it). 
The real output hardware 
interface/port is hardware output device dependent, such as RS232, speaker, 
LED, .... You should implement the I/O interface/port when you want to program 
FPGA and wire I/O device to the I/O port. 
Through running the compiled code on Verilog simulator, Cpu0 backend compiled 
result and CPU cycles are verified and calculated.
Currently, this Cpu0 Verilog program is not a pipeline architecture, but 
according the instruction set it can be implemented as a pipeline model.
The cycle time of Cpu0 pipeline model is more than 1/5 of "total cpu cycles" 
displayed as above since there are dependences exist between instructions.
Though the Verilog simulator is slow in running the whole system program and
not include the cycles counting in cache and I/O, it is a simple and easy way
to verify your idea about CPU design at early stage with small program pattern.
The overall system simulator is complex to create. Even wiki web site here 
[#wiki-sim]_ include tools for creating the simulator, it needs a lot of effort.

To generate cpu032I as well as little endian code, you can run with the 
following command. File build-run_backend.sh write the endian information to 
../verilog/cpu0.config as below.

.. code-block:: console

  JonathantekiiMac:input Jonathan$ bash build-run_backend.sh cpu032I le

.. rubric:: ../verilog/cpu0.config
.. code-block:: c++

  1   /* 0: big endian, 1: little endian */

The following files test more features.

.. rubric:: lbdex/input/ch_nolld2.h
.. literalinclude:: ../lbdex/input/ch_nolld2.h
    :start-after: /// start

.. rubric:: lbdex/input/ch_nolld2.cpp
.. literalinclude:: ../lbdex/input/ch_nolld2.cpp
    :start-after: /// start

.. rubric:: lbdex/input/ch_run_backend2.cpp
.. literalinclude:: ../lbdex/input/ch_run_backend2.cpp
    :start-after: /// start

.. rubric:: lbdex/input/build-run_backend2.sh
.. literalinclude:: ../lbdex/input/build-run_backend2.sh
  
.. code-block:: console

  JonathantekiiMac:input Jonathan$ bash build-run_backend.sh cpu032II le
  ...
  JonathantekiiMac:input Jonathan$ cd ../verilog
  JonathantekiiMac:verilog Jonathan$ ./cpu0IIs
  ...
  31
  ...


Other llvm based tools for Cpu0 processor
------------------------------------------

You can find the Cpu0 ELF linker implementation based on lld which is the 
llvm official linker project, as well as elf2hex which modified from llvm-objdump
driver at web: http://jonathan2251.github.io/lbt/index.html.


.. [#free-doc1] http://ccckmit.wikidot.com/ve:main

.. [#free-doc2] http://www.ece.umd.edu/courses/enee359a/

.. [#free-doc3] http://www.ece.umd.edu/courses/enee359a/verilog_tutorial.pdf

.. [#free-doc4] http://d1.amobbs.com/bbs_upload782111/files_33/ourdev_585395BQ8J9A.pdf

.. [#free-doc5] http://en.wikipedia.org/wiki/Verilog

.. [#vhdl] http://en.wikipedia.org/wiki/VHDL

.. [#wiki-sim] https://en.wikipedia.org/wiki/Computer_architecture_simulator
