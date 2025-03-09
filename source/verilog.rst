.. _sec-verilog:

Verify backend on Verilog simulator
===================================

.. contents::
   :local:
   :depth: 4

Until now, we have developed an LLVM backend capable of compiling C or 
assembly code, as illustrated in the white part of :numref:`runbackend-f1`.  
If the program does not contain global variables, the ELF object file can  
be dumped to a hex file using the following command:

.. code-block:: bash

   llvm-objdump -d

This functionality was completed in Chapter *ELF Support*.

.. _runbackend-f1: 
.. graphviz:: ../Fig/verilog/verilog.gv
   :caption: Cpu0 backend without linker

This chapter implements the Cpu0 instructions using the Verilog language, 
as represented by the gray part in the figure above.

With this Verilog-based machine, we can execute the hex program generated 
by the LLVM backend on the Cpu0 Verilog simulator running on a PC. 
This allows us to observe and verify the execution results of Cpu0 instructions 
directly on the hardware model.


Create Verilog Simulator of Cpu0
--------------------------------

Verilog is an IEEE-standard language widely used in IC design. There are many
books and free online resources available for learning Verilog [#free-doc1]_
[#free-doc2]_ [#free-doc3]_ [#free-doc4]_ [#free-doc5]_.

Verilog is also known as Verilog HDL (Hardware Description Language), not to be
confused with **VHDL**, which serves the same purpose but is a competing
language [#vhdl]_.

An example implementation, ``lbdex/verilog/cpu0.v``, contains the Cpu0 processor
design written in Verilog. As described in Appendix A, we have installed the
Icarus Verilog tool on both iMac and Linux systems. The ``cpu0.v`` design is
relatively simple, with only a few hundred lines of code in total.

Although this implementation does not include pipelining, it simulates delay
slots (via the ``SIMULATE_DELAY_SLOT`` section of the code) to accurately
estimate pipeline machine cycles.

Verilog has a C-like syntax, and since this book focuses on compiler
implementation, we present the ``cpu0.v`` code and the build commands below
**without an in-depth explanation**. We expect that readers with some patience
and curiosity will be able to understand the Verilog code.

Cpu0 supports **memory-mapped I/O**, one of the two primary I/O models in
computer architecture (the other being **instruction-based I/O**). Cpu0 maps the
output port to memory address ``0x80000``. When executing the instruction:

::

  st $ra, cx($rb)

where ``cx($rb)`` equals ``0x80000``, the Cpu0 processor outputs the content to
that I/O port, as demonstrated below.

.. code-block:: verilog

      ST : begin
        ...
        if (R[b]+c16 == `IOADDR) begin
          outw(R[a]);

.. rubric:: lbdex/verilog/cpu0.v
.. literalinclude:: ../lbdex/verilog/cpu0.v

.. rubric:: lbdex/verilog/Makefile
.. literalinclude:: ../lbdex/verilog/Makefile


Since the Cpu0 Verilog machine supports both big-endian and little-endian  
modes, the memory and CPU modules communicate this configuration through  
a dedicated wire.

The endian information is stored in the ROM of the memory module. Upon  
system startup, the memory module reads this configuration and sends the  
endian setting to the CPU via the connected wire.

This mechanism is implemented according to the following code snippet:

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


Verify backend
--------------

Now let's compile ``ch_run_backend.cpp`` as shown below. Since code size grows
from low to high addresses and the stack grows from high to low addresses, the
``$sp`` register is set to ``0x7fffc``. This is because ``cpu0.v`` is assumed to
use ``0x80000`` bytes of memory.

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

To run program without linker implementation at this point, the ``boot.cpp`` must
be set at the beginning of code, and the ``main()`` of ``ch_run_backend.cpp``
comes immediately after it.

Let's run ``Chapter11_2/`` with ``llvm-objdump -d`` for input file
``ch_run_backend.cpp`` to generate the hex file via ``build-run_bacekend.sh``,
then feed the hex file to ``cpu0``'s Verilog simulator to get the output result
as below.

Remind that ``ch_run_backend.cpp`` has to be compiled with the option
``clang -target mips-unknown-linux-gnu`` since the example code
``ch9_3_vararg.cpp``, which uses vararg, needs to be compiled with this option.
Other example codes have no differences between this option and the default
option.

.. code-block:: console

  JonathantekiiMac:input Jonathan$ pwd
  /Users/Jonathan/llvm/test/lbdex/input
  JonathantekiiMac:input Jonathan$ bash build-run_backend.sh cpu032I eb
  JonathantekiiMac:input Jonathan$ cd ../verilog cd ../verilog
  JonathantekiiMac:input Jonathan$ pwd
  /Users/Jonathan/llvm/test/lbdex/verilog
  JonathantekiiMac:verilog Jonathan$ make
  JonathantekiiMac:verilog Jonathan$ ./cpu0Is
  WARNING: cpu0Is.v:386: $readmemh(cpu0.hex): Not enough words in the file for the 
  taskInterrupt(001)
  68
  7
  0
  0
  253
  3
  1
  13
  3
  -126
  130
  -32766
  32770
  393307
  16777218
  3
  4
  51
  2
  3
  1
  2147483647
  -2147483648
  15
  5
  0
  31
  49
  total cpu cycles = 50645               
  RET to PC < 0, finished!

  JonathantekiiMac:input Jonathan$ bash build-run_backend.sh cpu032II eb
  JonathantekiiMac:input Jonathan$ cd ../verilog
  JonathantekiiMac:verilog Jonathan$ ./cpu0IIs
  ...
  total cpu cycles = 48335               
  RET to PC < 0, finished!

The "total CPU cycles" are calculated in this Verilog simulator to allow
performance review of both the backend compiler and the CPU.

Only CPU cycles are counted in this implementation, as I/O cycle times are
unknown.

As explained in Chapter "Control Flow Statements", ``cpu032II``, which uses
instructions ``slt`` and ``beq``, performs better than ``cmp`` and ``jeq`` in
``cpu032I``.

The instruction ``jmp`` has no delay slot, making it preferable for use in
dynamic linker implementations.

You can trace memory binary code and changes to destination registers at every
instruction execution by unmarking ``TRACE`` in the Makefile, as shown below:

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


As shown in the result above, ``cpu0.v`` dumps the memory content after reading
the input file ``cpu0.hex``. Next, it runs instructions from address 0 and prints
each destination register value in the fourth column.

The first column is the timestamp in nanoseconds. The second column is the
instruction address. The third column is the instruction content.

Most of the example codes discussed in previous chapters are verified by printing
variables using ``print_integer()``.

Since the ``cpu0.v`` machine is written in Verilog, it is assumed to be capable of
running on a real FPGA device (though I have not tested this myself). The actual
output hardware interface or port depends on the specific output device, such as
RS232, speaker, LED, etc. You must implement the I/O interface or port and wire
your I/O device accordingly when programming an FPGA.

By running the compiled code on the Verilog simulator, the compiled result from
the Cpu0 backend and the total CPU cycles can be verified and measured.

Currently, this Cpu0 Verilog implementation does not support pipeline
architecture. However, based on the instruction set, it can be extended to a
pipelined model.

The cycle time of the pipelined Cpu0 model is expected to be more than 1/5 of the
"total CPU cycles" shown above, due to dependencies between instructions.

Although the Verilog simulator is slow for running full system programs and does
not count cycles for cache and I/O operations, it provides a simple and effective
way to validate CPU design ideas in the early development stages using small
program patterns.

Creating a full system simulator is complex. While the Wiki website [#wiki-sim]_
provides tools for building simulators, doing so requires significant effort.

To generate ``cpu032I`` code with little-endian format, you can run the following
command. The script ``build-run_backend.sh`` writes the endian configuration to
``../verilog/cpu0.config`` as shown below.

.. code-block:: console

  JonathantekiiMac:input Jonathan$ bash build-run_backend.sh cpu032I el

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

  JonathantekiiMac:input Jonathan$ bash build-run_backend.sh cpu032II el
  ...
  JonathantekiiMac:input Jonathan$ cd ../verilog
  JonathantekiiMac:verilog Jonathan$ ./cpu0IIs
  ...
  31
  ...


Other LLVM-Based Tools for Cpu0 Processor
------------------------------------------

You can find the Cpu0 ELF linker implementation based on ``lld``, which is the
official LLVM linker project, as well as ``elf2hex``, which is modified from the
``llvm-objdump`` driver, at the following website:

::

  http://jonathan2251.github.io/lbt/index.html


.. [#free-doc1] http://ccckmit.wikidot.com/ve:main

.. [#free-doc2] http://www.ece.umd.edu/courses/enee359a/

.. [#free-doc3] http://www.ece.umd.edu/courses/enee359a/verilog_tutorial.pdf

.. [#free-doc4] http://d1.amobbs.com/bbs_upload782111/files_33/ourdev_585395BQ8J9A.pdf

.. [#free-doc5] http://en.wikipedia.org/wiki/Verilog

.. [#vhdl] http://en.wikipedia.org/wiki/VHDL

.. [#wiki-sim] https://en.wikipedia.org/wiki/Computer_architecture_simulator
