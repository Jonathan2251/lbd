.. _sec-appendix-installing:

Appendix A: Getting Started: Installing LLVM and the Cpu0 example code
======================================================================

.. contents::
   :local:
   :depth: 4

Cpu0 example code, lbdex, can be found at near left bottom of this web site. Or 
here http://jonathan2251.github.io/lbd/lbdex.tar.gz.

In this chapter, we will run through how to set up LLVM using if you are using 
Mac OS X or Linux.
For information on using ``cmake`` to build LLVM, please refer to the "Building 
LLVM with CMake" [#llvm-cmake]_ documentation for further information. 

We will install two llvm directories in this chapter. One is the directory 
~/llvm/release/ which contains the clang and clang++ compiler we will use to 
translate the C/C++ input file into llvm IR. 
The other is the directory ~/llvm/test/ which contains our cpu0 backend 
program and clang.

This chapter details the installation of related software for this book.
If you are know well in llvm/clang installation or think it is too details, you
can run the bash script files after you install the Xcode and cmake as follows,

.. code-block:: console

  118-165-78-111:Jonathan$ pwd
  /Users/Jonathan/test
  118-165-78-111:Jonathan$ cp /Users/Jonathan/Downloads/
  lbdex.tar.gz .
  118-165-78-111:Jonathan$ tar -zxvf lbdex.tar.gz
  118-165-78-111:Jonathan$ cd lbdex/install_llvm
  118-165-78-111:install_llvm Jonathan$ ls
  build-llvm.sh
  118-165-78-111:install_llvm Jonathan$ bash build-llvm.sh
  ...

The contents of these two script files as follows,

.. rubric:: lbdex/install_llvm/build-llvm.sh
.. literalinclude:: ../lbdex/install_llvm/build-llvm.sh


Setting Up Your Mac
-------------------

The Xcode include clang and llvm already. The following three sub-sections are 
needless. List them just for readers who like to build clang and llvm with 
cmake GUI interface.

Installing Xcode and cmake
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. todo:: Fix centering for figure captions.


Install Xcode from the Mac App Store. Then install cmake, which can be found 
here: [#cmake]_. 
Before installing cmake, ensure you can install applications you download 
from the Internet. 
Open :menuselection:`System Preferences --> Security & Privacy`. Click the 
**lock** to make changes, and under "Allow applications downloaded from:" select 
the radio button next to "Anywhere." See :numref:`install-f2` below for an 
illustration. You may want to revert this setting after installing cmake.

.. _install-f2:
.. figure:: ../Fig/install/2.png
  :align: center

  Adjusting Mac OS X security settings to allow cmake installation.
  
Alternatively, you can mount the cmake .dmg image file you downloaded. Untar 
the latest cmake for Darwin, copy the cmake /Applications/ and set PATH as follows,

.. code-block:: console
  
  114-43-208-90:build Jonathan$ cat ~/.profile
  export PATH=$PATH:/Applications/CMake.app/Contents/bin

.. stop 12/5/12 10PM (just a bookmark for me to continue from)

Install Icarus Verilog tool on iMac
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install Icarus Verilog tool by command ``brew install icarus-verilog`` as follows,

.. code-block:: console

  JonathantekiiMac:~ Jonathan$ brew install icarus-verilog
  ==> Downloading ftp://icarus.com/pub/eda/verilog/v0.9/verilog-0.9.5.tar.gz
  ######################################################################## 100.0%
  ######################################################################## 100.0%
  ==> ./configure --prefix=/usr/local/Cellar/icarus-verilog/0.9.5
  ==> make
  ==> make installdirs
  ==> make install
  /usr/local/Cellar/icarus-verilog/0.9.5: 39 files, 12M, built in 55 seconds


Install other tools on iMac
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These tools mentioned in this section is for coding and debug. 
You can work even without these tools. 
Files compare tools Kdiff3 came from web site [#kdiff3]_. 
FileMerge is a part of Xcode, you can type FileMerge in Finder – Applications 
as :numref:`install-f11` and drag it into the Dock as 
:numref:`install-f12`.

.. _install-f11:
.. figure:: ../Fig/install/11.png
  :align: center

  Type FileMerge in Finder – Applications

.. _install-f12:
.. figure:: ../Fig/install/12.png
  :align: center

  Drag FileMege into the Dock

Download tool Graphviz for display llvm IR nodes in debugging, 
[#graphviz-dm]_. 
We choose mountainlion as :numref:`install-f13` since our iMac is Mountain 
Lion.

.. _install-f13:
.. figure:: ../Fig/install/13.png
  :height: 738 px
  :width: 1181 px
  :scale: 80 %
  :align: center

  Download graphviz for llvm IR node display

After install Graphviz, please set the path to .profile. 
For example, we install the Graphviz in directory 
/Applications/Graphviz.app/Contents/MacOS/, so add this path to 
/User/Jonathan/.profile as follows,

.. code-block:: console

  118-165-12-177:input Jonathan$ cat /Users/Jonathan/.profile
  export PATH=$PATH:/Applications/Xcode.app/Contents/bin:
  /Applications/Graphviz.app/Contents/MacOS/:/Users/Jonathan/llvm/release/
  build/bin

The Graphviz information for llvm is at section "SelectionDAG Instruction 
Selection Process" " of "The LLVM Target-Independent Code Generator" here 
[#isp]_  and at section "Viewing graphs while debugging code" of "LLVM 
Programmer’s Manual" here [#vgwdc]_.
TextWrangler is for edit file with line number display and dump binary file 
like the obj file, \*.o, that will be generated in chapter of Generating object 
files if you havn't gobjdump available. 
You can download from App Store. 
To dump binary file, first, open the binary file, next, select menu 
**“File – Hex Front Document”** as :numref:`install-f14`. 
Then select **“Front document's file”** as :numref:`install-f15`.

.. _install-f14:
.. figure:: ../Fig/install/14.png
  :align: center

  Select Hex Dump menu

.. _install-f15:
.. figure:: ../Fig/install/15.png
  :align: center

  Select Front document's file in TextWrangler
  
Install binutils by command ``brew install binutils`` as follows,

.. code-block:: console

  // get brew by the following ruby command if you don't have installed brew
  118-165-77-214:~ Jonathan$ ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" < /dev/null 2> /dev/null
  118-165-77-214:~ Jonathan$ brew install binutils
  ==> Downloading http://ftpmirror.gnu.org/binutils/binutils-2.22.tar.gz
  ######################################################################## 100.0%
  ==> ./configure --program-prefix=g --prefix=/usr/local/Cellar/binutils/2.22 
  --infodir=/usr/loca
  ==> make
  ==> make install
  /usr/local/Cellar/binutils/2.22: 90 files, 19M, built in 4.7 minutes
  118-165-77-214:~ Jonathan$ ls /usr/local/Cellar/binutils/2.22
  COPYING     README      lib
  ChangeLog     bin       share
  INSTALL_RECEIPT.json    include       x86_64-apple-darwin12.2.0
  118-165-77-214:binutils-2.23 Jonathan$ ls /usr/local/Cellar/binutils/2.22/bin
  gaddr2line  gc++filt  gnm   gobjdump  greadelf  gstrings
  gar   gelfedit  gobjcopy  granlib gsize   gstrip


Setup env vaiable
~~~~~~~~~~~~~~~~~

To access those execution files, edit .profile (if you .profile not exists, 
please create file .profile), save .profile to /Users/Jonathan/, and enable 
$PATH by command ``source .profile`` as follows. 
Please add path /Applications//Xcode.app/Contents/Developer/usr/bin to .profile 
if you didn't add it after Xcode download.

.. code-block:: console

  118-165-65-128:~ Jonathan$ pwd
  /Users/Jonathan
  118-165-65-128:~ Jonathan$ cat .profile 
  export PATH=$PATH:/Applications/Xcode.app/Contents/Developer/usr/bin:/Applicatio
  ns/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/:/Ap
  plications/Graphviz.app/Contents/MacOS/:/Users/Jonathan/llvm/release/build/bin
  export WORKON_HOME=$HOME/.virtualenvs
  source /usr/local/bin/virtualenvwrapper.sh # where Homebrew places it
  export VIRTUALENVWRAPPER_VIRTUALENV_ARGS='--no-site-packages' # optional
  118-165-65-128:~ Jonathan$ 

Build llvm with Cpu0 by terminal cmake command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have installed llvm with clang on directory llvm/release/. 
Now, we want to install llvm with our cpu0 backend code on directory 
llvm/test/ in this section.


.. code-block:: console

  118-165-78-111:lbdex Jonathan$ pwd
  /Users/Jonathan/Dowload/lbdex
  118-165-78-111:lbdex Jonathan$ bash build-cpu0.sh

Since Mac uses clang compiler and lldb instead of gcc and gdb, we can run lldb 
debug as follows, 

.. code-block:: console

  118-165-65-128:input Jonathan$ pwd
  /Users/Jonathan/Download/lbdex/input
  118-165-65-128:input Jonathan$ clang -c ch3.cpp -emit-llvm -o ch3.bc
  118-165-65-128:input Jonathan$ /Users/Jonathan/llvm/test/
  build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm 
  ch3.bc -o -
  118-165-65-128:input Jonathan$ lldb -- /Users/Jonathan/llvm/test/
  build/bin/llc -march=cpu0 -relocation-model=pic -filetype=
  asm ch3.bc -o -
  Current executable set to '/Users/Jonathan/llvm/test/build/bin/
  llc' (x86_64).
  (lldb) b Cpu0TargetInfo.cpp:19
  breakpoint set --file 'Cpu0TargetInfo.cpp' --line 19
  Breakpoint created: 1: file ='Cpu0TargetInfo.cpp', line = 19, locations = 1
  (lldb) run
  Process 6058 launched: '/Users/Jonathan/llvm/test/build/bin/
  llc' (x86_64)
  Process 6058 stopped
  * thread #1: tid = 0x1c03, 0x000000010077f231 llc`LLVMInitializeCpu0TargetInfo 
  + 33 at Cpu0TargetInfo.cpp:20, stop reason = breakpoint 1.1
    frame #0: 0x000000010077f231 llc`LLVMInitializeCpu0TargetInfo + 33 at 
    Cpu0TargetInfo.cpp:20
     16   
     17   extern "C" void LLVMInitializeCpu0TargetInfo() {
     18     RegisterTarget<Triple::cpu0,
  -> 19           /*HasJIT=*/true> X(TheCpu0Target, "cpu0", "Cpu0");
     [experimental]");
  (lldb) print X
  (llvm::RegisterTarget<llvm::Triple::ArchType, true>) $0 = {}
  (lldb) quit
  118-165-65-128:input Jonathan$ 

About the lldb debug command, please reference [#lldb-gdb]_ or lldb portal 
[#lldb]_. 



Setting Up Your Linux Machine
-----------------------------

Install Icarus Verilog tool on Linux
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the snapshot version of Icarus Verilog tool from web site, 
ftp://icarus.com/pub/eda/verilog/snapshots or go to http://iverilog.icarus.com/ 
and click snapshot version link. Follow the INSTALL file guide to install it. 


Install other tools on Linux
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Download Graphviz from [#graphviz-download]_ according your 
Linux distribution. Files compare tools Kdiff3 came from web site [#kdiff3]_. 


Install Release build on Linux
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, install the llvm release build by,

  1) Untar llvm source, rename llvm source with llvm.
  
  2) Untar clang and move it llvm/tools/clang.


Next, build with cmake command, ``cmake -DCMAKE_BUILD_TYPE=Release -DCLANG_BUILD
_EXAMPLES=ON -DLLVM_BUILD_EXAMPLES=ON -G "Unix Makefiles" ../llvm/``, as follows.

.. code-block:: console

  [Gamma@localhost build]$ pwd
  /home/cschen/llvm/release/build
  [Gamma@localhost build]$ cmake -DCMAKE_BUILD_TYPE=Release 
  -DCLANG_BUILD_EXAMPLES=ON -DLLVM_BUILD_EXAMPLES=ON -G "Unix Makefiles" ../llvm/
  -- The C compiler identification is GNU 4.8.2
  ...
  -- Targeting XCore
  -- Clang version: 3.9
  -- Found Subversion: /usr/bin/svn (found version "1.7.6") 
  -- Configuring done
  -- Generating done
  -- Build files have been written to: /home/cschen/llvm/release/build

After cmake, run command ``make``, then you can get clang, llc, llvm-as, ..., 
in build/bin/ after a few tens minutes of build. 
To speed up make process via SMP power, please check your core numbers by the 
following command then do make the next.

.. code-block:: console

  [Gamma@localhost build]$ cat /proc/cpuinfo | grep processor | wc -l
  8
  [Gamma@localhost build]$ make -j8 -l8

Next, edit 
/home/Gamma/.bash_profile with adding /home/cschen/llvm/release/build/
bin to PATH to enable the clang, llc, ..., command search path, as follows,

.. code-block:: console

  [Gamma@localhost ~]$ pwd
  /home/Gamma
  [Gamma@localhost ~]$ cat .bash_profile
  # .bash_profile
  
  # Get the aliases and functions
  if [ -f ~/.bashrc ]; then
    . ~/.bashrc
  fi
  
  # User specific environment and startup programs
  
  PATH=$PATH:/usr/local/sphinx/bin:~/llvm/release/build/bin:
  ... 
  export PATH
  [Gamma@localhost ~]$ source .bash_profile
  [Gamma@localhost ~]$ $PATH
  bash: /usr/lib64/qt-3.3/bin:/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:
  /usr/sbin:/usr/local/sphinx/bin:/home/Gamma/.local/bin:/home/Gamma/bin:
  /usr/local/sphinx/bin:/home/cschen/llvm/release/build/bin


Build llvm with Cpu0 by terminal cmake command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have installed llvm with clang on directory llvm/release/. 
Now, we want to install llvm with our cpu0 backend code on directory 
llvm/test/ in this section.


.. code-block:: console

  118-165-78-111:lbdex Jonathan$ pwd
  /home/Gamma/Dowload/lbdex
  118-165-78-111:lbdex Jonathan$ bash build-cpu0.sh

Now, we are ready for the cpu0 backend development. We can run gdb debug
as follows. 
If your setting has anything about gdb errors, please follow the errors 
indication (maybe need to download gdb again). 
Finally, try gdb as follows.

.. code-block:: console

  [Gamma@localhost input]$ pwd
  /home/Gamma/Downloads/lbdex/input
  [Gamma@localhost input]$ clang -c ch3.cpp -emit-llvm -o ch3.bc
  [Gamma@localhost input]$ gdb -args ~/llvm/test/
  build/bin/llc -march=cpu0 -relocation-model=pic -filetype=obj 
  ch3.bc -o ch3.cpu0.o
  GNU gdb (GDB) Fedora (7.4.50.20120120-50.fc17)
  Copyright (C) 2012 Free Software Foundation, Inc.
  License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
  This is free software: you are free to change and redistribute it.
  There is NO WARRANTY, to the extent permitted by law.  Type "show copying"
  and "show warranty" for details.
  This GDB was configured as "x86_64-redhat-linux-gnu".
  For bug reporting instructions, please see:
  <http://www.gnu.org/software/gdb/bugs/>...
  Reading symbols from /home/cschen/llvm/test/build/bin/llc.
  ..done.
  (gdb) break Cpu0TargetInfo.cpp:19
  Breakpoint 1 at 0xd54441: file /home/cschen/llvm/test/llvm/lib/Target/
  Cpu0/TargetInfo/Cpu0TargetInfo.cpp, line 19.
  (gdb) run
  Starting program: /home/cschen/llvm/test/build/bin/llc 
  -march=cpu0 -relocation-model=pic -filetype=obj ch3.bc -o ch3.cpu0.o
  [Thread debugging using libthread_db enabled]
  Using host libthread_db library "/lib64/libthread_db.so.1".
  
  Breakpoint 1, LLVMInitializeCpu0TargetInfo ()
    at /home/cschen/llvm/test/llvm/lib/Target/Cpu0/TargetInfo/Cpu0TargetInfo.cpp:20
  19          /*HasJIT=*/true> X(TheCpu0Target, "cpu0", "Cpu0");
  (gdb) quit
  A debugging session is active.
  
    Inferior 1 [process 10165] will be killed.
  
  Quit anyway? (y or n) y
  [Gamma@localhost input]$ 




.. [#llvm-cmake] http://llvm.org/docs/CMake.html?highlight=cmake

.. [#llvm-download] http://llvm.org/releases/download.html#3.9

.. [#cmake] http://www.cmake.org/cmake/resources/software.html

.. [#lldb-gdb] http://lldb.llvm.org/lldb-gdb.html

.. [#lldb] http://lldb.llvm.org/

.. [#test] http://llvm.org/docs/TestingGuide.html

.. [#kdiff3] http://kdiff3.sourceforge.net

.. [#graphviz-dm] http://www.graphviz.org/Download_macos.php

.. [#isp] http://llvm.org/docs/CodeGenerator.html#selectiondag-instruction-selection-process

.. [#vgwdc] http://llvm.org/docs/ProgrammersManual.html#viewing-graphs-while-debugging-code

.. [#graphviz-download] http://www.graphviz.org/Download.php
