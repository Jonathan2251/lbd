.. _sec-appendix-installing:

Appendix A: Getting Started: Installing LLVM and the Cpu0 example code
======================================================================

.. contents::
   :local:
   :depth: 4

Cpu0 example code, lbdex, can be found near left bottom of this web site. Or 
here http://jonathan2251.github.io/lbd/lbdex.tar.gz.

For information in using ``cmake`` to build LLVM, please refer to the "Building 
LLVM with CMake" [#llvm-cmake]_ documentation for further information. 

We install two llvm directories in this chapter. One is the directory 
~/llvm/debug/ which contains the clang and clang++ compiler we will use to 
translate the C/C++ input file into llvm IR. 
The other is the directory ~/llvm/test/ which contains our cpu0 backend 
program and clang.

Build steps
------------

On linux, multi-threading -DLLVM_PARALLEL_COMPILE_JOBS=4, need more than 16GB
memory. I create 64GB of swapfile to fix the problem of link failure
[#swapfile1]_ [#swapfile2]_. The iMac has no this problem.

.. code-block:: console

  $ cat /etc/fstab
  # <file system> <mount point>   <type>  <options>       <dump>  <pass>
  ...
  /swapfile       swap            swap    default         0       0

After setup brew install for iMac or install necessory packages. Build as 
https://github.com/Jonathan2251/lbd/blob/master/README.md.


Setting Up Your Mac
-------------------

Brew install and setup first [#installbrew]_.

.. code-block:: console

  % /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  [#installbrew]_.

Then add brew command to your path as the bottom of installed message of bash above, like the 
following.

.. code-block:: console

  % echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /Users/cschen/.zprofile
  % eval "$(/opt/homebrew/bin/brew shellenv)"

Install Icarus Verilog tool on iMac
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install Icarus Verilog tool by command ``brew install icarus-verilog`` as follows,

.. code-block:: console

  % brew install icarus-verilog
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

Install cmake as follows,

.. code-block:: console

  $ brew install cmake

Download Graphviz from [#graphviz-download]_ according your 
Linux distribution. Files compare tools Kdiff3 came from web site [#kdiff3]_. 

.. code-block:: console

  $ sudo apt install graphviz
  $ dot -V
  dot - graphviz version 2.40.1 (20161225.0304)

Install Graphviz for display llvm IR nodes in debugging, 
[#graphviz-dm]_. 

.. code-block:: console

  % brew install graphviz


The Graphviz information for llvm is at section "SelectionDAG Instruction 
Selection Process" " of "The LLVM Target-Independent Code Generator" here 
[#isp]_  and at section "Viewing graphs while debugging code" of "LLVM 
Programmer’s Manual" here [#vgwdc]_.
  
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


Setting Up Your Linux Machine
-----------------------------

Install Icarus Verilog tool on Linux
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download icarus verilog as follows [#icarus]_,

.. code-block:: console

  $ git clone http://iverilog.icarus.com/


Follow the README or INSTALL file guide to install it. 

Install `sh autoconf.sh` dependences and other dependences as follows,

.. code-block:: console

  $ pwd
  $ ~/git/iverilog
  $ sudo apt-get install autoconf automake autotools-dev curl python3 libmpc-dev \
  libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool \
  patchutils bc zlib1g-dev libexpat-dev
  $ sh autoconf.sh

Then install icarus as follows,

.. code-block:: console

  $ ./configure
  // or below if you are in shared server
  $ ./configure --prefix=$HOME/local
  $ make
  $ make check


Install other tools on Linux
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install cmake as follows,

.. code-block:: console

  $ pwd
  $ ~/local
  $ wget -b https://github.com/Kitware/CMake/releases/download/v3.23.3/cmake-3.23.3-linux-x86_64.sh
  $ bash cmake-3.23.3-linux-x86_64.sh
  Do you accept the license? [yn]: 
  y
  By default the CMake will be installed in:
    "/u/jonathanchen/local/cmake-3.23.3-linux-x86_64"
  Do you want to include the subdirectory cmake-3.23.3-linux-x86_64?
  Saying no will install in: "/u/jonathanchen/local" [Yn]: 
  Y
  ...
  Unpacking finished successfully
  $ ls
  bin  cmake-3.23.3-linux-x86_64 ...
  

Download Graphviz from [#graphviz-download]_ according your 
Linux distribution. Files compare tools Kdiff3 came from web site [#kdiff3]_. 

.. code-block:: console

  $ sudo apt install graphviz
  $ dot -V
  dot - graphviz version 2.40.1 (20161225.0304)

Set "~/.profile" as follows,

.. rubric:: ~/.profile
.. code-block:: text

   ~/.profile: executed by the command interpreter for login shells.
  ...
  # set PATH so it includes user's private bin if it exists
  if [ -d "$HOME/local/bin" ] ; then
      PATH="$HOME/local/bin:$PATH"
  fi
  # set PATH for cmake
  if [ -d "$HOME/local/cmake-3.23.3-linux-x86_64/bin" ] ; then
      PATH="$HOME/local/cmake-3.23.3-linux-x86_64/bin:$PATH"
  fi
  ...


.. [#llvm-cmake] http://llvm.org/docs/CMake.html?highlight=cmake

.. [#swapfile1] https://bogdancornianu.com/change-swap-size-in-ubuntu/

.. [#swapfile2] https://linuxize.com/post/how-to-add-swap-space-on-ubuntu-18-04/

.. [#installbrew] https://brew.sh/

.. [#kdiff3] http://kdiff3.sourceforge.net

.. [#graphviz-dm] https://graphviz.org/download/

.. [#isp] http://llvm.org/docs/CodeGenerator.html#selectiondag-instruction-selection-process

.. [#vgwdc] http://llvm.org/docs/ProgrammersManual.html#viewing-graphs-while-debugging-code

.. [#icarus] http://iverilog.icarus.com/

.. [#graphviz-download] http://www.graphviz.org/Download.php
