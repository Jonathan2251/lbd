.. _sec-appendix-installing:

Appendix A: Getting Started: Installing LLVM and the Cpu0 Example Code
======================================================================

.. contents::
   :local:
   :depth: 4

The Cpu0 example code, `lbdex`, can be found at the lower left section of this
website, or directly via this link:
http://jonathan2251.github.io/lbd/lbdex.tar.gz.

For details on using ``cmake`` to build LLVM, refer to "Building LLVM with 
CMake" [#llvm-cmake]_ documentation.

We will install two LLVM directories in this chapter. One is the directory 
``~/llvm/debug/``, which contains the `clang` and `clang++` compilers used to 
translate C/C++ source files into LLVM IR. The other is ``~/llvm/test/``, which 
contains our Cpu0 backend program and another `clang` build.

Build Steps
-----------

On Linux, using multi-threading (`-DLLVM_PARALLEL_COMPILE_JOBS=4`) requires more 
than 16GB of memory. I created a 64GB swap file to avoid link failures 
[#swapfile1]_ [#swapfile2]_. iMac systems typically do not encounter this issue.

.. code-block:: console

  $ cat /etc/fstab
  # <file system> <mount point>   <type>  <options>       <dump>  <pass>
  ...
  /swapfile       swap            swap    default         0       0

After installing necessary packages (via `brew` on iMac), follow the build 
instructions here:
https://github.com/Jonathan2251/lbd/blob/master/README.md.

Setting Up Your Mac
-------------------

Install Homebrew first [#installbrew]_:

.. code-block:: console

  % /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

After installation, add the `brew` command to your shell's PATH, as shown in the 
final message of the install script. The command typically looks like the 
following:

.. code-block:: console

  % echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /Users/cschen/.zprofile
  % eval "$(/opt/homebrew/bin/brew shellenv)"

For installing Homebrew in China, use the following install script instead
[#installbrew-china]_.

.. code-block:: console

  % /bin/zsh -c "$(curl -fsSL https://gitee.com/cunkai/HomebrewCN/raw/master/Homebrew.sh)"
  ...
  % source /Users/cschen/.zprofile
  % brew --version
  Homebrew 3.6.7-28-g560f571
  fatal: detected dubious ownership in repository at '/usr/local/Homebrew/Library/Taps/homebrew/homebrew-core'
  To add an exception for this directory, call:
  
          git config --global --add safe.directory /usr/local/Homebrew/Library/Taps/homebrew/homebrew-core
  Homebrew/homebrew-core (no Git repository)
  fatal: detected dubious ownership in repository at '/usr/local/Homebrew/Library/Taps/homebrew/homebrew-cask'
  To add an exception for this directory, call:

	  git config --global --add safe.directory /usr/local/Homebrew/Library/Taps/homebrew/homebrew-cask

  % git config --global --add safe.directory /usr/local/Homebrew/Library/Taps/homebrew/homebrew-core
  % git config --global --add safe.directory /usr/local/Homebrew/Library/Taps/homebrew/homebrew-cask
  % brew install cmake
  ...
  ==> Running `brew cleanup cmake`...
  Disable this behaviour by setting HOMEBREW_NO_INSTALL_CLEANUP.
  Hide these hints with HOMEBREW_NO_ENV_HINTS (see `man brew`).

  % brew install ninja



Install Icarus Verilog Tool on iMac
***********************************

Install Icarus Verilog using the command ``brew install icarus-verilog`` as
shown below:

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


.. _install-other-tools-on-imac:

Install Other Tools on iMac
***************************

Install CMake and Ninja with the following command:

.. code-block:: console

  brew install cmake ninja

Install Graphviz for displaying LLVM IR nodes during debugging [#graphviz-dm]_.

.. code-block:: console

  % brew install graphviz

Information about using Graphviz with LLVM is available in the section
"SelectionDAG Instruction Selection Process" of "The LLVM Target-Independent
Code Generator" [#isp]_, and in the section "Viewing graphs while debugging
code" of the "LLVM Programmer’s Manual" [#vgwdc]_.

Install binutils with the following command:

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
************************************

Download Icarus Verilog as follows [#icarus]_.

.. code-block:: console

  $ git clone http://iverilog.icarus.com/


Follow the README or INSTALL file guide to install it.

Install `sh autoconf.sh` dependencies and other dependencies as follows,

.. code-block:: console

  $ pwd
  $ ~/git/iverilog
  $ sudo apt-get install autoconf automake autotools-dev curl python3 libmpc-dev \
  libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool \
  patchutils bc zlib1g-dev libexpat-dev
  $ sh autoconf.sh

Then install Icarus Verilog using the following commands,

.. code-block:: console

  $ ./configure
  // or below if you are in shared server
  $ ./configure --prefix=$HOME/local
  $ make
  $ make check


Install Other Tools on Linux
****************************

Install CMake and Ninja as follows,

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

  $ sudo apt install ninja-build 
  

Download Graphviz from [#graphviz-download]_ according to your Linux
distribution. The file comparison tool KDiff3 can be downloaded from the
website [#kdiff3]_.

.. code-block:: console

  $ sudo apt install graphviz
  $ dot -V
  dot - graphviz version 2.40.1 (20161225.0304)

Set ``~/.profile`` as follows,

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


Toolchain
---------

List some GNU and LLVM tools as follows,

.. code-block:: console

  // Linux
  ~/git/lbd/lbdex/input$ ~/llvm/debug/build/bin/clang -fpic hello.c
  ~/git/lbd/lbdex/input$ man ldd
  ldd - print shared object dependencies
  ~/git/lbd/lbdex/input$ ldd a.out
	linux-vdso.so.1 (0x00007fffd1fe5000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f2c92a82000)
	/lib64/ld-linux-x86-64.so.2 (0x00007f2c92e73000)

  // MacOS
  % man otool
  otool-classic - object file displaying tool
    ...
    -L     Display the names and version numbers of the shared libraries that 
           the object file uses, as well as the shared library ID if the file 
           is a shared library.
  % otool -L a.out   
  a.out:
  	/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current 
        version 1292.100.5)

  // Linux
  ~/git/lbd/lbdex/input$ man objcopy
  objcopy - copy and translate object files
    ...
    [-O bfdname|--output-target=bfdname]
  ~/git/lbd/lbdex/input$ objcopy -O verilog a.out a.hex
  ~/git/lbd/lbdex/input$ vi a.hex
  @00400238
  2F 6C 69 62 36 34 2F 6C 64 2D 6C 69 6E 75 78 2D
  78 38 36 2D 36 34 2E 73 6F 2E 32 00
  @00400254
  04 00 00 00 10 00 00 00 01 00 00 00 47 4E 55 00
  ...



.. [#llvm-cmake] http://llvm.org/docs/CMake.html?highlight=cmake

.. [#swapfile1] https://bogdancornianu.com/change-swap-size-in-ubuntu/

.. [#swapfile2] https://linuxize.com/post/how-to-add-swap-space-on-ubuntu-18-04/

.. [#installbrew] https://brew.sh/

.. [#installbrew-china] https://blog.csdn.net/weixin_45571585/article/details/126977413

.. [#kdiff3] http://kdiff3.sourceforge.net

.. [#graphviz-dm] https://graphviz.org/download/

.. [#isp] http://llvm.org/docs/CodeGenerator.html#selectiondag-instruction-selection-process

.. [#vgwdc] http://llvm.org/docs/ProgrammersManual.html#viewing-graphs-while-debugging-code

.. [#icarus] http://iverilog.icarus.com/

.. [#graphviz-download] http://www.graphviz.org/Download.php
