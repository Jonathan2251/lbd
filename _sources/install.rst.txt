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
~/llvm/release/ which contains the clang and clang++ compiler we will use to 
translate the C/C++ input file into llvm IR. 
The other is the directory ~/llvm/test/ which contains our cpu0 backend 
program and clang.

Build steps
------------

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

Download the snapshot version of Icarus Verilog tool from web site, 
ftp://icarus.com/pub/eda/verilog/snapshots or go to http://iverilog.icarus.com/ 
and click snapshot version link. Follow the INSTALL file guide to install it. 


Install other tools on Linux
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Download Graphviz from [#graphviz-download]_ according your 
Linux distribution. Files compare tools Kdiff3 came from web site [#kdiff3]_. 

Set /home/Gamma/.bash_profile as follows,

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



.. [#llvm-cmake] http://llvm.org/docs/CMake.html?highlight=cmake

.. [#installbrew] https://brew.sh/

.. [#kdiff3] http://kdiff3.sourceforge.net

.. [#graphviz-dm] https://graphviz.org/download/

.. [#isp] http://llvm.org/docs/CodeGenerator.html#selectiondag-instruction-selection-process

.. [#vgwdc] http://llvm.org/docs/ProgrammersManual.html#viewing-graphs-while-debugging-code

.. [#graphviz-download] http://www.graphviz.org/Download.php
