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

Build steps
------------

After setup brew install for iMac or install necessory packages. Build as 
https://github.com/Jonathan2251/lbd/blob/master/README.md.


Setting Up Your Mac
-------------------

Brew install and setup first.

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


Setup path
~~~~~~~~~~~

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
