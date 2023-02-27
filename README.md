lbd: llvm backend document
===========================

This document aims to provide a simple, concise, and clear step-by-step 
tutorial in creating a new LLVM backend from scratch. 
It is written in reStructuredText, and built using the Sphinx Python 
Documentation Generator.

If you would like to view an up to date version of this book in your 
browser without checking out and building the book, please visit: 
http://jonathan2251.github.io/lbd/index.html

Linux prerequisite:
Add swapfile for Linux: http://jonathan2251.github.io/lbd/install.html#build-steps

BUILD steps:

$ pwd

$ $HOME  // HOME directory

$ mkdir git

$ cd git

$ pwd

$ $HOME/git

$ git clone https://github.com/Jonathan2251/lbd

$ cd lbd/lbdex/install_llvm

$ bash build-llvm.sh

...

Please remember to add "${LLVM_RELEASE_DIR}/bin" to variable "${PATH}" to your
environment for clang++, clang. Reference last line of 
lbd/lbdex/install_llvm/build-llvm.sh

$ cd ..

$ pwd

$ $HOME/git/lbd/lbdex

$ bash build-cpu0.sh

CHECK step:

$ pwd

$ $HOME/git/lbd

$ bash check.sh
