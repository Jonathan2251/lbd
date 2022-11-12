llvm-lit can be changed to support the installed clang/llvm. Then llvm-lit can
be run directly without build the clang/llvm.

## Run llvm-lit without building llvm-project

    $ pwd
    $ $HOME/test/llvm/llvm/test/CodeGen/Cpu0
    // change the following to dir for your llvm-project
    $ export LLVM_DIR=$HOME/test/llvm
    $ export LLVM_INSTALLED_DIR=$HOME/llvm-installed
    $ ~/llvm-installed/bin/llvm-lit addc.ll -a
    $ ~/llvm-installed/bin/llvm-lit .
    ~/test/llvm/clang/test/CodeGen/Cpu0$ ~/riscv/riscv_newlib/bin/llvm-lit . -a


