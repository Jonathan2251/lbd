.. _sec-appendix-doc:

Appendix B: Cpu0 document and test
===================================

.. contents::
   :local:
   :depth: 4

github
------

Create gh-pages
***************

1. Create a new orphan branch for GitHub Pages and clean the directory.

..code-block:: console

  % git branch
  * gh-pages
  master

  % git checkout --orphan gh-pages
  % git rm -rf .

2. GitHub Pages build error

  - Cause: GitHub tries to build Jekyll by default
  - Fix: Add a .nojekyll file to the root of gh-pages:

..code-block:: console

  % git branch
  * gh-pages
  master

  % touch .nojekyll
  % git add .nojekyll
  % git commit -m "Disable Jekyll"
  % git push


reST format
-----------

The reSTructure format (Sphinx).

Six Levels of Chapter, section, subsections
*******************************************

.. rubric:: Six Levels of Chapter, section, subsections
.. code-block:: text

  1. Chapter -- Level 1
  ===============
  Level 1 Title
  ===============

  2. Section -- Level 2
  ---------------
  Level 2 Title
  ---------------

  3. Subsection -- Level3
  ***************
  Level 3 Title
  ***************

  4. Subsection -- Level4
  ^^^^^^^^^^^^^^^
  Level 4 Title
  ^^^^^^^^^^^^^^^

  5. Subsection -- Level5
  """""""""""""""
  Level 5 Title
  """""""""""""""

  6. Subsection -- Level6
  ~~~~~~~~~~~~~~~
  Level 6 Title
  ~~~~~~~~~~~~~~~



Cpu0 document
-------------

This section illustrates how to generate Cpu0 backend document.

Install sphinx
~~~~~~~~~~~~~~

LLVM and this book use Sphinx to generate HTML documents. This book uses
Sphinx to generate PDF and EPUB formats as well. See the installation guide
here [#sphinx-install]_.

Sphinx uses reStructuredText format; see [#rst-ref]_, [#rst-dir]_, and [#rst]_.
For code-block formatting in this document, see [#llvm-sphinx-quick]_ and
[#sphinx-lexers]_.

On iMac you can install as follows:

.. code-block:: console

  brew install sphinx-doc
  echo 'export PATH="/opt/homebrew/opt/sphinx-doc/bin:$PATH"' >> ~/.zshrc
  % source ~/.zshrc

On Linux, install with:

.. code-block:: console

  sudo apt-get install python3-sphinx

The above installs Sphinx for making HTML documents but not PDF.

For PDF/LaTeX generation, install the following.

On iMac, install MacTeX.pkg from here [#maxtex]_ and restart your computer.

On Linux, install texlive as follows:

.. code-block:: console

  sudo apt-get install texlive texlive-latex-extra latexmk

or

.. code-block:: console

  sudo yum install texlive texlive-latex-extra latexmk

On Fedora 17, the texlive-latex-extra package is missing. Instead, install
the package that includes pdflatex. For example, install pdfjam on Fedora 17
as follows,

.. code-block:: console

  [root@localhost lbd]$ yum list pdfjam
  Loaded plugins: langpacks, presto, refresh-packagekit
  Installed Packages
  pdfjam.noarch                        2.08-3.fc17                         @fedora
  [root@localhost lbd]$ 

Do

.. code-block:: console

  sudo apt-get update -y
  sudo apt-get install -y latexmk

if latexmk gives error during 'make latexpdf' [#latexmk]_.

After upgrading to iMac OS X 10.11.1, the pdflatex link may be missing. Fix it by
adding the following to your ``.profile``:

.. code-block:: console
  
  114-37-153-62:lbd Jonathan$ ls /usr/local/texlive/2012/bin/universal-darwin/pdflatex 
  /usr/local/texlive/2012/bin/universal-darwin/pdflatex
  114-37-153-62:lbd Jonathan$ cat ~/.profile
  export PATH=$PATH:...:/usr/local/texlive/2012/bin/universal-darwin


Note of Sphinx
~~~~~~~~~~~~~~

- Do not use more than 90 characters per line in ``code-block:: console`` or any
  other code block, otherwise errors may occur during `make latexpdf`.

- Use ``\clearpage`` (see gpu.rst) to force page breaks in pdf and avoid content
  splitting across two pages.

.. code-block:: console

  $HOME/git/lbd$ make latexpdf
  
  Latexmk: applying rule 'pdflatex'...
  Rule 'pdflatex': File changes, etc:
     Changed files, or newly in use since previous run(s):
    LLVMToolchainCpu0.aux
    LLVMToolchainCpu0.toc
  Rule 'pdflatex': The following rules & subrules became out-of-date:
    pdflatex
  Latexmk: Maximum runs of pdflatex reached without getting stable files
  Latexmk: All targets (LLVMToolchainCpu0.pdf) are up-to-date
  ----------------------
  This message may duplicate earlier message.
  Latexmk: Failure in processing file 'LLVMToolchainCpu0.tex':
     'pdflatex' needed too many passes
  ----------------------
  Latexmk: If appropriate, the -f option can be used to get latexmk
    to try to force complete processing.
  make[1]: *** [LLVMToolchainCpu0.pdf] Error 12
  make: *** [latexpdf] Error 2
  make latexpdf  14.57s user 1.11s system 99% cpu 15.790 total

- Sphinx supports math symbols; see references [#sphinx-math]_ and
  [#mathbase-latex]_ for more details.

Note of graphviz
~~~~~~~~~~~~~~~~

- To draw nodes inside a parent node, use "compound=true;" and subgraph
  cluster_name [#graphviz-cluster]_. To create edges from/to parent nodes,
  use the attributes "ltail" and "lhead" set to the parent node.

- See graphviz documents here [#gv-doc-pdf]_.

Generate Cpu0 document
~~~~~~~~~~~~~~~~~~~~~~~

The Cpu0 example code is added chapter by chapter.
It can be configured to a specific chapter by changing the CH definition in
Cpu0SetChapter.h.
For example, the following definition configures it to chapter 2.

.. rubric:: lbdex/Cpu0/Cpu0SetChapter.h
.. code-block:: c++

  #define CH       CH2

To help readers understand the backend structure step by step, the Cpu0 example
code can be generated chapter by chapter using the following commands:

.. code-block:: console

  118-165-12-177:lbd Jonathan$ pwd
  /home/Jonathan/test/lbd
  118-165-12-177:lbd Jonathan$ make genexample 
  ...
  118-165-12-177:lbd Jonathan$ ls lbdex/chapters/
  Chapter10_1  Chapter2    Chapter3_4  Chapter5_1  Chapter8_2
  Chapter11_1  Chapter3_1  Chapter3_5  Chapter6_1  Chapter9_1
  Chapter11_2  Chapter3_2  Chapter4_1  Chapter7_1  Chapter9_2
  Chapter12_1  Chapter3_3  Chapter4_2  Chapter8_1  Chapter9_3

Besides the example code in each chapter, the above HTML and PDF Cpu0 documents 
also include the \*.ll and \*.s files located in the lbd/lbdex/output directory.
  
.. code-block:: console

  JonathantekiiMac:lbd Jonathan$ ls lbdex/output/
  ch12_eh.cpu0.s			ch12_thread_var.cpu0.pic.s	ch12_thread_var.ll
  ch12_eh.ll			ch12_thread_var.cpu0.static.s	ch4_math.s
  
Then, the HTML and PDF versions of this book can be generated by running the
following commands.

.. code-block:: console

  118-165-12-177:lbd Jonathan$ pwd
  /home/Jonathan/test/lbd
  118-165-12-177:lbd Jonathan$ make html
  ...
  118-165-12-177:lbd Jonathan$ make latexpdf
  ...


About Cpu0 Document
~~~~~~~~~~~~~~~~~~~

Since LLVM has a new release about every 6 months, and names of files, 
functions, classes, variables, etc. may change, maintaining the Cpu0 document 
is a continuous effort. The document adds code chapter by chapter.

To keep the document correct and easy to maintain, I use the ``:start-after:``
and ``:end-before:`` directives of reStructuredText to keep the document up to 
date.

For every new release, when the Cpu0 backend code changes, the document will 
reflect those changes in most of its content.

In the ``lbdex/Cpu0`` folder, text beginning with ``//\@`` and ``#ifdef CH > CHxx``
is referenced by document files ``*.rst``.

In ``lbdex/llvm/modify/llvm``, the ``*.rst`` files reference the code by copying
it directly. Most references exist in ``llvmstructure.rst`` and ``elf.rst``.

The example C/C++ code in ``lbdex/input`` comes from my own ideas and refers to
the ``clang/test/CodeGen`` directory in the Clang source code release.

Cpu0 Regression Test
--------------------

The last chapter can verify the Cpu0 backend's generated code by using a Verilog 
simulator for code without global variable access.

The chapter *lld* in the repository https://github.com/Jonathan2251/lbt.git includes
an LLVM ELF linker implementation and can verify test items involving global variable access.

However, LLVM provides its own test cases (regression tests) for each backend to verify
the backend compiler [#test]_ without needing a simulator or real hardware platform.

Cpu0 regression test items exist in the lbdex.tar.gz example code. Untar it into the
lbdex/ directory.

For both iMac and Linux, copy the folder:

:: 

  lbdex/regression-test/Cpu0

To

:: 

  ~/llvm/test/llvm/test/CodeGen/Cpu0

Then run the tests as follows on iMac, for both single and all test cases:

.. code-block:: console

  $ llvm-lit -a ~/llvm/test/llvm/test/CodeGen/Cpu0

The option **-a** shows the executing commands for each test case.

.. code-block:: console

  1-160-130-77:Cpu0 Jonathan$ pwd
  /Users/Jonathan/llvm/test/llvm/test/CodeGen/Cpu0
  1-160-130-77:Cpu0 Jonathan$ ~/llvm/test/build/bin/llvm-lit seteq.ll
  -- Testing: 1 tests, 1 threads --
  PASS: LLVM :: CodeGen/Cpu0/seteq.ll (1 of 1)
  Testing Time: 0.08s
    Expected Passes    : 1
  1-160-130-77:Cpu0 Jonathan$ ~/llvm/test/build/bin/llvm-lit .
  ...
  PASS: LLVM :: CodeGen/Cpu0/zeroreg.ll
  PASS: LLVM :: CodeGen/Cpu0/tailcall.ll
  ...


Run the following command to execute the test on Linux.

.. code-block:: console

  $ pwd
  /home/cschen/llvm/test/llvm/test/CodeGen/Cpu0
  $ ~/llvm/test/build/bin/llvm-lit seteq.ll
  -- Testing: 1 tests, 1 threads --
  PASS: LLVM :: CodeGen/Cpu0/seteq.ll (1 of 1)
  Testing Time: 0.08s
    Expected Passes    : 1
  $ ~/llvm/test/build/bin/llvm-lit .
  ...
  PASS: LLVM :: CodeGen/Cpu0/zeroreg.ll
  PASS: LLVM :: CodeGen/Cpu0/tailcall.ll
  ...

The chapters of this book and their related regression test items are listed 
as follows:

.. table:: Chapters

  ==== ==================
  1    about
  2    Cpu0 architecture and LLVM structure
  3    Backend structure
  4    Arithmetic and logic instructions
  5    Generating object files
  6    Global variables
  7    Other data type
  8    Control flow statements
  9    Function call
  10   ELF Support
  11   Assembler
  12   C++ support
  13   Verify backend on verilog simulator
  ==== ==================

.. table:: Regression test items for Cpu0

  ===============================  =============  =======================================================  ===========
  File                             v:pass x:fail  test ir, -> output asm                                   chapter
  ===============================  =============  =======================================================  ===========
  2008-06-05-Carry.ll              v                                                                       7
  2008-07-15-InternalConstant.ll   v                                                                       6
  2008-07-15-SmallSection.ll       v                                                                       6
  2008-07-03-SRet.ll               v                                                                       9
  2008-07-29-icmp.ll               v                                                                       8
  2008-08-06-Alloca.ll             v                                                                       9
  2008-08-01-AsmInline.ll          v                                                                       11
  2008-08-08-ctlz.ll               v                                                                       7
  2008-08-08-bswap.ll              v              bswap                                                    12
  2008-10-13-LegalizerBug.ll       v                                                                       8
  2010-11-09-Mul.ll                v                                                                       4                         
  2010-11-09-CountLeading.ll       v                                                                       7
  2008-11-10-xint_to_fp.ll         v                                                                       7
  addc.ll                          v              64-bit add                                               7
  addi.ll                          v              32-bit add, sub                                          4
  address-mode.ll                  v              br, -> BB0_2:                                            8
  alloca.ll                        v              alloca i8, i32 %size, dynamic allocation                 9
  analyzebranch.ll                 v              br, -> bne, beq                                          8
  and1.ll                          v              and                                                      4
  asm-large-immediate.ll           v              inline asm                                               11
  atomic-1.ll                      v              atomic                                                   12
  atomic-2.ll                      v              atomic                                                   12
  atomics.ll                       v              atomic                                                   12
  atomics-index.ll                 v              atomic                                                   12
  atomics-fence.ll                 v              atomic                                                   12
  br-jmp.ll                        v              br, -> jmp                                               8
  brockaddress.ll                  v              blockaddress, -> lui, ori                                8
  cmov.ll                          v              select, -> movn, movz                                    8
  cprestore.ll                     v              -> .cprestore                                            9
  div.ll                           v              sdiv, -> div, mflo                                       4
  divrem.ll                        v              sdiv, srem, udiv, urem, -> div, divu                     4
  div_rem.ll                       v              sdiv, srem, -> div, mflo, mfhi                           4
  divu.ll                          v              udiv, -> divu, mflo                                      4
  divu_reml.ll                     v              udiv, urem -> div, mflo, mfhi                            4
  double2int.ll                    v              double to int, -> %call16(__fixdfsi)                     7
  eh-dwraf-cfa.ll                  v                                                                       9
  eh-return32.ll                   v              Spill and reload all registers used for exception        9 
  eh.ll                            v              c++ exception handling                                   12
  ex2.ll                           v              c++ exception handling                                   12
  fastcc.ll                        v              No effect in fastcc but can pass                         9
  fneg.ll                          v              verify Cpu0 don't uses hard float instruction            7
  fp-spill-reload.ll               v              -> st $fp, ld $fp                                        9
  frame-address.ll                 v              addu $2, $zero, $fp                                      9
  global-address.ll                v              global address, global variable                          6
  global-pointer.ll                v              global register load and retore, -> .cpload, .cprestore  9
  gprestore.ll                     v              global register retore, -> .cprestore                    9
  helloworld.ll                    v              global register load and retore, -> .cpload, .cprestore  9
  hf16_1.ll                        v              function call in PIC, -> ld, jalr                        9
  i32k.ll                          v              argument of constant int passing in register             9 
  i64arg.ll                        v              argument of constant 64-bit passing in register          9 
  imm.ll                           v              return constant 32-bit in register                       9 
  indirectcall.ll                  v              indirect function call                                   9
  init-array.ll                    v              check .init                                              6
  inlineasm_constraint.ll          v              inline asm                                               11
  inlineasm-cnstrnt-reg.ll         v              -                                                        11
  inlineasmmemop.ll                v              -                                                        11
  inlineasm-operand-code.ll        v              -                                                        11
  internalfunc.ll                  v              internal function                                        9
  jstat.ll                         v              switch, -> JTI                                           8
  lb1.ll                           v              load i8*, sext i8, -> lb                                 7
  lbu1.ll                          v              load i8*, zext i8, -> lbu                                7
  lh1.ll                           v              load i16*, sext i16, -> lh                               7
  lhu1.ll                          v              load i16*, zext i16, -> lhu                              7
  llcarry.ll                       v              64-bit add sub                                           7
  longbranch.ll                    v                                                                       8
  machineverifier.ll               v              delay slot, (comment in machineverifier.ll)              8
  mipslopat.ll                     v              no check output (comment in mipslopat.ll)                6
  misha.ll                         v              miss alignment half word access                          7
  module-asm.ll                    v              module asm                                               11
  module-asm-cpu032II.ll           v              module asm                                               11
  mul.ll                           v              mul                                                      4
  mulll.ll                         v              64-bit mul                                               4
  mulull.ll                        v              64-bit mul                                               4
  not1.ll                          v              not 1                                                    4
  null.ll                          v              ret i32 0, -> ret $lr                                    3
  o32_cc_byval.ll                  v              by value                                                 9
  o32_cc_vararg.ll                 v              variable argument                                        9
  private.ll                       v              private function call                                    9
  rem.ll                           v              srem, -> div, mfhi                                       4
  remat-immed-load.ll              v              immediate load                                           3
  remul.ll                         v              urem, -> div, mfhi                                       4
  return-vector-float4.ll          v              return vector, -> lui lui ...                            3
  return-vector.ll                 v              return vector, -> ld ld ..., st st ...                   3
  return_address.ll                v              llvm.returnaddress, -> addu $2, $zero, $lr               9
  rotate.ll                        v              rotl, rotr, -> rolv, rol, rorv                           4
  sb1.ll                           v              store i8, sb                                             7
  select.ll                        v              select, -> movn, movz                                    8
  seleq.ll                         v              following for br with different condition                8
  seleqk.ll                        v              -                                                        8
  selgek.ll                        v              -                                                        8
  selgt.ll                         v              -                                                        8
  selle.ll                         v              -                                                        8
  selltk.ll                        v              -                                                        8
  selne.ll                         v              -                                                        8
  selnek.ll                        v              -                                                        8
  seteq.ll                         v              -                                                        8
  seteqz.ll                        v              -                                                        8
  setge.ll                         v              -                                                        8
  setgek.ll                        v              -                                                        8
  setle.ll                         v              -                                                        8
  setlt.ll                         v              -                                                        8
  setltk.ll                        v              -                                                        8
  setne.ll                         v              -                                                        8
  setuge.ll                        v              -                                                        8
  setugt.ll                        v              -                                                        8
  setule.ll                        v              -                                                        8
  setult.ll                        v              -                                                        8
  setultk.ll                       v              -                                                        8
  sext_inreg.ll                    v              sext i1, -> shl, sra                                     4
  shift-parts.ll                   v              64-bit shl, lshr, ashr, -> call function                 9
  shl1.ll                          v              shl, -> shl                                              4
  shl2.ll                          v              shl, -> shlv                                             4
  shr1.ll                          v              shr, -> shr                                              4
  shr2.ll                          v              shr, -> shrv                                             4
  sitofp-selectcc-opt.ll           v              comment in sitofp-selectcc-opt.ll                        7
  small-section-reserve-gp.ll      v              Cpu0 option -cpu0-use-small-section=true                 6
  sra1.ll                          v              ashr, -> sra                                             4
  sra2.ll                          v              ashr, -> srav                                            4
  stacksave-restore.ll             v                                                                       9
  stacksize.ll                     v              comment in stacksize.ll                                  9
  stchar.ll                        v              load and store i16, i8                                   7
  stldst.ll                        v              register sp spill                                        9
  sub1.ll                          v              sub, -> addiu                                            4
  sub2.ll                          v              sub, -> sub                                              4
  tailcall.ll                      v              tail call                                                9
  tls.ll                           v              ir thread_local global is for c++ "__thread int b;"      12
  tls-alias.ll                     v              thread_local global and thread local alias               12
  tls-models.ll                    v              ir external/internal thread_local global                 12
  uitofp.ll                        v              integer2float, uitofp, -> jsub __floatunsisf             9
  uli.ll                           v              unalignment init, -> sb sb ...                           6
  unalignedload.ll                 v              unalignment init, -> sb sb ...                           6
  vector-setcc.ll                  v                                                                       7
  weak.ll                          v              extern_weak function, -> .weak                           9
  xor1.ll                          v              xor, -> xor                                              4
  zeroreg.ll                       v              check register $zero                                     4
  ===============================  =============  =======================================================  ===========
  

These supported test cases are located in ``lbdex/regression-test/Cpu0``, which 
can be extracted from ``tar -xf lbdex.tar.gz``.

The regression test is useful for two major reasons. First, it provides the LLVM 
input, assembly output, and the corresponding command and options within the same 
sample input file. This makes it a well-documented reference for both end users 
and developers.

Second, when developers make changes to their backend compiler—especially 
for optimization—these tests help detect side effects or bugs caused by the 
modifications. This is the core purpose of "regression testing."

The following file includes the assembly output patterns for two subtargets 
of the Cpu0 backend. In addition to checking opcodes, it can also verify 
register numbers. For example, the destination register of the "andi" instruction 
must match the first source register of the following "xori" instruction. This is 
ensured when both are specified as register T1 in the corresponding assembly output.

.. rubric:: lbdex/regression-test/Cpu0/setule.ll
.. literalinclude:: ../lbdex/regression-test/Cpu0/setule.ll

Running regression tests must occur after building LLVM. However, the following 
README.rst and changes in related config files allow you to set up `llvm-lit` 
for a pre-built or pre-installed LLVM environment.

This setup enables running `llvm-lit` without rebuilding LLVM, making the 
regression testing more efficient when testing changes to the backend only.

.. rubric:: lbdex/set-llvm-lit/README.txt
.. literalinclude:: ../lbdex/set-llvm-lit/README.txt

set-llvm-lit % `diff -r origin modify &> set-llvm-lit.diff`

.. rubric:: lbdex/set-llvm-lit/set-llvm-lit.diff
.. literalinclude:: ../lbdex/set-llvm-lit/set-llvm-lit.diff

- Only `tools/clang/test/lit.site.cfg.py` and `test/lit.site.cfg.py` need to be
  modified.

- The other files, `tools/clang/test/Unit/lit.site.cfg.py`,
  `test/Unit/lit.site.cfg.py`, and `utils/lit/tests/lit.site.cfg.in`, are empty
  and not used. However, I modified them as well for completeness.

.. [#sphinx-install] https://www.sphinx-doc.org/en/master/usage/installation.html

.. [#maxtex] http://www.tug.org/mactex/

.. [#latexmk] https://zoomadmin.com/HowToInstall/UbuntuPackage/latexmk

.. [#sphinx-math] https://sphinx-rtd-trial.readthedocs.io/en/latest/ext/math.html#module-sphinx.ext.mathbase

.. [#mathbase-latex] https://mirrors.mit.edu/CTAN/info/short-math-guide/short-math-guide.pdf

.. [#gv-doc-pdf] https://www.graphviz.org/pdf/dotguide.pdf

.. [#graphviz-cluster] Ex. lbd/Fig/gpu/opengl-flow.gv. If the name of the subgraph begins with cluster, Graphviz notes the subgraph as a special cluster subgraph. If supported, the layout engine will do the layout so that the nodes belonging to the cluster are drawn together, with the entire drawing of the cluster contained within a bounding rectangle. https://graphviz.org/doc/info/lang.html. 

.. [#rst-ref] http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html

.. [#rst-dir] http://docutils.sourceforge.net/docs/ref/rst/directives.html

.. [#rst] http://docutils.sourceforge.net/rst.html

.. [#llvm-sphinx-quick] http://llvm.org/docs/SphinxQuickstartTemplate.html If you need to show LLVM IR use the llvm code block. https://llvm.org/docs/SphinxQuickstartTemplate.html#code-blocks

.. [#sphinx-lexers] http://pygments.org/docs/lexers/

.. [#test] http://llvm.org/docs/TestingGuide.html

