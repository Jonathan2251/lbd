The reStructuredText skill used in this book
=============================================

Build reStructuredText by command sphinx-build. 
Reference <http://sphinx-doc.org/.

Reference web:
http://docs.geoserver.org/latest/en/docguide/sphinx.html
http://docutils.sourceforge.net/docs/ref/rst/directives.html
http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html

Currently skills/setting use in this document

1. Font
  Bold **bold**
  Command ``cp -f source dest``

2. hyper link
  `Building LLVM with CMake`_
    .. _Building LLVM with CMake: http://llvm.org/docs/CMake.html?highlight=cmake

3. figure

  for example:
  
  Shown as as :ref:`_install_f18`.
  
  .. _install_f18: 
  .. figure:: Fig/install/18.png
    :height: 175 px
    :width: 1020 px
    :scale: 50 %
    :align: center
  
    Edit .profile and save .profile to /Users/Jonathan/

4. Code fragment and terminal io

  for example:

  .. code-block:: c++
  
    //  Cpu0ISelLowering.cpp
    ...
      // %hi/%lo relocation
      SDValue GAHi = DAG.getTargetGlobalAddress(GV, dl, MVT::i32, 0,
                            Cpu0II::MO_ABS_HI);
      SDValue GALo = DAG.getTargetGlobalAddress(GV, dl, MVT::i32, 0,
                            Cpu0II::MO_ABS_LO);
      SDValue HiPart = DAG.getNode(Cpu0ISD::Hi, dl, VTs, &GAHi, 1);
      SDValue Lo = DAG.getNode(Cpu0ISD::Lo, dl, MVT::i32, GALo);
      return DAG.getNode(ISD::ADD, dl, MVT::i32, HiPart, Lo);

  .. code-block:: bash
  
    118-165-16-22:InputFiles Jonathan$ /Users/Jonathan/llvm/3.1.test/cpu0/1/
    cmake_debug_build/bin/Debug/llc -march=cpu0 -debug -relocation-model=pic 
    -filetype=asm ch5_3.bc -o ch5_3.cpu0.s
    ...

5. Add footnote for web reference (It's easy to check the web reference by 
  group them in the end of each chapter).

  for example:
  [#]_ are the Chinese documents

  .. [#] http://ccckmit.wikidot.com/lk:aout

6. Use hyper link in reference our book section. But use "section name" of 
   `web link`_ to reference outside web section. Because I find the hyper link 
   for reference section of LLVM is changed from version to version.

7. For easy to verify the out of date reference. 
Put outside web reference at end of chapter with footnote [#].

8. Use :menuselection:`Start Menu --> Programs --> GeoServer`.


