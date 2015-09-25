// clang -target mips-unknown-linux-gnu -c memcpy.cpp -emit-llvm -o - | llvm-dis -o memcpy.1.ll
// Then modify memcpy.ll from memcpy.1.ll

/// start
void memcpy(char* dest, char* source, int size)
{
  return;
}

