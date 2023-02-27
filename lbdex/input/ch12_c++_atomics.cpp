// clang -S ch12_c++_atomics.cpp -emit-llvm -o -
// Uses /opt/homebrew/opt/llvm/bin/clang in macOS.

#include <atomic>

std::atomic<bool> winner (false);

int test_atomics() {
  int count = 0;
  bool res = winner.exchange(true);
  if (res) count++;

  return count;
}
