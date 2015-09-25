// bash build-run_backend.sh

/// start

#include "ch_nolld.h"

int main()
{
  bool pass = true;
  pass = test_nolld();

  return pass;
}

#include "ch_nolld.cpp"
