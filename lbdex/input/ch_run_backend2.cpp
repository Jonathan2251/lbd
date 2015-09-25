// bash build-run_backend2.sh cpu032I le
// bash build-run_backend2.sh cpu032II be

/// start

#include "ch_nolld2.h"

int main()
{
  bool pass = true;
  pass = test_nolld2();

  return pass;
}

#include "ch_nolld2.cpp"
