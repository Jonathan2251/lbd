
/// start
// Not work at this point
void dump_mem(unsigned char *str, int n)
{
  int i = 0;
  const unsigned char *p;

  for (i = 0, p = str; i < n; i++, p++) {
    unsigned char x = (*p >> 4);
    if (x > 0x0f) print_string("Err");
    if (x <= 0x09)
      print_char(x+'0');
    else
      print_char(x+'a');
    x = (*p & 0x0f);
    if (x > 0x0f) print_string("Err");
    if (x <= 0x09)
      print_char(x+'0');
    else
      print_char(x+'a');
  }
  print_char('\n');

  return;
}
