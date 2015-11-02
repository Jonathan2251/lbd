// clang -target mips-unknown-linux-gnu -c ch9_1_constructor.cpp -emit-llvm -o ch9_1_constructor.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch9_1_constructor.bc -o -

/// start
extern "C" int printf(const char *format, ...);

class Date_9_2_2
{
public:
  int year;
  int month;
  int day;
  int hour;
  int minute;
  int second;
public:
  Date_9_2_2(int year, int month, int day, int hour, int minute, int second);
  Date_9_2_2 getDate();
};

Date_9_2_2::Date_9_2_2(int year, int month, int day, int hour, int minute, int second)
{
  this->year = year;
  this->month = month;
  this->day = day;
  this->hour = hour;
  this->minute = minute;
  this->second = second;
}

Date_9_2_2 Date_9_2_2::getDate()
{ 
  return *this;
}

int test_constructor()
{
  Date_9_2_2 date1 = Date_9_2_2(2013, 1, 26, 12, 21, 10);
  Date_9_2_2 date2 = date1.getDate();
  if (!(date1.year == 2013 && date1.month == 1 && date1.day == 26 && date1.hour 
      == 12 && date1.minute == 21 && date1.second == 10))
    return 1;
  if (!(date2.year == 2013 && date2.month == 1 && date2.day == 26 && date2.hour 
      == 12 && date2.minute == 21 && date2.second == 10))
    return 1;

#ifdef PRINT_TEST
  printf("date1 = %d %d %d %d %d %d", date1.year, date1.month, date1.day,
    date1.hour, date1.minute, date1.second); // date1 = 2013 1 26 12 21 10
  if (date1.year == 2013 && date1.month == 1 && date1.day == 26 && date1.hour 
      == 12 && date1.minute == 21 && date1.second == 10)
    printf(", PASS\n");
  else
    printf(", FAIL\n");
  printf("date2 = %d %d %d %d %d %d", date2.year, date2.month, date2.day,
    date2.hour, date2.minute, date2.second); // date2 = 2013 1 26 12 21 10
  if (date2.year == 2013 && date2.month == 1 && date2.day == 26 && date2.hour 
      == 12 && date2.minute == 21 && date2.second == 10)
    printf(", PASS\n");
  else
    printf(", FAIL\n");
#endif

  return 0;
}
