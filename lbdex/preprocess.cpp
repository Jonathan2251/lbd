#include "fstream"
#include "iostream"
#include "stack"
#include "cstdio"
#include "string.h"

//#define DEBUG

using namespace std;

enum FileType { CPP_FILE=0, CMAKE_FILE, LLVMBUILD_FILE, TD_FILE };
enum IfType {IF_CH, IF_OTHER};
enum {SUCCESS, FAIL};

struct chapterIdMap {
  char symbol[25];
  int id;
};

struct IfCtrType {
  IfType type;
  bool hidden;
};

const struct chapterIdMap ch[] = {
  {"CH2", 20}, {"CH3_1", 31}, {"CH3_2", 32}, {"CH3_3", 33}, {"CH3_4", 34}, 
  {"CH3_5", 35}, {"CH4_1", 41}, {"CH4_2", 42}, {"CH5_1", 51}, {"CH6_1", 61}, 
  {"CH7_1", 71}, {"CH8_1", 81}, {"CH8_2", 82}, {"CH9_1", 91}, {"CH9_2", 92}, 
  {"CH9_3", 93}, {"CH9_4", 94}, {"CH10_1", 101}, {"CH11_1", 111},
  {"CH11_2", 112}, {"CH12_1", 141}, {"CHEND", -1}
};

class Preprocess {
private:
  char PatternIf[25];
  char PatternIfSpace[25];
  char PatternElse[25];
  char PatternEndif[25];
  int skipUntilmatchedEndifOrElse(ifstream& in, ofstream& out, char* line);
  bool inCondition(char* CH, char* compareCondition, char* CHXX);
  
public:
  void setFileType(FileType fileType);
  int currentChapterId;
  int getChapterId(char* chapter);
  int run(ifstream& in, ofstream& out);
  void removeSuccessiveEmptyLines(ifstream& in, ofstream& out);
};

void Preprocess::setFileType(FileType fileType) {
  if (fileType == CPP_FILE || fileType == CMAKE_FILE || fileType == 
      LLVMBUILD_FILE) {
     strcpy(PatternIf, "#if");
     strcpy(PatternIfSpace, "#if ");
     strcpy(PatternElse, "#else");
     strcpy(PatternEndif, "#endif");
  }
  else if (fileType == TD_FILE) {
     strcpy(PatternIf, "//#if");
     strcpy(PatternIfSpace, "//#if ");
     strcpy(PatternElse, "//#else");
     strcpy(PatternEndif, "//#endif");
  }
}

int Preprocess::getChapterId(char* chapter) {
  for (int i = 0; ch[i].id != -1; i++) {
    if (strcmp(ch[i].symbol, chapter) == 0)
	  return ch[i].id;
  }
  return -1;
}

int Preprocess::skipUntilmatchedEndifOrElse(ifstream& in, ofstream& out, 
                                            char* line) {
  int result = SUCCESS;
  int ifNestCount = 0;
#ifdef DEBUG
  cout << "ifNestCount: " << ifNestCount << "in.eof(): " << in.eof() << endl;
  cout << "result: " << result << endl;
#endif

  while (ifNestCount >= 0 && !in.eof()) {
    in.getline(line, 255);
    if (strncmp(line, PatternIfSpace, strlen(PatternIfSpace)) == 0) {
      ifNestCount++;
    }
    else if (strncmp(line, PatternIf, strlen(PatternIf)) == 0) {
      ifNestCount++;
    }
    else if (strncmp(line, PatternElse, strlen(PatternElse)) == 0) {
      if (ifNestCount == 0)
        return result;
    }
    else if (strncmp(line, PatternEndif, strlen(PatternEndif)) == 0) {
      ifNestCount--;
    }
  }
  if (in.eof() && ifNestCount > 0)
    result = FAIL;
#ifdef DEBUG
  cout << "ifNestCount: " << ifNestCount << "in.eof(): " << in.eof() << endl;
  cout << "result: " << result << endl;
#endif

  return result;
}

bool Preprocess::inCondition(char* CH, char* compareCondition, 
                    char* CHXX) {
#ifdef DEBUG
  printf("CH: %s compareCond: %s CHXX: %s currentChapterId: %d, getChapterId(CHXX): %d\n", 
         CH, compareCondition, CHXX, currentChapterId, getChapterId(CHXX));
#endif
  if (strcmp(CH, "CH") != 0)
    return false;
  if (strcmp(compareCondition, ">=") != 0)
    return false;
  if (currentChapterId >= getChapterId(CHXX)) {
    return true;
  }
  else {
    return false;
  }
}

// expand chapter added code through directive (eg. #if CH > CH3_1)
// for example, #define CH CH9_1:
// 1:  #if CH >= CH3_2
// 2:    addRegisterClass();
// 3:  #if 0
// 4:    setBooleanContents();
// 5:  #endif
// 6:  #if CH >= CH8_1
// 7:    unsigned Opc = MI->getOpcode();
// 8:  #else
// 9:    unsigned LO = Cpu0::LO;
// 10: #endif
// 11:   SmallString<128> Str;
// 12: #endif
//
// translate into:
// 1:    addRegisterClass();
// 2:  #if 0
// 3:    setBooleanContents();
// 4:  #endif
// 5:    unsigned Opc = MI->getOpcode();
// 6:    SmallString<128> Str;

// #if #endif pair processing belong to push automata (not finite state machine),
// So, solve this problem by stack.

// When read at line 1 (after process line 1), the output and stack content as follows:
// output: nothing // since (inCondition(CH, compareCond, CHXX))
// (IFCH, false)

// When read at line 2 (after process line ), output as follows:
// output: addRegisterClass(); // since !(stack.top().type == IF_CH && stack.top().hidden) == true

// When read at line 3, the output and stack content as follows:
// (IFOTHER, false)
// (IFCH, false)
// output: #if 0 // since strncmp(line, PatternIf, strlen(PatternIf)) == 0

// When read at line 4, output as follows:
// output: setBooleanContents(); // since !(stack.top().type == IF_CH && stack.top().hidden) == true

// When read at line 5, the output and stack content as follows:
// output: #endif // since (strncmp(line, PatternEndif, strlen(PatternEndif)) == 0) && (stack.top().type == IF_OTHER)
// (IFCH, false)

// When read at line 6, the output and stack content as follows:
// output: nothing // since (inCondition(CH, compareCond, CHXX))
// (IFCH, false)
// (IFCH, false)

// When read at line 7, output as follows:
// output: unsigned Opc = MI->getOpcode(); // since !(stack.top().type == IF_CH && stack.top().hidden) == true

// When read at line 8, the output and stack content as follows:
// output: nothing // since (strncmp(line, PatternElse, strlen(PatternElse)) == 0) && (stack.top().type == IF_CH)
// (IFCH, true)
// (IFCH, false)

// When read at line 9, output as follows:
// output: nothing // since !(stack.top().type == IF_CH && stack.top().hidden) == false

// When read at line 10, the stack content as follows:
// (IFCH, false)

int Preprocess::run(ifstream& in, ofstream& out) {
  char line[256];
  char If[20];
  char CH[20];
  char compareCond[20];
  char CHXX[20];
  IfCtrType aa;
  stack<IfCtrType> stack;
  
  // When IfCtrType.hidden == true, meaning don't output statement.

  while (!in.eof()) {
    in.getline(line, 255);
  #ifdef DEBUG
    printf("stack.size(): %lu\n", stack.size());
    printf("%s\n", line);
  #endif
    if (strncmp(line, PatternIfSpace, strlen(PatternIfSpace)) == 0) {
      sscanf(line, "%s %s", If, CH);
      if (strcmp(CH, "CH") == 0) {
      // #if CH ... (cpu0 chapter control pattern).
        sscanf(line, "%s %s %s %s", If, CH, compareCond, CHXX);
      #ifdef DEBUG
        printf("if: %s CH: %s compareCond: %s CHXX: %s\n", If, CH, compareCond, CHXX);
      #endif
        if (inCondition(CH, compareCond, CHXX)) {
          aa.type = IF_CH;
          aa.hidden = false;
          stack.push(aa);
        }
        else {
          if (skipUntilmatchedEndifOrElse(in, out, line) == SUCCESS) {
            if (strncmp(line, PatternElse, strlen(PatternElse)) == 0) {
              aa.type = IF_CH;
              aa.hidden = false;
              stack.push(aa);
            }
            else if (strncmp(line, PatternEndif, strlen(PatternEndif)) == 0) {
              continue;
            }
            else
              return FAIL;
          }
          else
            return FAIL;
        }
      }
      else {
      // #if ... (other #if pattern)
        out << line << endl;
        aa.type = IF_OTHER;
        aa.hidden = false;
        stack.push(aa);
      }
    }
    else if (strncmp(line, PatternIf, strlen(PatternIf)) == 0) {
    // #if ... (other #if pattern)
      out << line << endl;
      aa.type = IF_OTHER;
      aa.hidden = false;
      stack.push(aa);
    }
    else if (strncmp(line, PatternElse, strlen(PatternElse)) == 0) {
      if (stack.empty())
        return FAIL;
      if (stack.top().type == IF_OTHER)
        out << line << endl;
      else if (stack.top().type == IF_CH) {
        aa = stack.top();
        stack.pop();
        aa.type = IF_CH;
        aa.hidden = !aa.hidden;
        stack.push(aa);
      }
    }
    else if (strncmp(line, PatternEndif, strlen(PatternEndif)) == 0) {
      if (stack.empty())
        return FAIL;
      if (stack.top().type == IF_OTHER) {
        out << line << endl;
      }
      stack.pop();
    }
    else {
      if (stack.empty())
        out << line << endl;
      else if ( !(stack.top().type == IF_CH && stack.top().hidden))
        out << line << endl;
    }
  } // while

  return SUCCESS;
}

void Preprocess::removeSuccessiveEmptyLines(ifstream& in, ofstream& out) {
  char line[256];
  bool lastLineIsEmpty = false;
  
  while (!in.eof()) {
    in.getline(line, 255);
    if (strlen(line) == 0) {
      if (!lastLineIsEmpty)
        out << endl;
      lastLineIsEmpty = true;
    }
    else {
      out << line << endl;
      lastLineIsEmpty = false;
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    cout << "example: ./cpu0filepreprocess Cpu0/Cpu0AsmPrinter.cpp Chapter6_1/\
            Cpu0AsmPrinter.cpp CH6_1" << endl;
	  return 1;
  }
#ifdef DEBUG
  cout << "argv[1]: " << argv[1] << endl;
  cout << "argv[2]: " << argv[2] << endl;
  cout << "argv[3]: " << argv[3] << endl;
#endif
  ifstream in(argv[1]);
  if (!in.is_open()) {
    cout << "input file " << argv[1] << " not exist!" << endl;
    return 1;
  }
  
  Preprocess preprocess;
  if (strncmp(argv[2]+strlen(argv[2])-strlen(".cpp"), ".cpp", strlen(".cpp")) 
              == 0) {
    preprocess.setFileType(CPP_FILE);
  }
  else if (strncmp(argv[2]+strlen(argv[2])-strlen(".h"), ".h", strlen(".h")) 
                   == 0) {
    preprocess.setFileType(CPP_FILE);
  }
  else if (strncmp(argv[2]+strlen(argv[2])-strlen(".td"), ".td", strlen(".td")) 
                   == 0) {
    preprocess.setFileType(TD_FILE);
  }
  else if (strncmp(argv[2]+strlen(argv[2])-strlen("CMakeLists.txt"), 
                   "CMakeLists.txt", strlen("CMakeLists.txt")) == 0) {
    preprocess.setFileType(CMAKE_FILE);
  }
  else if (strncmp(argv[2]+strlen(argv[2])-strlen("LLVMBuild.txt"), 
                   "LLVMBuild.txt", strlen("LLVMBuild.txt")) == 0) {
    preprocess.setFileType(LLVMBUILD_FILE);
  }
  else {
    return 0;
  }

  ofstream out("tmp.txt");

  preprocess.currentChapterId = preprocess.getChapterId(argv[3]);
  preprocess.run(in, out);
  in.close();
  out.close();
  in.open("tmp.txt");
  out.open(argv[2]);
  preprocess.removeSuccessiveEmptyLines(in, out);
  in.close();
  out.close();

  return 0;
}
