/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Calling Convention Implementation Fragment                                 *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

static bool RetCC_Cpu0(unsigned ValNo, MVT ValVT,
                       MVT LocVT, CCValAssign::LocInfo LocInfo,
                       ISD::ArgFlagsTy ArgFlags, CCState &State);
static bool RetCC_Cpu0EABI(unsigned ValNo, MVT ValVT,
                           MVT LocVT, CCValAssign::LocInfo LocInfo,
                           ISD::ArgFlagsTy ArgFlags, CCState &State);


static bool RetCC_Cpu0(unsigned ValNo, MVT ValVT,
                       MVT LocVT, CCValAssign::LocInfo LocInfo,
                       ISD::ArgFlagsTy ArgFlags, CCState &State) {

  if (!RetCC_Cpu0EABI(ValNo, ValVT, LocVT, LocInfo, ArgFlags, State))
    return false;

  return true; // CC didn't match.
}


static bool RetCC_Cpu0EABI(unsigned ValNo, MVT ValVT,
                           MVT LocVT, CCValAssign::LocInfo LocInfo,
                           ISD::ArgFlagsTy ArgFlags, CCState &State) {

  if (LocVT == MVT::i32) {
    static const MCPhysReg RegList1[] = {
      Cpu0::V0, Cpu0::V1, Cpu0::A0, Cpu0::A1
    };
    if (unsigned Reg = State.AllocateReg(RegList1)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  return true; // CC didn't match.
}
