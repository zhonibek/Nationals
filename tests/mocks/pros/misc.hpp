#pragma once
namespace pros::competition{inline int testMode=0;inline bool is_disabled(){return testMode&1;}inline int get_status(){return testMode;}}
