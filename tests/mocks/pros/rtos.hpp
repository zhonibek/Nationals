#pragma once
#include <cstdint>
#define TIMEOUT_MAX 0xffffffff
#define PROS_ERR 2147483647
namespace pros {
inline uint32_t testTime=0;
inline void (*delayHook)()=nullptr;
inline uint32_t millis(){return testTime;}
class Mutex{public:void take(uint32_t){}void give(){}};
class Task{public:static void delay_until(uint32_t* next,uint32_t step){*next+=step;testTime=*next;if(delayHook)delayHook();}};
}
