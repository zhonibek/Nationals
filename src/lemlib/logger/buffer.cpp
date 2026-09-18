#include "lemlib/logger/buffer.hpp"
#include "lemlib/safety.hpp"
namespace lemlib {
Buffer::Buffer(std::function<void(const std::string&)> fn):bufferFunc(fn),task([this]{taskLoop();}) {}
Buffer::~Buffer(){running=false;task.join();}
bool Buffer::buffersEmpty(){Lock guard(mutex);return buffer.empty();}
void Buffer::pushToBuffer(const std::string& data){
    Lock guard(mutex);
    if(buffer.size()>=128 || data.size()>4096) {++dropped;return;}
    buffer.push_back(data);
}
void Buffer::setRate(uint32_t value){rate=std::clamp<uint32_t>(value,1u,1000u);}
void Buffer::taskLoop(){
    while(running){
        std::string line;bool have=false;
        {Lock guard(mutex);if(!buffer.empty()){line=std::move(buffer.front());buffer.pop_front();have=true;}}
        if(have) bufferFunc(line); // Slow I/O must never hold the producer mutex.
        pros::delay(rate.load());
    }
}
}
