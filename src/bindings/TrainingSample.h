#ifndef MLP_NATIVE_ADDON_TRAININGSAMPLE_H
#define MLP_NATIVE_ADDON_TRAININGSAMPLE_H

#include <napi.h>
#include "../lib/Sample.h"

class TrainingSample : public Napi::ObjectWrap<TrainingSample> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    TrainingSample(const Napi::CallbackInfo& info);
    MLP_Lib::TrainingSample* GetInternalInstance();
    static Napi::FunctionReference constructor;
private:
    Napi::Value InputVector(const Napi::CallbackInfo &info);
    Napi::Value OutputVector(const Napi::CallbackInfo &info);
    Napi::Value AddBiasValue(const Napi::CallbackInfo &info);
    MLP_Lib::TrainingSample *actualClass_;
};

#endif //MLP_NATIVE_ADDON_TRAININGSAMPLE_H