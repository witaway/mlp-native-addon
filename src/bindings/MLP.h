#ifndef MLP_NATIVE_ADDON_MLP_H
#define MLP_NATIVE_ADDON_MLP_H

#include <napi.h>
#include "../lib/MLP.h"
#include "TrainingSample.h"

class MLP : public Napi::ObjectWrap<MLP> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    MLP(const Napi::CallbackInfo& info);
    MLP_Lib::MLP* GetInternalInstance();
    static Napi::FunctionReference constructor;
private:
    Napi::Value SaveMLPNetwork(const Napi::CallbackInfo &info);
    Napi::Value LoadMLPNetwork(const Napi::CallbackInfo &info);

    Napi::Value GetOutput(const Napi::CallbackInfo &info);
    Napi::Value GetOutputClass(const Napi::CallbackInfo &info);

    Napi::Value Train(const Napi::CallbackInfo &info);

    MLP_Lib::MLP *actualClass_;
};

#endif //MLP_NATIVE_ADDON_MLP_H
