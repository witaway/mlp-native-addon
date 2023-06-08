#include <napi.h>
#include "TrainingSample.h"

#ifndef MLP_NATIVE_ADDON_TEST_H
#define MLP_NATIVE_ADDON_TEST_H

namespace Test {

    Napi::Boolean Test(const Napi::CallbackInfo &info) {
        Napi::Env env = info.Env();
        bool x = info[0].ToObject().InstanceOf(TrainingSample::constructor.Value());
        TrainingSample* a = TrainingSample::Unwrap(info[0].ToObject());
        //        x->GetInternalInstance()->AddBiasValue(1);
        return Napi::Boolean::New(env, x);
    }

    Napi::Object Init(Napi::Env env, Napi::Object exports) {
        exports.Set("test", Napi::Function::New(env, Test));
    }

}

#endif //MLP_NATIVE_ADDON_TEST_H
