#include <napi.h>
#include "bindings/TrainingSample.h"
#include "bindings/test.h"
#include "bindings/MLP.h"

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
	TrainingSample::Init(env, exports);
    MLP::Init(env, exports);
    Test::Init(env, exports);
    return exports;
}

NODE_API_MODULE(mlp_addon, InitAll)