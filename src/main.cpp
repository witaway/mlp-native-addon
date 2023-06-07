#include <napi.h>
#include "bindings/functionexample.h"
#include "bindings/classexample.h"

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
    functionexample::Init(env, exports);
    ClassExample::Init(env, exports);
	return exports;
}

NODE_API_MODULE(mlp_addon, InitAll)