#include "TrainingSample.h"

Napi::FunctionReference TrainingSample::constructor;

Napi::Object TrainingSample::Init(Napi::Env env, Napi::Object exports) {
    Napi::HandleScope scope(env);

    Napi::Function func = DefineClass(env, "TrainingSample", {
            InstanceAccessor<&TrainingSample::InputVector>("inputVector"),
            InstanceAccessor<&TrainingSample::OutputVector>("outputVector"),
            InstanceMethod<&TrainingSample::AddBiasValue>("addBiasValue"),
    });

    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();

    exports.Set("TrainingSample", func);
    return exports;
}

TrainingSample::TrainingSample(const Napi::CallbackInfo& info) : Napi::ObjectWrap<TrainingSample>(info)  {
    Napi::Env env = info.Env();
    Napi::HandleScope scope(env);

    if(info.Length() != 2) {
        Napi::TypeError::New(env, "Only two arguments expected").ThrowAsJavaScriptException();
    }

    if(!info[0].IsArray()) {
        Napi::TypeError::New(env, "inputVector must be an array of numbers").ThrowAsJavaScriptException();
    }

    if(!info[1].IsArray()) {
        Napi::TypeError::New(env, "outputVector must be an array of numbers").ThrowAsJavaScriptException();
    }

    Napi::Array _inputVector = info[0].As<Napi::Array>();
    std::vector<double> inputVector;

    Napi::Array _outputVector = info[1].As<Napi::Array>();
    std::vector<double> outputVector;

    for(int i = 0; i < _inputVector.Length(); i++) {
        if(!_inputVector.Get(i).IsNumber()) {
            Napi::TypeError::New(env, "inputVector must be an array of numbers").ThrowAsJavaScriptException();
        } else {
            double value = _inputVector.Get(i).As<Napi::Number>().DoubleValue();
            inputVector.push_back(value);
        }
    }

    for(int i = 0; i < _outputVector.Length(); i++) {
        if(!_outputVector.Get(i).IsNumber()) {
            Napi::TypeError::New(env, "inputVector must be an array of numbers").ThrowAsJavaScriptException();
        } else {
            double value = _outputVector.Get(i).As<Napi::Number>().DoubleValue();
            outputVector.push_back(value);
        }
    }

    this->actualClass_ = new MLP_Lib::TrainingSample(inputVector, outputVector);
}

MLP_Lib::TrainingSample* TrainingSample::GetInternalInstance() {
    return this->actualClass_;
}

Napi::Value TrainingSample::InputVector(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();
    Napi::HandleScope scope(env);

    std::vector<double> input_vector = this->GetInternalInstance()->input_vector();

    Napi::Array result = Napi::Array::New(env, input_vector.size());
    for(int i = 0; i < input_vector.size(); i++) {
        result.Set(i, Napi::Number::New(env, input_vector[i]));
    }

    return result;
}

Napi::Value TrainingSample::OutputVector(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();
    Napi::HandleScope scope(env);

    std::vector<double> output_vector = this->GetInternalInstance()->output_vector();

    Napi::Array result = Napi::Array::New(env, output_vector.size());
    for(int i = 0; i < output_vector.size(); i++) {
        result.Set(i, Napi::Number::New(env, output_vector[i]));
    }

    return result;
}

Napi::Value TrainingSample::AddBiasValue(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();
    Napi::HandleScope scope(env);

    if(info.Length() != 1 || !info[0].IsNumber()) {
        Napi::TypeError::New(env, "Number expected").ThrowAsJavaScriptException();
    }

    Napi::Number _biasValue = info[0].As<Napi::Number>();
    double biasValue = _biasValue.DoubleValue();

    this->GetInternalInstance()->AddBiasValue(biasValue);

    return env.Undefined();
}
