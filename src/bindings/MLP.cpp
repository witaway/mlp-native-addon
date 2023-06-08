#include "MLP.h"

Napi::FunctionReference MLP::constructor;

Napi::Object MLP::Init(Napi::Env env, Napi::Object exports) {
    Napi::HandleScope scope(env);

    Napi::Function func = DefineClass(env, "MLP", {
            InstanceMethod<&MLP::SaveMLPNetwork>("saveMLPNetwork"),
            InstanceMethod<&MLP::LoadMLPNetwork>("loadMLPNetwork"),
            InstanceMethod<&MLP::GetOutput>("getOutput"),
            InstanceMethod<&MLP::GetOutputClass>("getOutputClass"),
            InstanceMethod<&MLP::Train>("train")
    });

    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();

    exports.Set("MLP", func);
    return exports;
}

MLP::MLP(const Napi::CallbackInfo& info) : Napi::ObjectWrap<MLP>(info)  {
    Napi::Env env = info.Env();
    Napi::HandleScope scope(env);

    // If passed
    //  filename: string
    if(info.Length() == 1 && info[0].IsString()) {
        std::string filename = info[0].ToString();
        this->actualClass_ = new MLP_Lib::MLP(filename);
        return;
    }

    // If passed
    //  layersNodes: number[],
    //  layersActivations: string[],
    //  customWeightInit: number,

    if(info.Length() < 2) {
        Napi::TypeError::New(env, "Constructor excepts at least 2 arguments").ThrowAsJavaScriptException();
        return;
    }

    if(!info[0].IsArray()) {
        Napi::TypeError::New(env, "layersNodes must be an array of numbers").ThrowAsJavaScriptException();
        return;
    }

    if(!info[1].IsArray()) {
        Napi::TypeError::New(env, "layersActivations must be an array of strings").ThrowAsJavaScriptException();
        return;
    }

    Napi::Array _layersNodes = info[0].As<Napi::Array>();
    std::vector<uint64_t> layersNodes;

    Napi::Array _layersActivations = info[1].As<Napi::Array>();
    std::vector<std::string> layersActivations;

    for(int i = 0; i < _layersNodes.Length(); i++) {
        if(!_layersNodes.Get(i).IsNumber()) {
            Napi::TypeError::New(env, "layersNodes must contain only positive numbers").ThrowAsJavaScriptException();
            return;
        }
        const int64_t value = _layersNodes.Get(i).As<Napi::Number>().Int64Value();
        if(value < 0) {
            Napi::TypeError::New(env, "layersNodes must contain only positive numbers").ThrowAsJavaScriptException();
        }
        layersNodes.push_back((uint64_t)value);
    }

    for(int i = 0; i < _layersActivations.Length(); i++) {
        if(!_layersActivations.Get(i).IsString()) {
            Napi::TypeError::New(env, "layersActivations must contain only strings").ThrowAsJavaScriptException();
            return;
        }
        std::string value = _layersActivations.Get(i).As<Napi::String>().ToString();
        if(value != "sigmoid" && value != "linear") {
            Napi::TypeError::New(env, "Activation function must only be 'sigmoid' or 'linear'").ThrowAsJavaScriptException();
            return;
        }
        layersActivations.push_back(value);
    }

    bool useCustomWeightInit = false;
    double customWeightInit;

    if(info.Length() == 3) {
        if(!info[2].IsNumber()) {
            Napi::TypeError::New(env, "customWeightInit must be a number").ThrowAsJavaScriptException();
            return;
        }
        useCustomWeightInit = true;
        customWeightInit = info[2].As<Napi::Number>().DoubleValue();
    }

    if(!useCustomWeightInit) {
        this->actualClass_ = new MLP_Lib::MLP(layersNodes, layersActivations);
    } else {
        this->actualClass_ = new MLP_Lib::MLP(layersNodes, layersActivations, true, customWeightInit);
    }
}

MLP_Lib::MLP* MLP::GetInternalInstance() {
    return this->actualClass_;
}

Napi::Value MLP::SaveMLPNetwork(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();

    if(info.Length() != 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Filename (string) expected").ThrowAsJavaScriptException();
    }

    std::string filename = info[0].ToString();
    this->GetInternalInstance()->SaveMLPNetwork(filename);

    return env.Undefined();
}

Napi::Value MLP::LoadMLPNetwork(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();

    if(info.Length() != 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Filename (string) expected").ThrowAsJavaScriptException();
    }

    std::string filename = info[0].ToString();
    this->GetInternalInstance()->LoadMLPNetwork(filename);

    return env.Undefined();
}

Napi::Value MLP::GetOutput(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();

    if(info.Length() != 1 || !info[0].IsArray()) {
        Napi::TypeError::New(env, "input (array of numbers) expected").ThrowAsJavaScriptException();
    }

    Napi::Array _input = info[0].As<Napi::Array>();
    std::vector<double> input;

    for(int i = 0; i < _input.Length(); i++) {
        if(!_input.Get(i).IsNumber()) {
            Napi::TypeError::New(env, "input must contain only numbers").ThrowAsJavaScriptException();
        }
        const double value = _input.Get(i).As<Napi::Number>().DoubleValue();
        input.push_back(value);
    }

    std::vector<double> output;
    this->GetInternalInstance()->GetOutput(input, &output);

    Napi::Array _output = Napi::Array::New(env);
    for(int i = 0; i < output.size(); i++) {
        _output.Set(i, Napi::Number::New(env, output[i]));
    }

    return _output;
}

Napi::Value MLP::GetOutputClass(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();

    if(info.Length() != 1 || !info[0].IsArray()) {
        Napi::TypeError::New(env, "input (array of numbers) expected").ThrowAsJavaScriptException();
    }

    Napi::Array _input = info[0].As<Napi::Array>();
    std::vector<double> input;

    for(int i = 0; i < _input.Length(); i++) {
        if(!_input.Get(i).IsNumber()) {
            Napi::TypeError::New(env, "input must contain only numbers").ThrowAsJavaScriptException();
        }
        const double value = _input.Get(i).As<Napi::Number>().DoubleValue();
        input.push_back(value);
    }

    size_t class_id;
    this->GetInternalInstance()->GetOutputClass(input, &class_id);

    return Napi::Number::New(env, class_id);
}

Napi::Value MLP::Train(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();

    if(info.Length() != 4 && info.Length() != 2) {
        Napi::TypeError::New(env, "train must receive 4 or 2 arguments");
    }

    if(!info[0].IsArray()) {
        Napi::TypeError::New(env, "trainingSampleSet (array of TrainingSample) expected").ThrowAsJavaScriptException();
    }

    Napi::Array _trainingSampleSet = info[0].As<Napi::Array>();
    std::vector<MLP_Lib::TrainingSample> trainingSampleSet;

    for(int i = 0; i < _trainingSampleSet.Length(); i++) {
        if(!_trainingSampleSet.Get(i).ToObject().InstanceOf(TrainingSample::constructor.Value())) {
            Napi::TypeError::New(env, "trainingSampleSet must contain only TrainingSample instances").ThrowAsJavaScriptException();
        }
        TrainingSample *sample = TrainingSample::Unwrap(_trainingSampleSet.Get(i).ToObject());
        trainingSampleSet.push_back(*sample->GetInternalInstance());
    }

    if(!info[1].IsNumber()) {
        Napi::TypeError::New(env, "learningRate must be a number").ThrowAsJavaScriptException();
    }
    double learningRate = info[1].As<Napi::Number>().DoubleValue();

    int maxIterations = 5000;
    double minErrorCost = 0.001;


    if(info.Length() == 4) {
        if (!info[2].IsNumber()) {
            Napi::TypeError::New(env, "maxIterations must be a number").ThrowAsJavaScriptException();
        }
        maxIterations = info[2].As<Napi::Number>().Int32Value();

        if (!info[3].IsNumber()) {
            Napi::TypeError::New(env, "minErrorCost must be a number").ThrowAsJavaScriptException();
        }
        minErrorCost = info[3].As<Napi::Number>().DoubleValue();
    }

    this->GetInternalInstance()->Train(trainingSampleSet, learningRate, maxIterations, minErrorCost);

    return env.Undefined();
}