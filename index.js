const mlp = require('./lib/index');

const trainingSamples = [
    new mlp.TrainingSample([0, 0], [0]),
    new mlp.TrainingSample([0, 0], [0]),
    new mlp.TrainingSample([0, 0], [0]),
    new mlp.TrainingSample([0, 1], [1]),
    new mlp.TrainingSample([1, 0], [1]),
    new mlp.TrainingSample([1, 1], [1])
]

for(let sample of trainingSamples) {
    sample.addBiasValue(1);
}

numFeatures = trainingSamples[0].inputVector.length;
numOutputs = trainingSamples[0].outputVector.length;

network = new mlp.MLP([numFeatures, 2, numOutputs], ['sigmoid', 'linear']);
network.train(trainingSamples, 0.5, 500, 0.25);

for(let sample of trainingSamples) {
    const output = network.getOutput(sample.inputVector);
    for(let i = 0; i < numOutputs; i++) {
        const predicted = output[i] > 0.5;
        const correct = sample.outputVector[i] > 0.5;
        console.log(predicted === correct);
    }
}