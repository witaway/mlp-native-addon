export class TrainingSample {
    constructor(inputVector: number[], outputVector: number[]);
    get inputVector(): number[];
    get outputVector(): number[];
    addBiasValue(biasValue: number): void;
}

type Activation = "sigmoid" | "linear";

export class MLP {

    constructor(filename: string);
    constructor(layersNodes: number[],
                layersActivations: Activation[], customWeightInit?: number);

    saveMLPNetwork(filename: string): void;
    loadMLPNetwork(filename: string): void;

    getOutput(input: number[]): number[];
    getOutputClass(input: number[]): number;

    train(trainingSampleSet: TrainingSample[], learningRate: number): void;
    train(trainingSampleSet: TrainingSample[], learningRate: number,
          maxIterations: number, minErrorCost: number): void;
}



