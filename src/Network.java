import java.util.*;
import java.util.stream.Collectors;

public class Network {

    private double learningRate = 0.7;
    private Activation activation = new Activation.Sigmoid();
    private Activation activationPrime = new Activation.SigmoidPrime();

    private List<Matrix> weights;

    public Network(Matrix input, Matrix expectedOutput, int[] hiddenLayerSizes) {
        weights = getWeights(input.getNumColumns(), hiddenLayerSizes[0], expectedOutput.getNumColumns());
        train(input, expectedOutput);
    }

    public Matrix predict(Matrix input) {
        List<LayerResult> predictResults = forward(input, weights);
        return predictResults.get(predictResults.size() - 1).getOutput();
    }

    private void train(Matrix input, Matrix expectedOutput) {
        Matrix error = null;
        for (int i = 0; i < 10000; i++) {
            List<LayerResult> forwardResults = forward(input, weights);
            error = MatrixOperator.subtract(expectedOutput, forwardResults.get(forwardResults.size() - 1).output);
            weights = back(input, expectedOutput, forwardResults, weights);

            //if error rate is not decreasing at an acceptable rate, do random restart

//            if (i % 1000 == 0) {
//                System.out.println("weights" + weights);
//                System.out.println("error: " + error);
//            }
            //TODO: adjust learning rate
//            if (learningRate > 0) {
//                learningRate -= 0.0001;
//            }
        }
        System.out.println("final error:" + error);
    }

    private List<Matrix> back(Matrix input, Matrix expectedOutput, List<LayerResult> layerResults, List<Matrix> weights) {
        List<Matrix> updatedWeights = new LinkedList<>();
        int backIndex = layerResults.size() - 1;

        //output > hidden
        LayerResult forwardLayer = layerResults.get(backIndex);
        Matrix errorOutput = MatrixOperator.subtract(expectedOutput, forwardLayer.output);
        Matrix sumPrime = MatrixOperator.transform(forwardLayer.sum, activationPrime);
        Matrix delta = MatrixOperator.dot(sumPrime, errorOutput);
        Matrix weightDelta = MatrixOperator.multiply(MatrixOperator.transpose(layerResults.get(backIndex - 1).output), delta);
        Matrix weight = MatrixOperator.add(weights.get(backIndex), MatrixOperator.multiply(weightDelta, learningRate));
        updatedWeights.add(0, weight);

        forwardLayer = layerResults.get(--backIndex);

        //hidden > hidden


        //hidden > input
        errorOutput = MatrixOperator.multiply(delta, MatrixOperator.transpose(weights.get(1)));
        sumPrime = MatrixOperator.transform(forwardLayer.sum, activationPrime);
        delta = MatrixOperator.dot(errorOutput, sumPrime);
        weightDelta = MatrixOperator.multiply(MatrixOperator.transpose(input), delta);
        weight = MatrixOperator.add(weights.get(backIndex), MatrixOperator.multiply(weightDelta, learningRate));
        updatedWeights.add(0, weight);


        return updatedWeights;
    }

    private LayerResult getLayerResult(Matrix input, Matrix weight, Activation activation) {
        LayerResult layerResult = new LayerResult();
        layerResult.sum = MatrixOperator.multiply(input, weight);
        layerResult.output = MatrixOperator.transform(layerResult.sum, activation);
        return layerResult;
    }

    private List<LayerResult> forward(Matrix input, List<Matrix> weights) {
        List<LayerResult> forwardResults = new ArrayList<>();
        forwardResults.add(getLayerResult(input, weights.get(0), activation));


        //TODO: add middle layers


        Matrix lastIn = forwardResults.get(forwardResults.size() - 1).output;
        Matrix lastWeight = weights.get(weights.size() - 1);
        forwardResults.add(getLayerResult(lastIn, lastWeight, activation));

        return forwardResults;
    }

    //rows: length of input feature vector
    //cols: number of node in subsequent layer
    private List<Matrix> getWeights(int inputSize, int hiddenSize, int outputSize) {
        List<Matrix> weights = new ArrayList<>();
        weights.add(new Matrix(inputSize, hiddenSize));

        //TODO: add hidden > hidden layers

        weights.add(new Matrix(hiddenSize, outputSize));
        return weights;
    }


}
