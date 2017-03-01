import java.util.*;
import java.util.stream.Collectors;

public class MLP {

    private double learningRate = 0.8;
    private Activation activation = new Activation.Sigmoid();
    private Activation activationPrime = new Activation.SigmoidPrime();

    private List<Matrix> weights;

    public MLP(MlpOptions mlpOptions) {
        weights = getInitialWeights(mlpOptions.getInput().getNumColumns(), mlpOptions.getHiddenLayers(), mlpOptions.getExpectedOutput().getNumColumns());
        String wieghtString = weights.stream()
                .map(Matrix::getDimensionString)
                .collect(Collectors.joining("\n"));
        System.out.println("wieghtString: \n" + wieghtString);
        train(mlpOptions.getInput(), mlpOptions.getExpectedOutput(), mlpOptions.getNumIterations(), mlpOptions.getMeanErrorExitThreshold());
    }

    public Matrix predict(Matrix input) {
        List<ForwardResult> predictResults = forward(input, weights);
        return predictResults.get(predictResults.size() - 1).output;
    }

    private void train(Matrix input, Matrix expectedOutput, int numIterations, double exitThreshold) {
        Matrix error = null;
        for (int i = 0; i < numIterations; i++) {
            List<ForwardResult> forwardResults = forward(input, weights);
            error = MatrixOperator.subtract(expectedOutput, forwardResults.get(forwardResults.size() - 1).output);
            weights = back(input, expectedOutput, forwardResults, weights);

            if (i % 1000 == 0) {
                if (error.getMeanValue() <= exitThreshold) {
                    System.out.println("exit threshold meet at iteration: " + i);
                    break;
                }
            }
            //TODO: adjust learning rate
            //  if (learningRate > 0) { learningRate -= 0.000001; }
        }

        double meanError = error.getMeanValue();
        double errorPercent = Math.round(meanError * 10000) / 100.0;
        System.out.println("final error:" + error + "\nmean error: " + meanError + ", or: " + errorPercent + "%");
    }

    private List<Matrix> back(Matrix input, Matrix expectedOutput, List<ForwardResult> layerResults, List<Matrix> weights) {
        List<Matrix> updatedWeights = new LinkedList<>();
        int backIndex = layerResults.size() - 1;

        //output > hidden
        Matrix errorOutput = MatrixOperator.subtract(expectedOutput, layerResults.get(backIndex).output);
        Matrix sumPrime = MatrixOperator.transform(layerResults.get(backIndex).sum, activationPrime);
        Matrix delta = MatrixOperator.multiplyElements(sumPrime, errorOutput);
        Matrix weightDelta = MatrixOperator.multiply(MatrixOperator.transpose(layerResults.get(backIndex - 1).output), delta);
        Matrix weight = MatrixOperator.add(weights.get(backIndex), MatrixOperator.multiplyScalar(weightDelta, learningRate));
        updatedWeights.add(0, weight);

        backIndex--;

        //hidden > hidden
        while (backIndex > 0) {
            errorOutput = MatrixOperator.multiply(delta, MatrixOperator.transpose(weights.get(backIndex + 1)));
            sumPrime = MatrixOperator.transform(layerResults.get(backIndex).sum, activationPrime);
            delta = MatrixOperator.multiplyElements(errorOutput, sumPrime);
            weightDelta = MatrixOperator.multiply(MatrixOperator.transpose(layerResults.get(backIndex - 1).output), delta);
            weight = MatrixOperator.add(weights.get(backIndex), MatrixOperator.multiplyScalar(weightDelta, learningRate));
            updatedWeights.add(0, weight);

            backIndex--;
        }


        //hidden > input
        errorOutput = MatrixOperator.multiply(delta, MatrixOperator.transpose(weights.get(1)));
        sumPrime = MatrixOperator.transform(layerResults.get(backIndex).sum, activationPrime);
        delta = MatrixOperator.multiplyElements(errorOutput, sumPrime);
        weightDelta = MatrixOperator.multiply(MatrixOperator.transpose(input), delta);
        weight = MatrixOperator.add(weights.get(backIndex), MatrixOperator.multiplyScalar(weightDelta, learningRate));
        updatedWeights.add(0, weight);

        return updatedWeights;
    }

    private ForwardResult getForwardLayer(Matrix input, Matrix weight, Activation activation) {
        ForwardResult layerResult = new ForwardResult();
        layerResult.sum = MatrixOperator.multiply(input, weight);
        layerResult.output = MatrixOperator.transform(layerResult.sum, activation);
        return layerResult;
    }

    private List<ForwardResult> forward(Matrix input, List<Matrix> weights) {
        int forwardIndex = 0;
        List<ForwardResult> forwardResults = new LinkedList<>();
        forwardResults.add(getForwardLayer(input, weights.get(forwardIndex), activation));

        while (forwardIndex++ < weights.size() - 2) {
            Matrix in = forwardResults.get(forwardResults.size() - 1).output;
            forwardResults.add(getForwardLayer(in, weights.get(forwardIndex), activation));
        }

        Matrix lastIn = forwardResults.get(forwardResults.size() - 1).output;
        Matrix lastWeight = weights.get(forwardIndex);
        forwardResults.add(getForwardLayer(lastIn, lastWeight, activation));

        return forwardResults;
    }

    private List<Matrix> getInitialWeights(int inputSize, int[] hiddenLayerSizes, int outputSize) {
        int firstHiddenLayerSize = hiddenLayerSizes[0];
        int lastHiddenLayerSize = hiddenLayerSizes[hiddenLayerSizes.length - 1];

        List<Matrix> weights = new LinkedList<>();
        weights.add(new Matrix(inputSize, firstHiddenLayerSize));

        for (int i = 0; i < hiddenLayerSizes.length - 1; i++) {
            weights.add(new Matrix(hiddenLayerSizes[i], hiddenLayerSizes[i + 1]));
        }

        weights.add(new Matrix(lastHiddenLayerSize, outputSize));
        return weights;
    }

    private class ForwardResult {
        Matrix sum;
        Matrix output;
        public String toString() {
            return "\nsum: " + sum.toString() + "\noutput: " + output;
        }
    }

}
