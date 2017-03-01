import java.util.*;
import java.util.stream.Collectors;

public class MLP {

    private double learningRate = 0.8;
    private Activation activation = new Activation.Sigmoid();
    private Activation activationPrime = new Activation.SigmoidPrime();

    private List<Matrix> weights;

    public MLP(Matrix input, Matrix expectedOutput, int[] hiddenLayerSizes) {
        weights = getInitialWeights(input.getNumColumns(), hiddenLayerSizes, expectedOutput.getNumColumns());
        String wieghtString = weights.stream()
                .map(Matrix::getDimensionString)
                .collect(Collectors.joining("\n"));
        System.out.println("wieghtString: \n" + wieghtString);
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
//                learningRate -= 0.000001;
//            }
        }

        double meanError = error.getMeanValue();
        double errorPercent = Math.round(meanError * 10000) / 100.0;
        System.out.println("final error:" + error);
        System.out.println("mean error: " + error.getMeanValue());
        System.out.println("or: " + errorPercent + "%");
    }

    private List<Matrix> back(Matrix input, Matrix expectedOutput, List<LayerResult> layerResults, List<Matrix> weights) {
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

    private LayerResult getLayerResult(Matrix input, Matrix weight, Activation activation) {
        LayerResult layerResult = new LayerResult();
        layerResult.sum = MatrixOperator.multiply(input, weight);
        layerResult.output = MatrixOperator.transform(layerResult.sum, activation);
        return layerResult;
    }

    private List<LayerResult> forward(Matrix input, List<Matrix> weights) {
        int forwardIndex = 0;
        List<LayerResult> forwardResults = new LinkedList<>();
        forwardResults.add(getLayerResult(input, weights.get(forwardIndex), activation));

        while (forwardIndex++ < weights.size() - 2) {
            Matrix in = forwardResults.get(forwardResults.size() - 1).output;
            forwardResults.add(getLayerResult(in, weights.get(forwardIndex), activation));
        }

        Matrix lastIn = forwardResults.get(forwardResults.size() - 1).output;
        Matrix lastWeight = weights.get(forwardIndex);
        forwardResults.add(getLayerResult(lastIn, lastWeight, activation));

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

}
