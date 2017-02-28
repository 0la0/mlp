import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Layer {

    private List<Neuron> neurons;

    public Layer(int numNeurons, int nexLayerSize) {
        neurons = IntStream.range(0, numNeurons)
                .mapToObj(i -> new Neuron(nexLayerSize))
                .collect(Collectors.toList());
    }

    public Layer(double[] output, int nexLayerSize) {
        neurons = Arrays.stream(output)
                .mapToObj(outputValue -> new Neuron(outputValue, nexLayerSize))
                .collect(Collectors.toList());
    }

    public double[] getInputs() {
        return neurons.stream()
                .mapToDouble(Neuron::getInput)
                .toArray();
    }

    public double[] getOutputs() {
        return neurons.stream()
                .mapToDouble(Neuron::getOutput)
                .toArray();
    }

    public Matrix getWeights() {
        double[][] weights = neurons.stream()
                .map(Neuron::getWeights)
                .toArray(double[][]::new);
        return new Matrix(weights);
    }

    public void connectTo(Layer outputLayer) {

    }

    public int getSize() {
        return neurons.size();
    }

}
