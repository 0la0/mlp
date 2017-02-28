import java.util.Random;
import java.util.stream.IntStream;

public class Neuron {

    private double input;
    private double output;
    private double[] weights;

    public Neuron(int numSynapses) {
        weights = getRandomWeights(numSynapses);
    }

    public Neuron(double output, int numSynapses) {
        this.output = output;
        weights = getRandomWeights(numSynapses);
    }

    public Neuron(double input, double output, int numSynapses) {
        this.input = input;
        this.output = output;
        weights = getRandomWeights(numSynapses);
    }

    public double getOutput() {
        return output;
    }

    public double getInput() {
        return input;
    }

    public double[] getWeights() {
        return weights;
    }

    private double[] getRandomWeights(int numSynapses) {
        Random rand = new Random();
        return IntStream.range(0, numSynapses)
                .mapToDouble(i -> rand.nextGaussian())
                .toArray();
    }

}
