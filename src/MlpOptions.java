public class MlpOptions {

    private Matrix input;
    private Matrix expectedOutput;
    private int[] hiddenLayers = new int[]{3, 4};
    private int numIterations = 10000;
    private double meanErrorExitThreshold = 0.005; //0.5%

    public Matrix getInput() {
        return input;
    }

    public Matrix getExpectedOutput() {
        return expectedOutput;
    }

    public int[] getHiddenLayers() {
        return hiddenLayers;
    }

    public int getNumIterations() {
        return numIterations;
    }

    public double getMeanErrorExitThreshold() {
        return meanErrorExitThreshold;
    }

    public static class Builder {

        private MlpOptions options = new MlpOptions();

        public Builder setInput(Matrix input) {
            options.input = input;
            return this;
        }

        public Builder setExpectedOutput(Matrix expectedOutput) {
            options.expectedOutput = expectedOutput;
            return this;
        }

        public Builder setHiddenLayers(int[] hiddenLayers) {
            options.hiddenLayers = hiddenLayers;
            return this;
        }

        public Builder setMeanErrorExitThreshold(double meanErrorExitThreshold) {
            options.meanErrorExitThreshold = meanErrorExitThreshold;
            return this;
        }

        public Builder setNumIterations(int numIterations) {
            options.numIterations = numIterations;
            return this;
        }

        public MlpOptions build() {
            return options;
        }

    }

}
