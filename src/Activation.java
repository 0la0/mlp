public interface Activation {

    double run(double input);

    class Sigmoid implements Activation {
        @Override
        public double run(double input) {
            return 1 / (1 + Math.exp(-input));
        }
    }

    class SigmoidPrime implements Activation {
        @Override
        public double run(double input) {
            double result = 1 / (1 + Math.exp(-input));
            return result * (1 - result);
        }
    }

}
