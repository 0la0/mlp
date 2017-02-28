public class LayerResult {

    public Matrix sum;
    public Matrix output;

    public String toString() {
        return "\nsum: " + sum.toString() + "\noutput: " + output;
    }

    public Matrix getOutput() {
        return output;
    }

}
