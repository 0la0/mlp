import java.util.Arrays;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Matrix {

    private int numRows;
    private int numColumns;
    private double[][] data;
    private static Random random = new Random();

    public Matrix(int numRows, int numColumns) {
        this.numRows = numRows;
        this.numColumns = numColumns;

        data = IntStream.range(0, numRows)
                .mapToObj(i -> IntStream.range(0, numColumns)
                        .mapToDouble(j -> random.nextGaussian())
                        .toArray()
                )
                .toArray(double[][]::new);
    }

    public Matrix(double[][] data) {
        this.numRows = data.length;
        this.numColumns = data[0].length;
        this.data = data;
    }

    public double[][] getData() {
        return data;
    }


    public int getNumRows() {
        return numRows;
    }

    public int getNumColumns() {
        return numColumns;
    }

    public int getSize() {
        return numRows * numColumns;
    }

    public double getElement(int row, int column) {
        return data[row][column];
    }

    public void setElement(int row, int column, double element) {
        data[row][column] = element;
    }

    public String getDimensionString() {
        return getNumRows() + " x " + getNumColumns();
    }

    public double getMeanValue() {
        return Arrays.stream(data)
                .flatMapToDouble(Arrays::stream)
                .map(Math::abs)
                .sum() / (getSize() * 1.0);
    }

    public String toString() {
        return Arrays.stream(data)
                .map(row -> "\n[" +
                        Arrays.stream(row)
                                .mapToObj(String::valueOf)
                                .collect(Collectors.joining(", "))
                        + "]"
                )
                .collect(Collectors.joining(""));
    }

}
