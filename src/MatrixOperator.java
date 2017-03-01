import java.util.Arrays;
import java.util.stream.IntStream;

public class MatrixOperator {

    private interface ElementFunction {
        double run(double w, double v);
    }

    public static Matrix multiply(Matrix m1, Matrix m2) {
        if (m1.getNumColumns() != m2.getNumRows()) {
            String m1Dims = m1.getDimensionString();
            String m2Dims = m2.getDimensionString();
            System.out.println("MatrixOperator.multiply ERROR: incompatible dimensions " + m1Dims + " by " + m2Dims);
            return null;
        }
        int numRows = m1.getNumRows();
        int numCols = m2.getNumColumns();
        int size = numRows * numCols;
        Matrix result = new Matrix(numRows, numCols);



        for (int i = 0; i < m1.getNumRows(); i++) { //for each row of m1
            for (int j = 0; j < m2.getNumColumns(); j++) { // for each column in m1
                double sum = 0;
                for (int k = 0; k < m2.getNumRows(); k++) { // multiply by row element in m2
                    sum += m1.getElement(i, k) * m2.getElement(k, j);
                }
                result.setElement(i, j, sum);
            }
        }

        return result;
    }

    public static Matrix multiply(Matrix matrix, double scalar) {
        Matrix result = new Matrix(matrix.getNumRows(), matrix.getNumColumns());
        for (int i = 0; i < matrix.getNumRows(); i++) { //for each row
            for (int j = 0; j < matrix.getNumColumns(); j++) { // for each column
                double product = matrix.getElement(i, j) * scalar;
                result.setElement(i, j, product);
            }
        }
        return result;
    }

    public static Matrix transpose(Matrix matrix) {
        Matrix result = new Matrix(matrix.getNumColumns(), matrix.getNumRows());
        for (int i = 0; i < matrix.getNumColumns(); i++) {
            for (int j = 0; j < matrix.getNumRows(); j++) {
                result.setElement(i, j, matrix.getElement(j, i));
            }
        }
        return result;
    }

    public static Matrix transform(Matrix matrix, Activation activation) {
        Matrix result = new Matrix(matrix.getNumRows(), matrix.getNumColumns());
        for (int i = 0; i < matrix.getNumRows(); i++) {
            for (int j = 0; j < matrix.getNumColumns(); j++) {
                double transformedValue = activation.run(matrix.getElement(i, j));
                result.setElement(i, j, transformedValue);
            }
        }
        return result;
    }

    public static Matrix subtract(Matrix m1, Matrix m2) {
        return elementWise(m1, m2, (w, v) -> w - v);
    }

    public static Matrix add(Matrix m1, Matrix m2) {
        return elementWise(m1, m2, (w, v) -> w + v);
    }

    public static Matrix dot(Matrix m1, Matrix m2) {
        return elementWise(m1, m2, (w, v) -> w * v);
    }

    private static Matrix elementWise(Matrix m1, Matrix m2, ElementFunction elementFunction) {
        Matrix result = new Matrix(m1.getNumRows(), m1.getNumColumns());
        int size = m1.getNumRows() * m1.getNumColumns();

        IntStream.range(0, size).forEach(i -> {
            int row = (int) Math.floor(i / m1.getNumColumns());
            int col = i % m1.getNumColumns();
            double transformed = elementFunction.run(m1.getElement(row, col), m2.getElement(row, col));
            result.setElement(row, col, transformed);
        });

//        for (int i = 0; i < m1.getNumRows(); i++) {
//            for (int j = 0; j < m1.getNumColumns(); j++) {
//                double transformed = elementFunction.run(m1.getElement(i, j), m2.getElement(i, j));
//                result.setElement(i, j, transformed);
//            }
//        }
        return result;
    }

}
