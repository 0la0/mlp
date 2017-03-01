import java.util.List;

public class driver {

    public static void main(String[] args) {
        long startTime = System.currentTimeMillis();

//        testLogicGates();
        testMultOut();

        System.out.println("time elapsed: " + (System.currentTimeMillis() - startTime));
    }

    private static void testMultOut() {
        Matrix input = new Matrix(new double[][]{
                {0, 0, 0},
                {0, 0, 1},
                {0, 1, 0},
                {0, 1, 1},
                {1, 0, 0},
                {1, 0, 1},
                {1, 1, 0},
                {1, 1, 1},
        });

        Matrix expectedOutput = new Matrix(new double[][]{
                {0, 1},
                {0.5, 0},
                {0, 1},
                {0, 0},
                {0.3, 0.4},
                {.5, .5},
                {1, 1},
                {0.2, 0.9}
        });

        int[] hiddenLayers = new int[]{3, 6, 4, 3};

        Network network = new Network(input, expectedOutput, hiddenLayers);

        Matrix testData0 = new Matrix(new double[][]{
                {0, 0, 0},
                {0, 0, 1},
                {0, 1, 0},
                {0, 1, 1},
                {1, 0, 0},
                {1, 0, 1},
                {1, 1, 0},
                {1, 1, 1},
        });


        Matrix predictions0 = network.predict(testData0);
        System.out.println("predictions0:" + predictions0);
    }

    private static void testLogicGates() {
        Matrix input = new Matrix(new double[][]{
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        });

        Matrix expectedOutput = new Matrix(new double[][]{
                {0},
                {0},
                {0},
                {1}
        });

        int[] hiddenLayers = new int[]{3, 6, 4, 3};

        Network network = new Network(input, expectedOutput, hiddenLayers);


        Matrix testData0 = new Matrix(new double[][]{
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        });
        Matrix testData1 = new Matrix(new double[][]{
                {0.1, 0.1},
                {0, 0.7},
                {.9, 0.01},
                {.7, 0.8}
        });

        Matrix predictions0 = network.predict(testData0);
        Matrix predictions1 = network.predict(testData1);
        System.out.println("predictions0:" + predictions0);
        System.out.println("predictions1:" + predictions1);
    }

    private static void test() {
        Matrix m1 = new Matrix(new double[][]{
                {1, 2, 0},
                {0, 1, 1},
                {2, 0, 1}
        });
        Matrix m2 = new Matrix(new double[][]{
                {1, 1, 2},
                {2, 1, 1},
                {1, 2, 1}
        });

        Matrix m3 = new Matrix(new double[][]{
                {1, 2},
                {3, 4},
                {5, 6}
        });



//        Matrix mult1 = MatrixOperator.multiply(m1, m2);
//        System.out.println(mult1.toString());
//
//        Matrix mult2 = MatrixOperator.multiply(m1, 2);
//        System.out.println(mult2.toString());
//
//        Matrix transpose = MatrixOperator.transpose(m3);
//        System.out.println(transpose.toString());
    }

}
