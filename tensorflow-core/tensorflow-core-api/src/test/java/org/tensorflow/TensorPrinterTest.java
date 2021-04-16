package org.tensorflow;

import org.junit.jupiter.api.Test;
import org.tensorflow.TensorPrinter.Options;
import org.tensorflow.ndarray.BooleanNdArray;
import org.tensorflow.ndarray.DoubleNdArray;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.IntNdArray;
import org.tensorflow.ndarray.LongNdArray;
import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.types.TBool;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TString;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TensorPrinterTest {

  @Test
  public void testPrint1D() {
    int[] ints = {1, 2, 3};
    float[] floats = {1f, 2f, 3f};
    double[] doubles = {1d, 2d, 3d};
    long[] longs = {1L, 2L, 3L};
    boolean[] bools = {true, false, true};
    String[] strings = {"A", "B", "C"};

    try (TInt32 tints = TInt32.vectorOf(ints);
        TFloat32 tfloats = TFloat32.vectorOf(floats);
        TFloat64 tdoubles = TFloat64.vectorOf(doubles);
        TInt64 tlongs = TInt64.vectorOf(longs);
        TBool tbools = TBool.vectorOf(bools);
        TString tstrings = TString.vectorOf(strings)) {

      TensorPrinter tp = new TensorPrinter(tints);
      assertEquals("[1, 2, 3]", tp.print());

      tp = new TensorPrinter(tfloats);
      assertEquals("[1.0, 2.0, 3.0]", tp.print());

      tp = new TensorPrinter(tdoubles);
      assertEquals("[1.0, 2.0, 3.0]", tp.print());

      tp = new TensorPrinter(tlongs);
      assertEquals("[1, 2, 3]", tp.print());

      tp = new TensorPrinter(tbools);
      assertEquals("[true, false, true]", tp.print());

      tp = new TensorPrinter(tstrings);
      assertEquals("[\"A\", \"B\", \"C\"]", tp.print());
    }
  }

  @Test
  public void testPrint1DSemiColon() {
    int[] ints = {1, 2, 3};
    float[] floats = {1f, 2f, 3f};
    double[] doubles = {1d, 2d, 3d};
    long[] longs = {1L, 2L, 3L};
    boolean[] bools = {true, false, true};
    String[] strings = {"A", "B", "C"};

    try (TInt32 tints = TInt32.vectorOf(ints);
         TFloat32 tfloats = TFloat32.vectorOf(floats);
         TFloat64 tdoubles = TFloat64.vectorOf(doubles);
         TInt64 tlongs = TInt64.vectorOf(longs);
         TBool tbools = TBool.vectorOf(bools);
         TString tstrings = TString.vectorOf(strings)) {

      Options options = Options.create().separator(':');
      TensorPrinter tp = new TensorPrinter(tints, options);
      assertEquals("[1: 2: 3]", tp.print());

      tp = new TensorPrinter(tfloats, options);
      assertEquals("[1.0: 2.0: 3.0]", tp.print());

      tp = new TensorPrinter(tdoubles, options);
      assertEquals("[1.0: 2.0: 3.0]", tp.print());

      tp = new TensorPrinter(tlongs, options);
      assertEquals("[1: 2: 3]", tp.print());

      tp = new TensorPrinter(tbools, options);
      assertEquals("[true: false: true]", tp.print());

      tp = new TensorPrinter(tstrings, options);
      assertEquals("[\"A\": \"B\": \"C\"]", tp.print());
    }
  }


  @Test
  public void testPrint1D2Decimal() {
    int[] ints = {1, 2, 3};
    float[] floats = {1f, 2f, 3f};
    double[] doubles = {1d, 2d, 3d};
    long[] longs = {1L, 2L, 3L};
    boolean[] bools = {true, false, true};
    String[] strings = {"A", "B", "C"};

    try (TInt32 tints = TInt32.vectorOf(ints);
        TFloat32 tfloats = TFloat32.vectorOf(floats);
        TFloat64 tdoubles = TFloat64.vectorOf(doubles);
        TInt64 tlongs = TInt64.vectorOf(longs);
        TBool tbools = TBool.vectorOf(bools);
        TString tstrings = TString.vectorOf(strings)) {

      Options options = Options.create().numDecimals(2);

      TensorPrinter tp = new TensorPrinter(tints, options);
      assertEquals("[1, 2, 3]", tp.print());

      tp = new TensorPrinter(tfloats, options);
      assertEquals("[1.00, 2.00, 3.00]", tp.print());

      tp = new TensorPrinter(tdoubles, options);
      assertEquals("[1.00, 2.00, 3.00]", tp.print());

      tp = new TensorPrinter(tlongs, options);
      assertEquals("[1, 2, 3]", tp.print());

      tp = new TensorPrinter(tbools, options);
      assertEquals("[true, false, true]", tp.print());

      tp = new TensorPrinter(tstrings, options);
      assertEquals("[\"A\", \"B\", \"C\"]", tp.print());
    }
  }

  @Test
  public void testPrint2D() {
    int[][] ints = {{1, 2}, {3, 4}};
    float[][] floats = {{1f, 2f}, {3f, 4f}};
    double[][] doubles = {{1d, 2d}, {3d, 4d}};
    long[][] longs = {{1L, 2L}, {3L, 4L}};
    boolean[][] bools = {{true, false}, {true, false}};
    String[][] strings = {{"A", "B"}, {"C", "D"}};

    IntNdArray iMatrix = StdArrays.ndCopyOf(ints);
    FloatNdArray fMatrix = StdArrays.ndCopyOf(floats);
    DoubleNdArray dMatrix = StdArrays.ndCopyOf(doubles);
    LongNdArray lMatrix = StdArrays.ndCopyOf(longs);
    BooleanNdArray bMatrix = StdArrays.ndCopyOf(bools);
    NdArray<String> sMatrix = StdArrays.ndCopyOf(strings);

    try (TInt32 tints = TInt32.tensorOf(iMatrix);
        TFloat32 tfloats = TFloat32.tensorOf(fMatrix);
        TFloat64 tdoubles = TFloat64.tensorOf(dMatrix);
        TInt64 tlongs = TInt64.tensorOf(lMatrix);
        TBool tbools = TBool.tensorOf(bMatrix);
        TString tstrings = TString.tensorOf(sMatrix)) {

      TensorPrinter tp = new TensorPrinter(tints);
      assertEquals("[\n  [1, 2]\n  [3, 4]\n]", tp.print());

      tp = new TensorPrinter(tfloats);
      assertEquals("[\n  [1.0, 2.0]\n  [3.0, 4.0]\n]", tp.print());

      tp = new TensorPrinter(tdoubles);
      assertEquals("[\n  [1.0, 2.0]\n  [3.0, 4.0]\n]", tp.print());

      tp = new TensorPrinter(tlongs);
      assertEquals("[\n  [1, 2]\n  [3, 4]\n]", tp.print());

      tp = new TensorPrinter(tbools);
      assertEquals("[\n  [true, false]\n  [true, false]\n]", tp.print());

      tp = new TensorPrinter(tstrings);
      assertEquals("[\n  [\"A\", \"B\"]\n  [\"C\", \"D\"]\n]", tp.print());
    }
  }

  @Test
  public void testPrint3D() {
    int[][][] ints = {{{1}, {2}}, {{3}, {4}}};
    float[][][] floats = {{{1f}, {2f}}, {{3f}, {4f}}};
    double[][][] doubles = {{{1d}, {2d}}, {{3d}, {4d}}};
    long[][][] longs = {{{1L}, {2L}}, {{3L}, {4L}}};
    boolean[][][] bools = {{{true}, {false}}, {{true}, {false}}};
    String[][][] strings = {{{"A"}, {"B"}}, {{"C"}, {"D"}}};

    IntNdArray iMatrix = StdArrays.ndCopyOf(ints);
    FloatNdArray fMatrix = StdArrays.ndCopyOf(floats);
    DoubleNdArray dMatrix = StdArrays.ndCopyOf(doubles);
    LongNdArray lMatrix = StdArrays.ndCopyOf(longs);
    BooleanNdArray bMatrix = StdArrays.ndCopyOf(bools);
    NdArray<String> sMatrix = StdArrays.ndCopyOf(strings);

    try (TInt32 tints = TInt32.tensorOf(iMatrix);
        TFloat32 tfloats = TFloat32.tensorOf(fMatrix);
        TFloat64 tdoubles = TFloat64.tensorOf(dMatrix);
        TInt64 tlongs = TInt64.tensorOf(lMatrix);
        TBool tbools = TBool.tensorOf(bMatrix);
        TString tstrings = TString.tensorOf(sMatrix)) {

      TensorPrinter tp = new TensorPrinter(tints);
      assertEquals("[\n  [\n    [1]\n    [2]\n  ]\n  [\n    [3]\n    [4]\n  ]\n]", tp.print());

      tp = new TensorPrinter(tfloats);
      assertEquals(
          "[\n  [\n    [1.0]\n    [2.0]\n  ]\n  [\n    [3.0]\n    [4.0]\n  ]\n]", tp.print());

      tp = new TensorPrinter(tdoubles);
      assertEquals(
          "[\n  [\n    [1.0]\n    [2.0]\n  ]\n  [\n    [3.0]\n    [4.0]\n  ]\n]", tp.print());

      tp = new TensorPrinter(tlongs);
      assertEquals("[\n  [\n    [1]\n    [2]\n  ]\n  [\n    [3]\n    [4]\n  ]\n]", tp.print());

      tp = new TensorPrinter(tbools);
      assertEquals(
          "[\n  [\n    [true]\n    [false]\n  ]\n  [\n    [true]\n    [false]\n  ]\n]", tp.print());

      tp = new TensorPrinter(tstrings);
      assertEquals(
          "[\n  [\n    [\"A\"]\n    [\"B\"]\n  ]\n  [\n    [\"C\"]\n    [\"D\"]\n  ]\n]",
          tp.print());
    }
  }

  @Test
  public void testPrint2DBrace() {
    int[][] ints = {{1, 2}, {3, 4}};
    float[][] floats = {{1f, 2f}, {3f, 4f}};
    double[][] doubles = {{1d, 2d}, {3d, 4d}};
    long[][] longs = {{1L, 2L}, {3L, 4L}};
    boolean[][] bools = {{true, false}, {true, false}};
    String[][] strings = {{"A", "B"}, {"C", "D"}};

    IntNdArray iMatrix = StdArrays.ndCopyOf(ints);
    FloatNdArray fMatrix = StdArrays.ndCopyOf(floats);
    DoubleNdArray dMatrix = StdArrays.ndCopyOf(doubles);
    LongNdArray lMatrix = StdArrays.ndCopyOf(longs);
    BooleanNdArray bMatrix = StdArrays.ndCopyOf(bools);
    NdArray<String> sMatrix = StdArrays.ndCopyOf(strings);

    try (TInt32 tints = TInt32.tensorOf(iMatrix);
        TFloat32 tfloats = TFloat32.tensorOf(fMatrix);
        TFloat64 tdoubles = TFloat64.tensorOf(dMatrix);
        TInt64 tlongs = TInt64.tensorOf(lMatrix);
        TBool tbools = TBool.tensorOf(bMatrix);
        TString tstrings = TString.tensorOf(sMatrix)) {

      Options options = Options.create().encloseWithBraces();

      TensorPrinter tp = new TensorPrinter(tints, options);
      assertEquals("{\n  {1, 2}\n  {3, 4}\n}", tp.print());

      tp = new TensorPrinter(tfloats, options);
      assertEquals("{\n  {1.0, 2.0}\n  {3.0, 4.0}\n}", tp.print());

      tp = new TensorPrinter(tdoubles, options);
      assertEquals("{\n  {1.0, 2.0}\n  {3.0, 4.0}\n}", tp.print());

      tp = new TensorPrinter(tlongs, options);
      assertEquals("{\n  {1, 2}\n  {3, 4}\n}", tp.print());

      tp = new TensorPrinter(tbools, options);
      assertEquals("{\n  {true, false}\n  {true, false}\n}", tp.print());

      tp = new TensorPrinter(tstrings, options);
      assertEquals("{\n  {\"A\", \"B\"}\n  {\"C\", \"D\"}\n}", tp.print());
    }
  }

  @Test
  public void testPrint2DParen() {
    int[][] ints = {{1, 2}, {3, 4}};
    float[][] floats = {{1f, 2f}, {3f, 4f}};
    double[][] doubles = {{1d, 2d}, {3d, 4d}};
    long[][] longs = {{1L, 2L}, {3L, 4L}};
    boolean[][] bools = {{true, false}, {true, false}};
    String[][] strings = {{"A", "B"}, {"C", "D"}};

    IntNdArray iMatrix = StdArrays.ndCopyOf(ints);
    FloatNdArray fMatrix = StdArrays.ndCopyOf(floats);
    DoubleNdArray dMatrix = StdArrays.ndCopyOf(doubles);
    LongNdArray lMatrix = StdArrays.ndCopyOf(longs);
    BooleanNdArray bMatrix = StdArrays.ndCopyOf(bools);
    NdArray<String> sMatrix = StdArrays.ndCopyOf(strings);

    try (TInt32 tints = TInt32.tensorOf(iMatrix);
        TFloat32 tfloats = TFloat32.tensorOf(fMatrix);
        TFloat64 tdoubles = TFloat64.tensorOf(dMatrix);
        TInt64 tlongs = TInt64.tensorOf(lMatrix);
        TBool tbools = TBool.tensorOf(bMatrix);
        TString tstrings = TString.tensorOf(sMatrix)) {

      Options options = Options.create().encloseWithParens();

      TensorPrinter tp = new TensorPrinter(tints, options);
      assertEquals("(\n  (1, 2)\n  (3, 4)\n)", tp.print());

      tp = new TensorPrinter(tfloats, options);
      assertEquals("(\n  (1.0, 2.0)\n  (3.0, 4.0)\n)", tp.print());

      tp = new TensorPrinter(tdoubles, options);
      assertEquals("(\n  (1.0, 2.0)\n  (3.0, 4.0)\n)", tp.print());

      tp = new TensorPrinter(tlongs, options);
      assertEquals("(\n  (1, 2)\n  (3, 4)\n)", tp.print());

      tp = new TensorPrinter(tbools, options);
      assertEquals("(\n  (true, false)\n  (true, false)\n)", tp.print());

      tp = new TensorPrinter(tstrings, options);
      assertEquals("(\n  (\"A\", \"B\")\n  (\"C\", \"D\")\n)", tp.print());
    }
  }

  @Test
  public void testPrint2DtrailingSeparator() {
    int[][] ints = {{1, 2}, {3, 4}};
    float[][] floats = {{1f, 2f}, {3f, 4f}};
    double[][] doubles = {{1d, 2d}, {3d, 4d}};
    long[][] longs = {{1L, 2L}, {3L, 4L}};
    boolean[][] bools = {{true, false}, {true, false}};
    String[][] strings = {{"A", "B"}, {"C", "D"}};

    IntNdArray iMatrix = StdArrays.ndCopyOf(ints);
    FloatNdArray fMatrix = StdArrays.ndCopyOf(floats);
    DoubleNdArray dMatrix = StdArrays.ndCopyOf(doubles);
    LongNdArray lMatrix = StdArrays.ndCopyOf(longs);
    BooleanNdArray bMatrix = StdArrays.ndCopyOf(bools);
    NdArray<String> sMatrix = StdArrays.ndCopyOf(strings);

    try (TInt32 tints = TInt32.tensorOf(iMatrix);
        TFloat32 tfloats = TFloat32.tensorOf(fMatrix);
        TFloat64 tdoubles = TFloat64.tensorOf(dMatrix);
        TInt64 tlongs = TInt64.tensorOf(lMatrix);
        TBool tbools = TBool.tensorOf(bMatrix);
        TString tstrings = TString.tensorOf(sMatrix)) {

      Options options = Options.create().trailingSeparator(true);

      TensorPrinter tp = new TensorPrinter(tints, options);
      assertEquals("[\n  [1, 2],\n  [3, 4]\n]", tp.print());

      tp = new TensorPrinter(tfloats, options);
      assertEquals("[\n  [1.0, 2.0],\n  [3.0, 4.0]\n]", tp.print());

      tp = new TensorPrinter(tdoubles, options);
      assertEquals("[\n  [1.0, 2.0],\n  [3.0, 4.0]\n]", tp.print());

      tp = new TensorPrinter(tlongs, options);
      assertEquals("[\n  [1, 2],\n  [3, 4]\n]", tp.print());

      tp = new TensorPrinter(tbools, options);
      assertEquals("[\n  [true, false],\n  [true, false]\n]", tp.print());

      tp = new TensorPrinter(tstrings, options);
      assertEquals("[\n  [\"A\", \"B\"],\n  [\"C\", \"D\"]\n]", tp.print());
    }
  }

  @Test
  public void testPrint2DtrailingIndent() {
    int[][] ints = {{1, 2}, {3, 4}};
    float[][] floats = {{1f, 2f}, {3f, 4f}};
    double[][] doubles = {{1d, 2d}, {3d, 4d}};
    long[][] longs = {{1L, 2L}, {3L, 4L}};
    boolean[][] bools = {{true, false}, {true, false}};
    String[][] strings = {{"A", "B"}, {"C", "D"}};

    IntNdArray iMatrix = StdArrays.ndCopyOf(ints);
    FloatNdArray fMatrix = StdArrays.ndCopyOf(floats);
    DoubleNdArray dMatrix = StdArrays.ndCopyOf(doubles);
    LongNdArray lMatrix = StdArrays.ndCopyOf(longs);
    BooleanNdArray bMatrix = StdArrays.ndCopyOf(bools);
    NdArray<String> sMatrix = StdArrays.ndCopyOf(strings);

    try (TInt32 tints = TInt32.tensorOf(iMatrix);
        TFloat32 tfloats = TFloat32.tensorOf(fMatrix);
        TFloat64 tdoubles = TFloat64.tensorOf(dMatrix);
        TInt64 tlongs = TInt64.tensorOf(lMatrix);
        TBool tbools = TBool.tensorOf(bMatrix);
        TString tstrings = TString.tensorOf(sMatrix)) {

      Options options = Options.create().indentSize(1);

      TensorPrinter tp = new TensorPrinter(tints, options);
      assertEquals("[\n [1, 2]\n [3, 4]\n]", tp.print());

      tp = new TensorPrinter(tfloats, options);
      assertEquals("[\n [1.0, 2.0]\n [3.0, 4.0]\n]", tp.print());

      tp = new TensorPrinter(tdoubles, options);
      assertEquals("[\n [1.0, 2.0]\n [3.0, 4.0]\n]", tp.print());

      tp = new TensorPrinter(tlongs, options);
      assertEquals("[\n [1, 2]\n [3, 4]\n]", tp.print());

      tp = new TensorPrinter(tbools, options);
      assertEquals("[\n [true, false]\n [true, false]\n]", tp.print());

      tp = new TensorPrinter(tstrings, options);
      assertEquals("[\n [\"A\", \"B\"]\n [\"C\", \"D\"]\n]", tp.print());
    }
  }
}
