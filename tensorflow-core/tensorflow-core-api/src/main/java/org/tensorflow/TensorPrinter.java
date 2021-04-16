/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/
package org.tensorflow;

import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.proto.framework.DataType;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.StringJoiner;

/** Utility class to print the contents of a Tensor */
public class TensorPrinter {

  private final Tensor tensor;
  private Options options;

  private String endOfLineSep;
  private String separatorString;
  private String openSet, closeSet;
  private String indentString = "  ";

  /**
   * Creates a TensorPrinter
   *
   * @param tensor the tensor
   */
  public TensorPrinter(Tensor tensor) {
    this(tensor, Options.create());
  }

  /**
   * Creates a TensorPrinter
   *
   * @param options the {@link Options} for formatting the print
   * @param tensor the tensor
   */
  public TensorPrinter(Tensor tensor, Options options) {
    this.tensor = tensor;
    setOptions(options);
  }

  /**
   * Gets a printable String for the Tensor data
   *
   * @return a printable String for the Tensor data
   */
  public String print() {
    Tensor tmpTensor = tensor;
    if (tmpTensor instanceof RawTensor) {
      tmpTensor = ((RawTensor) tensor).asTypedTensor();
    }
    if (!(tmpTensor instanceof NdArray<?>)) {
      // TODO can this ever happen?
      return dumpRawTensor(tmpTensor.asRawTensor());
    }

    NdArray<?> ndArray = (NdArray<?>) tmpTensor;
    Iterator<? extends NdArray<?>> iterator = ndArray.scalars().iterator();
    Shape shape = tmpTensor.shape();
    if (shape.numDimensions() == 0) {
      if (!iterator.hasNext()) {
        return "";
      }
      return String.valueOf(iterator.next().getObject());
    }
    return formatString(iterator, tmpTensor.dataType(), shape, 0);
  }

  /**
   * Prints the raw tensor in cases where a Tensor does not inherit from NDArray.
   *
   * @param tensor the tensor.
   * @return the printable raw tensor.
   */
  private String dumpRawTensor(RawTensor tensor) {
    StringBuilder sb = new StringBuilder();
    sb.append("actual  : ").append(tensor).append('\n');
    sb.append("dataType  : ").append(tensor.dataType()).append('\n');
    sb.append("class  : ").append(tensor.getClass()).append('\n');
    sb.append('\n');
    return sb.toString();
  }

  /**
   * @param iterator an iterator over the scalars
   * @param dataType the data type for the tensor
   * @param shape the shape of the tensor
   * @param dimension the current dimension being processed
   * @return the String representation of the tensor data at {@code dimension}
   */
  private String formatString(
      Iterator<? extends NdArray<?>> iterator, DataType dataType, Shape shape, int dimension) {

    if (dimension < shape.numDimensions() - 1) {
      StringJoiner joiner =
          new StringJoiner(
              endOfLineSep + "\n",
              indent(dimension) + openSet + "\n",
              "\n" + indent(dimension) + closeSet);
      for (long i = 0, size = shape.size(dimension); i < size; ++i) {
        String element = formatString(iterator, dataType, shape, dimension + 1);
        joiner.add(element);
      }
      return joiner.toString();
    }
    if (options.maxWidth == null) {
      StringJoiner joiner =
          new StringJoiner(separatorString, indent(dimension) + openSet, closeSet);
      for (long i = 0, size = shape.size(dimension); i < size; ++i) {
        Object element = iterator.next().getObject();
        joiner.add(elementToString(dataType, element));
      }

      return joiner.toString();
    }
    List<Integer> lengths = new ArrayList<>();
    StringJoiner joiner = new StringJoiner(separatorString, indent(dimension) + openSet, closeSet);
    int lengthBefore = closeSet.length();
    for (long i = 0, size = shape.size(dimension); i < size; ++i) {
      Object element = iterator.next().getObject();
      joiner.add(elementToString(dataType, element));
      int addedLength = joiner.length() - lengthBefore;
      lengths.add(addedLength);
      lengthBefore += addedLength;
    }
    return truncateWidth(joiner.toString(), options.maxWidth, lengths);
  }

  /**
   * Convert an element of a tensor to string, in a way that may depend on the data type.
   *
   * @param dataType the tensor's data type
   * @param data the element
   * @return the element's string representation
   */
  private String elementToString(DataType dataType, Object data) {
    if (dataType == DataType.DT_STRING) {
      return '"' + data.toString() + '"';
    } else if (options.numDecimals != null
        && (dataType == DataType.DT_DOUBLE || dataType == DataType.DT_FLOAT)) {
      String format = "%." + options.numDecimals + "f";
      return String.format(format, data);
    } else {
      return data.toString();
    }
  }

  /**
   * Truncates the width of a String if it's too long, inserting "{@code ...}" in place of the
   * removed data.
   *
   * @param input the input to truncate
   * @param maxWidth the maximum width of the output in characters
   * @param lengths the lengths of elements inside input
   * @return the (potentially) truncated output
   */
  private String truncateWidth(String input, int maxWidth, List<Integer> lengths) {
    if (input.length() <= maxWidth) {
      return input;
    }
    StringBuilder output = new StringBuilder(input);
    int midPoint = (maxWidth / 2) - 1;
    int width = 0;
    int indexOfElementToRemove = lengths.size() - 1;
    int widthBeforeElementToRemove = 0;
    for (int i = 0, size = lengths.size(); i < size; ++i) {
      width += lengths.get(i);
      if (width > midPoint) {
        indexOfElementToRemove = i;
        break;
      }
      widthBeforeElementToRemove = width;
    }
    if (indexOfElementToRemove == 0) {
      // Cannot remove first element
      return input;
    }
    output.insert(widthBeforeElementToRemove, separatorString + "...");
    widthBeforeElementToRemove += (separatorString + "...").length();
    width = output.length();
    while (width > maxWidth) {
      if (indexOfElementToRemove == 0) {
        // Cannot remove first element
        break;
      } else if (indexOfElementToRemove == lengths.size() - 1) {
        // Cannot remove last element
        --indexOfElementToRemove;
        continue;
      }
      Integer length = lengths.remove(indexOfElementToRemove);
      output.delete(widthBeforeElementToRemove, widthBeforeElementToRemove + length);
      width = output.length();
    }
    if (output.length() < input.length()) {
      return output.toString();
    }
    // Do not insert ellipses if it increases the length
    return input;
  }

  /**
   * Gets the indent string based on the indent level
   *
   * @param level the level of indent
   * @return the indentation string
   */
  private String indent(int level) {
    if (level <= 0) {
      return "";
    }
    StringBuilder result = new StringBuilder(level * 2);
    for (int i = 0; i < level; ++i) {
      result.append(indentString);
    }
    return result.toString();
  }

  /**
   * Gets the tensor
   *
   * @return the tensor
   */
  public Tensor getTensor() {
    return tensor;
  }

  /**
   * Gets the Options
   *
   * @return the Options
   */
  public Options getOptions() {
    return options;
  }

  /**
   * Sets the options for formatting the print string.
   *
   * @param options the options
   */
  public final void setOptions(Options options) {
    this.options = options == null ? Options.create() : options;

    switch (this.options.enclosure) {
      case BRACES:
        openSet = "{";
        closeSet = "}";
        break;

      case PARENS:
        openSet = "(";
        closeSet = ")";
        break;
      case BRACKETS:
      default:
        openSet = "[";
        closeSet = "]";
        break;
    }
    endOfLineSep = this.options.trailingSeparator ? String.valueOf(this.options.separator) : "";
    separatorString = String.format("%c ", this.options.separator);
    if (this.options.indentSize != null) {
      String format = "%" + this.options.indentSize + "s";
      indentString = String.format(format, "");
    }
  }

  /** Contains the options for TensorPrint */
  public static class Options {
    public static char DEFAULT_SEPARATOR = ',';
    public static Enclosure DEFAULT_ENCLOSURE = Enclosure.BRACKETS;

    /** The max width of a single line */
    public Integer maxWidth;
    /** The element separator character, default is {@link #DEFAULT_SEPARATOR} */
    public char separator = DEFAULT_SEPARATOR;
    /**
     * the number of digits after the decimal point for floating point numbers, null means to use
     * the default format
     */
    public Integer numDecimals;

    /** The number of spaces for each indent space */
    public Integer indentSize;

    /**
     * Indicator whether a trailing separator is present at the end of an inner set, default is
     * false.
     */
    public boolean trailingSeparator;

    /** The set of characters that enclose sets, default is {@link #DEFAULT_ENCLOSURE} */
    public Enclosure enclosure = DEFAULT_ENCLOSURE;

    /** Creates an Options instance. */
    Options() {}

    /**
     * Creates an Options instance.
     *
     * @return this Options instance.
     */
    public static Options create() {
      return new Options();
    }

    /**
     * Sets the maxWidth property
     *
     * @param maxWidth the maximum width of a line
     * @return this Options instance.
     */
    public Options maxWidth(int maxWidth) {
      this.maxWidth = maxWidth;
      return this;
    }

    /**
     * Sets the maxWidth property
     *
     * @param separator the item separator character, this is the character to separate each
     *     element. Default is {@link #DEFAULT_SEPARATOR}.
     * @return this Options instance.
     */
    public Options separator(char separator) {
      this.separator = separator;
      return this;
    }

    /**
     * Sets the enclosure to {@link Enclosure#BRACES} {@code {}}.
     *
     * @return this Options instance.
     */
    public Options encloseWithBraces() {
      this.enclosure = Enclosure.BRACES;
      return this;
    }

    /**
     * Sets the enclosure to {@link Enclosure#BRACKETS}, {@code []}.
     *
     * @return this Options instance.
     */
    public Options encloseWithBrackets() {
      this.enclosure = Enclosure.BRACKETS;
      return this;
    }

    /**
     * Sets the enclosure to {@link Enclosure#PARENS}, {@code ()}.
     *
     * @return this Options instance.
     */
    public Options encloseWithParens() {
      this.enclosure = Enclosure.PARENS;
      return this;
    }

    /**
     * Sets the trailingSeparator property.
     *
     * @param trailingSeparator whether or not to add the separator after inner sets.
     * @return this Options instance.
     */
    public Options trailingSeparator(boolean trailingSeparator) {
      this.trailingSeparator = trailingSeparator;
      return this;
    }

    /**
     * Sets the number of digits after the decimal point. Only applies to floating point data types.
     *
     * <p>The default is to use either {@link String#valueOf(float)} or {@link
     * String#valueOf(double)}
     *
     * @param numDecimals the number of digits after the decimal point.
     * @return this Options instance.
     */
    public Options numDecimals(int numDecimals) {
      this.numDecimals = numDecimals;
      return this;
    }

    /**
     * Sets the number of spaces for the indent
     *
     * @param indentSize the number of spaces for the indent
     * @return this Options instance.
     */
    public Options indentSize(int indentSize) {
      this.indentSize = indentSize;
      return this;
    }

    /** Enumerator for specifying the character pairs for enclosing sets. */
    public enum Enclosure {
      /** Enclose the set in brackets, {@code []} */
      BRACKETS,
      /** Enclose the set in curly braces, {@code {}} */
      BRACES,
      /** Enclose the set in parenthesis, {@code ()} */
      PARENS
    }
  }
}
