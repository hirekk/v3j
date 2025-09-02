/*
 * Copyright (c) 2025 Hieronim Kubica
 * Licensed under the MIT License. See LICENSE file for full terms.
 */
package data;

/**
 * Represents an N-dimensional data point with a binary label.
 *
 * <p>This record provides an immutable representation of a data point suitable for machine learning
 * datasets. The coordinates array represents the input features, while the label represents the
 * binary classification (0 or 1).
 *
 * <p>The record automatically validates that coordinates are not null or empty during construction.
 *
 * @param coordinates the N-dimensional coordinate array (must not be null or empty)
 * @param label the binary classification label (0 or 1)
 * @throws IllegalArgumentException if coordinates is null or empty
 */
public record DataPoint(double[] coordinates, int label) {

  /**
   * Canonical constructor that validates the coordinates array.
   *
   * @param coordinates the coordinate array to validate
   * @param label the binary label
   * @throws IllegalArgumentException if coordinates is null or empty
   */
  public DataPoint {
    if (coordinates == null || coordinates.length == 0) {
      throw new IllegalArgumentException("Coordinates array cannot be null or empty");
    }
  }

  /**
   * Returns the dimensionality of this data point.
   *
   * @return the number of dimensions (length of coordinates array)
   */
  public int getNumDimensions() {
    return coordinates.length;
  }

  /**
   * Retrieves the coordinate value at the specified index.
   *
   * @param index the index of the coordinate to retrieve
   * @return the coordinate value at the specified index
   * @throws IndexOutOfBoundsException if index is negative or >= number of dimensions
   */
  public double getCoordinate(int index) {
    if (index < 0 || index >= coordinates.length) {
      throw new IndexOutOfBoundsException(
          "Index " + index + " out of bounds for data point of dimension " + getNumDimensions());
    }
    return coordinates[index];
  }
}
