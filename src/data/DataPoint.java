package data;

/** Represents an N-dimensional data point and a binary label. */
public record DataPoint(double[] coordinates, int label) {

  /** Constructor that validates the inputs array */
  public DataPoint {
    if (coordinates == null || coordinates.length == 0) {
      throw new IllegalArgumentException("Coordinates array cannot be null or empty");
    }
  }

  /** Gets the dimensionality of the data point */
  public int getNumDimensions() {
    return coordinates.length;
  }

  /** Gets a specific coordinate value */
  public double getCoordinate(int index) {
    if (index < 0 || index >= coordinates.length) {
      throw new IndexOutOfBoundsException(
          "Index " + index + " out of bounds for data point of dimension " + getNumDimensions());
    }
    return coordinates[index];
  }
}
