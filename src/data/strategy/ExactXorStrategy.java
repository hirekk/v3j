/*
 * Copyright (c) 2025 Hieronim Kubica. Licensed under the MIT License. See LICENSE file for full
 * terms.
 */

package data.strategy;

import data.DataPoint;
import java.util.ArrayList;
import java.util.List;

/**
 * Strategy for generating exact XOR-style datasets representing hypercube vertices.
 *
 * <p>This strategy generates datasets where each data point represents a vertex of an N-dimensional
 * hypercube. The coordinates are binary (0.0 or 1.0), and the label is determined by the parity of
 * 1s in the coordinate vector - odd count of 1s results in label 1, even count results in label 0.
 *
 * <p>For example, in 2D this is the classic XOR dataset:
 *
 * <ul>
 *   <li>Vertex (0,0) has 0 ones → label 0
 *   <li>Vertex (0,1) has 1 one → label 1
 *   <li>Vertex (1,0) has 1 one → label 1
 *   <li>Vertex (1,1) has 2 ones → label 0
 * </ul>
 *
 * <p>This implements the XOR classification dataset generalized to N dimensions, where the label
 * represents whether the number of "active" inputs (value 1.0) is odd or even. The strategy is
 * deterministic and produces exactly 2^N data points for N dimensions.
 *
 * <p>The class is immutable and thread-safe.
 *
 * @see GenerationStrategy
 * @see DataPoint
 * @see FuzzyXorStrategy
 */
public final class ExactXorStrategy implements GenerationStrategy {

  /**
   * Generates a dataset of hypercube vertices with binary coordinates.
   *
   * <p>Creates exactly 2^numDimensions data points, each representing a vertex of the N-dimensional
   * hypercube. Each vertex has coordinates that are either 0.0 or 1.0, and the label is determined
   * by the parity of 1s in the coordinate vector.
   *
   * <p>The vertices are generated in order from (0,0,...,0) to (1,1,...,1), with each coordinate
   * bit representing one dimension.
   *
   * @param numDimensions the number of dimensions for the hypercube
   * @return a list of DataPoint objects representing hypercube vertices
   * @throws IllegalArgumentException if numDimensions is negative
   */
  @Override
  public List<DataPoint> generateDataset(int numDimensions) {
    if (numDimensions < 0) {
      throw new IllegalArgumentException(
          "Number of dimensions must be non-negative, got: " + numDimensions);
    }

    int numVertices = 1 << numDimensions; // 2^numDimensions
    List<DataPoint> dataset = new ArrayList<>(numVertices);

    for (int i = 0; i < numVertices; i++) {
      double[] coordinates = vertexCoordinatesFromIndex(i, numDimensions);
      int label = calculateVertexParity(coordinates);
      dataset.add(new DataPoint(coordinates, label));
    }

    return dataset;
  }

  /**
   * Converts a vertex index to its binary coordinate representation.
   *
   * <p>Uses bit manipulation to efficiently convert the integer index to binary coordinates. Each
   * bit position corresponds to a dimension: bit 0 → dimension 0, bit 1 → dimension 1, etc.
   *
   * <p>For example, index 5 (binary 101) in 3D produces coordinates [1,0,1].
   *
   * @param vertexIndex the vertex index (0 to 2^numDimensions - 1)
   * @param numDimensions the number of dimensions
   * @return array of coordinates where each element is 0.0 or 1.0
   */
  private double[] vertexCoordinatesFromIndex(int vertexIndex, int numDimensions) {
    double[] coordinates = new double[numDimensions];
    for (int i = 0; i < numDimensions; i++) {
      // Extract bit i from vertexIndex: (vertexIndex >> i) & 1
      // If bit is 1, coordinate is 1.0; if bit is 0, coordinate is 0.0
      coordinates[i] = ((vertexIndex >> i) & 1) == 1 ? 1.0 : 0.0;
    }
    return coordinates;
  }

  /**
   * Calculates the parity (odd/even count) of 1s in the coordinate vector.
   *
   * <p>Counts the number of coordinates with value 1.0 and returns 1 if the count is odd, 0 if the
   * count is even. This implements the XOR labeling logic where odd numbers of active inputs
   * produce label 1.
   *
   * <p>For example:
   *
   * <ul>
   *   <li>[0,0,0] has 0 ones → returns 0 (even)
   *   <li>[1,0,0] has 1 one → returns 1 (odd)
   *   <li>[1,1,0] has 2 ones → returns 0 (even)
   *   <li>[1,1,1] has 3 ones → returns 1 (odd)
   * </ul>
   *
   * @param coordinates the coordinate array to analyze
   * @return 1 if the count of 1s is odd, 0 if even
   */
  private int calculateVertexParity(double[] coordinates) {
    int count = 0;
    for (double val : coordinates) {
      if (val == 1.0) {
        count++;
      }
    }
    return (count % 2 == 1) ? 1 : 0;
  }
}
