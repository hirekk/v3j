/*
 * Copyright (c) 2025 Hieronim Kubica. Licensed under the MIT License. See LICENSE file for full
 * terms.
 */

package data.strategy;

import data.DataPoint;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Strategy for generating fuzzy XOR-style datasets with Gaussian noise around hypercube vertices.
 *
 * <p>This strategy extends the exact XOR concept by adding controlled randomness around each
 * hypercube vertex. Instead of generating only the exact vertices, it creates "clouds" of points
 * around each vertex using Gaussian noise with specified variance.
 *
 * <p>For each vertex of the N-dimensional hypercube, this strategy generates a configurable number
 * of points (blobCardinality) that are distributed around the vertex according to a multivariate
 * Gaussian distribution. The variance parameter controls how spread out these point clouds are -
 * higher variance creates wider, more diffuse clouds.
 *
 * <p>Key characteristics:
 *
 * <ul>
 *   <li><strong>Deterministic labeling</strong> - All points around a vertex inherit the same label
 *       (determined by vertex coordinate parity)
 *   <li><strong>Configurable noise</strong> - Each dimension can have different variance for
 *       anisotropic point cloud shapes
 *   <li><strong>Reproducible randomness</strong> - Seeded random generation ensures consistent
 *       results across runs
 *   <li><strong>Scalable generation</strong> - Dataset size scales as 2^N × cardinality
 * </ul>
 *
 * <p>This strategy is useful for creating training datasets that simulate real-world scenarios
 * where exact mathematical relationships are obscured by measurement noise or natural variation.
 *
 * <p>The class is immutable and thread-safe.
 *
 * @see GenerationStrategy
 * @see ExactXorStrategy
 * @see DataPoint
 */
public final class FuzzyXorStrategy implements GenerationStrategy {

  /** Number of points to generate around each hypercube vertex. */
  private final int blobCardinality;

  /** Variance for Gaussian noise in each dimension. */
  private final double[] blobVariance;

  /** Random number generator for noise generation. */
  private final Random random;

  /**
   * Constructs a fuzzy XOR strategy with default random seed.
   *
   * <p>Uses the current system time as the random seed, which means each instance will produce
   * different results. For reproducible results, use the constructor with an explicit seed.
   *
   * @param blobCardinality the number of points to generate around each vertex
   * @param blobVariance the variance for Gaussian noise in each dimension
   * @throws IllegalArgumentException if blobCardinality is not positive or if blobVariance contains
   *     negative values
   */
  public FuzzyXorStrategy(int blobCardinality, double[] blobVariance) {
    this(blobCardinality, blobVariance, System.currentTimeMillis());
  }

  /**
   * Constructs a fuzzy XOR strategy with a specified random seed.
   *
   * <p>Using the same seed will produce identical results across multiple runs, making this useful
   * for testing and reproducible experiments.
   *
   * @param blobCardinality the number of points to generate around each vertex
   * @param blobVariance the variance for Gaussian noise in each dimension
   * @param seed the random seed for reproducible noise generation
   * @throws IllegalArgumentException if blobCardinality is not positive or if blobVariance contains
   *     negative values
   */
  public FuzzyXorStrategy(int blobCardinality, double[] blobVariance, long seed) {
    validateBlobCardinality(blobCardinality);
    validateBlobVariance(blobVariance);

    this.blobCardinality = blobCardinality;
    this.blobVariance = Arrays.copyOf(blobVariance, blobVariance.length);
    this.random = new Random(seed);
  }

  /**
   * Validates that the blob cardinality is a positive integer.
   *
   * @param value the cardinality value to validate
   * @throws IllegalArgumentException if the value is not positive
   */
  private void validateBlobCardinality(int value) {
    if (value <= 0) {
      String errorMsg = String.format("Blob cardinality must be a positive integer; got %s", value);
      throw new IllegalArgumentException(errorMsg);
    }
  }

  /**
   * Validates that the blob variance array is valid.
   *
   * <p>Checks that the array is not null, not empty, and contains only non-negative values.
   *
   * @param value the variance array to validate
   * @throws IllegalArgumentException if the array is null, empty, or contains negative values
   */
  private void validateBlobVariance(double[] value) {
    if (value == null) {
      throw new IllegalArgumentException("Blob variance must not be null");
    }
    if (value.length == 0) {
      throw new IllegalArgumentException("Blob variance must not be empty");
    }

    for (int i = 0; i < value.length; i++) {
      if (value[i] < 0) {
        String errorMsg =
            String.format("Blob variance must be an array of non-negative reals; got %f", value[i]);
        throw new IllegalArgumentException(errorMsg);
      }
    }
  }

  /**
   * Generates a fuzzy XOR dataset with Gaussian noise around hypercube vertices.
   *
   * <p>Creates a dataset by first generating the exact hypercube vertices using {@link
   * ExactXorStrategy}, then adding Gaussian noise around each vertex to create point clouds. The
   * final dataset size is 2^numDimensions × blobCardinality.
   *
   * <p>Each point in the dataset inherits its label from the vertex it was generated around,
   * maintaining the XOR labeling logic while adding realistic noise.
   *
   * @param numDimensions the number of dimensions for the hypercube
   * @return a list of DataPoint objects with Gaussian noise around vertices
   * @throws IllegalArgumentException if numDimensions is negative or if there's a dimension
   *     mismatch between mean and variance arrays
   */
  @Override
  public List<DataPoint> generateDataset(int numDimensions) {
    if (numDimensions < 0) {
      throw new IllegalArgumentException(
          "Number of dimensions must be non-negative, got: " + numDimensions);
    }

    ExactXorStrategy exactStrategy = new ExactXorStrategy();
    List<DataPoint> hypercubeVertices = exactStrategy.generateDataset(numDimensions);

    int numVertices = hypercubeVertices.size();
    int datasetSize = numVertices * blobCardinality;
    List<DataPoint> dataset = new ArrayList<>(datasetSize);

    for (DataPoint vertex : hypercubeVertices) {
      double[][] pointCloud = createPointCloud(vertex.coordinates(), blobVariance, blobCardinality);
      for (double[] point : pointCloud) {
        dataset.add(new DataPoint(point, vertex.label()));
      }
    }

    return dataset;
  }

  /**
   * Creates a point cloud around a given mean point using Gaussian noise.
   *
   * <p>Generates multiple points around the specified mean coordinates, with each dimension
   * perturbed according to its corresponding variance. The resulting points form a cloud that
   * approximates a multivariate Gaussian distribution centered at the mean.
   *
   * @param mean the center coordinates around which to generate the point cloud
   * @param variance the variance for noise in each dimension
   * @param cardinality the number of points to generate in the cloud
   * @return a 2D array where each row represents a point's coordinates
   * @throws IllegalArgumentException if the mean and variance arrays have different dimensions
   */
  private double[][] createPointCloud(double[] mean, double[] variance, int cardinality) {
    if (mean.length != variance.length) {
      String errorMsg =
          String.format(
              "Dimensionality of mean and variance must be equal; got dim(mean) = %d and dim(variance) = %d",
              mean.length, variance.length);
      throw new IllegalArgumentException(errorMsg);
    }
    int numDimensions = mean.length;
    double[][] pointCloud = new double[cardinality][numDimensions];
    for (int i = 0; i < cardinality; i++) {
      pointCloud[i] = addGaussianNoise(mean, variance);
    }
    return pointCloud;
  }

  /**
   * Adds Gaussian noise to a point according to specified variance.
   *
   * <p>Perturbs each coordinate of the input point by sampling from a Gaussian distribution with
   * the point's coordinate as the mean and the square root of the corresponding variance as the
   * standard deviation.
   *
   * <p>This method uses Java's {@link Random#nextGaussian(double, double)} for efficient Gaussian
   * sampling, which is available in Java 17+.
   *
   * @param point the original point coordinates
   * @param variance the variance for noise in each dimension
   * @return a new array with perturbed coordinates
   */
  private double[] addGaussianNoise(double[] point, double[] variance) {
    double[] perturbedPoint = new double[point.length];
    for (int i = 0; i < perturbedPoint.length; i++) {
      // Generate Gaussian noise: N(point[i], sqrt(variance[i]))
      perturbedPoint[i] = random.nextGaussian(point[i], Math.sqrt(variance[i]));
    }
    return perturbedPoint;
  }
}
