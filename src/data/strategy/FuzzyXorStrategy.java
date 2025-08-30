package data.strategy;

import data.DataPoint;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public final class FuzzyXorStrategy implements GenerationStrategy {
  private final int blobCardinality;
  private final double[] blobVariance;
  private final Random random;

  public FuzzyXorStrategy(int blobCardinality, double[] blobVariance) {
    this(blobCardinality, blobVariance, System.currentTimeMillis());
  }

  public FuzzyXorStrategy(int blobCardinality, double[] blobVariance, long seed) {
    validateBlobCardinality(blobCardinality);
    validateBlobVariance(blobVariance);

    this.blobCardinality = blobCardinality;
    this.blobVariance = Arrays.copyOf(blobVariance, blobVariance.length);
    this.random = new Random(seed);
  }

  private void validateBlobCardinality(int value) {
    if (value <= 0) {
      String errorMsg = String.format("Blob cardinality must be a positive integer; got %s", value);
      throw new IllegalArgumentException(errorMsg);
    }
  }

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
            String.format("Blob variance must be an array of non-negative reals; got %f", value);
        throw new IllegalArgumentException(errorMsg);
      }
    }
  }

  @Override
  public List<DataPoint> generateDataset(int numDimensions) {
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

  private double[] addGaussianNoise(double[] point, double[] variance) {
    double[] perturbedPoint = new double[point.length];
    for (int i = 0; i < perturbedPoint.length; i++) {
      perturbedPoint[i] = random.nextGaussian(point[i], Math.sqrt(variance[i]));
    }
    return perturbedPoint;
  }
}
