package data.strategy;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import data.DataPoint;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

@DisplayName("FuzzyXorStrategy Tests")
class FuzzyXorStrategyTest {

  private FuzzyXorStrategy strategy;
  private static final double EPSILON = 1e-10;

  @BeforeEach
  void setUp() {
    strategy = new FuzzyXorStrategy(10, new double[] {0.15, 0.15, 0.15}, 42L);
  }

  @Nested
  @DisplayName("Construction Tests")
  class ConstructionTests {

    @Test
    @DisplayName("Valid construction with positive cardinality and variance")
    void testValidConstruction() {
      FuzzyXorStrategy validStrategy = new FuzzyXorStrategy(5, new double[] {0.1, 0.2});
      assertNotNull(validStrategy);
    }

    @Test
    @DisplayName("Construction with seed for reproducibility")
    void testConstructionWithSeed() {
      FuzzyXorStrategy strategy1 = new FuzzyXorStrategy(5, new double[] {0.1, 0.2}, 123L);
      FuzzyXorStrategy strategy2 = new FuzzyXorStrategy(5, new double[] {0.1, 0.2}, 123L);

      // Same seed should produce same results
      List<DataPoint> dataset1 = strategy1.generateDataset(2);
      List<DataPoint> dataset2 = strategy2.generateDataset(2);

      assertEquals(dataset1.size(), dataset2.size());
      for (int i = 0; i < dataset1.size(); i++) {
        DataPoint p1 = dataset1.get(i);
        DataPoint p2 = dataset2.get(i);
        assertEquals(p1.label(), p2.label());
        for (int j = 0; j < p1.getNumDimensions(); j++) {
          assertEquals(p1.getCoordinate(j), p2.getCoordinate(j), EPSILON);
        }
      }
    }

    @Test
    @DisplayName("Invalid blob cardinality throws exception")
    void testInvalidBlobCardinality() {
      assertThrows(
          IllegalArgumentException.class, () -> new FuzzyXorStrategy(0, new double[] {0.1, 0.2}));

      assertThrows(
          IllegalArgumentException.class, () -> new FuzzyXorStrategy(-1, new double[] {0.1, 0.2}));
    }

    @Test
    @DisplayName("Null variance array throws exception")
    void testNullVarianceArray() {
      assertThrows(IllegalArgumentException.class, () -> new FuzzyXorStrategy(5, null));
    }

    @Test
    @DisplayName("Empty variance array throws exception")
    void testEmptyVarianceArray() {
      assertThrows(IllegalArgumentException.class, () -> new FuzzyXorStrategy(5, new double[] {}));
    }

    @Test
    @DisplayName("Negative variance values throw exception")
    void testNegativeVarianceValues() {
      assertThrows(
          IllegalArgumentException.class, () -> new FuzzyXorStrategy(5, new double[] {0.1, -0.2}));

      assertThrows(
          IllegalArgumentException.class, () -> new FuzzyXorStrategy(5, new double[] {-0.1, 0.2}));
    }
  }

  @Nested
  @DisplayName("Dataset Generation Tests")
  class DatasetGenerationTests {

    @Test
    @DisplayName("Generate dataset with correct size")
    void testGenerateDatasetSize() {
      List<DataPoint> dataset = strategy.generateDataset(3);

      // 3D hypercube has 2³ = 8 vertices, each with 10 points
      int expectedSize = 8 * 10;
      assertEquals(expectedSize, dataset.size());
    }

    @Test
    @DisplayName("Generate dataset with correct dimensionality")
    void testGenerateDatasetDimensionality() {
      // Create a strategy with 4D variance array for 4D dataset
      FuzzyXorStrategy fourDimStrategy =
          new FuzzyXorStrategy(10, new double[] {0.15, 0.15, 0.15, 0.15}, 42L);
      List<DataPoint> dataset = fourDimStrategy.generateDataset(4);

      for (DataPoint point : dataset) {
        assertEquals(4, point.getNumDimensions());
      }
    }

    @Test
    @DisplayName("All points have valid labels (0 or 1)")
    void testGenerateDatasetValidLabels() {
      // Create a strategy with 2D variance array for 2D dataset
      FuzzyXorStrategy twoDimStrategy = new FuzzyXorStrategy(10, new double[] {0.15, 0.15}, 42L);
      List<DataPoint> dataset = twoDimStrategy.generateDataset(2);

      for (DataPoint point : dataset) {
        int label = point.label();
        assertTrue(label == 0 || label == 1, "Label should be 0 or 1, got: " + label);
      }
    }

    @Test
    @DisplayName("Points are properly distributed around vertices")
    void testPointsDistributedAroundVertices() {
      // Create a strategy with 2D variance array for 2D dataset
      FuzzyXorStrategy twoDimStrategy = new FuzzyXorStrategy(10, new double[] {0.15, 0.15}, 42L);
      List<DataPoint> dataset = twoDimStrategy.generateDataset(2);

      // 2D hypercube has 4 vertices: (0,0), (0,1), (1,0), (1,1)
      // Each vertex should have 10 points with variance 0.15
      double expectedStd = Math.sqrt(0.15);

      // Check that points around (0,0) are within reasonable bounds
      boolean foundPointNearOrigin = false;
      for (DataPoint point : dataset) {
        if (point.label() == 0) { // Points around (0,0) and (1,1) have label 0
          double distFromOrigin =
              Math.sqrt(
                  point.getCoordinate(0) * point.getCoordinate(0)
                      + point.getCoordinate(1) * point.getCoordinate(1));

          // Points should be within 3 standard deviations of origin
          if (distFromOrigin < 3 * expectedStd) {
            foundPointNearOrigin = true;
            break;
          }
        }
      }

      assertTrue(foundPointNearOrigin, "Should find points near origin within 3σ");
    }
  }

  @Nested
  @DisplayName("Noise Distribution Tests")
  class NoiseDistributionTests {

    @Test
    @DisplayName("Generated noise has approximately correct variance")
    void testNoiseVariance() {
      // Use a larger cardinality for better statistical accuracy
      FuzzyXorStrategy highCardinalityStrategy =
          new FuzzyXorStrategy(100, new double[] {0.25}, 42L);
      List<DataPoint> dataset = highCardinalityStrategy.generateDataset(1);

      // Extract first coordinate values (should be around vertex 0)
      List<Double> firstCoordinates =
          dataset.stream()
              .filter(p -> p.label() == 0) // Points around
              // vertex 0
              .map(p -> p.getCoordinate(0))
              .toList();

      // Calculate sample variance
      double mean =
          firstCoordinates.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
      double variance =
          firstCoordinates.stream().mapToDouble(x -> Math.pow(x - mean, 2)).average().orElse(0.0);

      // Variance should be close to 0.25 (within 20% for sample size 100)
      double expectedVariance = 0.25;
      double tolerance = 0.2 * expectedVariance;

      assertTrue(
          Math.abs(variance - expectedVariance) < tolerance,
          String.format("Expected variance %.3f, got %.3f", expectedVariance, variance));
    }

    @Test
    @DisplayName("Noise distribution is approximately normal")
    void testNoiseDistributionShape() {
      // Use high cardinality for better statistical analysis
      FuzzyXorStrategy highCardinalityStrategy =
          new FuzzyXorStrategy(200, new double[] {0.16}, 42L);
      List<DataPoint> dataset = highCardinalityStrategy.generateDataset(1);

      // Extract first coordinate values around vertex 0
      List<Double> coordinates =
          dataset.stream().filter(p -> p.label() == 0).map(p -> p.getCoordinate(0)).toList();

      // Check that approximately 68% of points are within 1 standard deviation
      double mean = coordinates.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
      double std = Math.sqrt(0.16);

      long pointsWithin1Std = coordinates.stream().filter(x -> Math.abs(x - mean) <= std).count();

      double percentageWithin1Std = (double) pointsWithin1Std / coordinates.size();

      // Should be approximately 68% (within 10% tolerance for sample size 200)
      assertTrue(
          Math.abs(percentageWithin1Std - 0.68) < 0.10,
          String.format("Expected ~68%% within 1σ, got %.1f%%", percentageWithin1Std * 100));
    }
  }

  @Nested
  @DisplayName("Edge Cases Tests")
  class EdgeCasesTests {

    @Test
    @DisplayName("Very small variance produces tight clusters")
    void testVerySmallVariance() {
      FuzzyXorStrategy tightStrategy = new FuzzyXorStrategy(50, new double[] {0.001, 0.001}, 42L);
      List<DataPoint> dataset = tightStrategy.generateDataset(2);

      // Points should be very close to vertices
      for (DataPoint point : dataset) {
        double[] coords = point.coordinates();
        for (double coord : coords) {
          // Coordinates should be very close to 0 or 1
          assertTrue(
              Math.abs(coord) < 0.1 || Math.abs(coord - 1.0) < 0.1,
              "Coordinate " + coord + " should be close to 0 or 1 with variance 0.001");
        }
      }
    }

    @Test
    @DisplayName("Very large variance produces wide spread")
    void testVeryLargeVariance() {
      FuzzyXorStrategy wideStrategy = new FuzzyXorStrategy(50, new double[] {10.0, 10.0}, 42L);
      List<DataPoint> dataset = wideStrategy.generateDataset(2);

      // Some points should be far from vertices
      boolean foundWidePoint = false;
      for (DataPoint point : dataset) {
        double[] coords = point.coordinates();
        for (double coord : coords) {
          if (Math.abs(coord) > 5.0 || Math.abs(coord - 1.0) > 5.0) {
            foundWidePoint = true;
            break;
          }
        }
        if (foundWidePoint) break;
      }

      assertTrue(foundWidePoint, "With variance 10.0, should find points far from vertices");
    }

    @Test
    @DisplayName("Single dimension dataset generation")
    void testSingleDimension() {
      FuzzyXorStrategy singleDimStrategy = new FuzzyXorStrategy(20, new double[] {0.1}, 42L);
      List<DataPoint> dataset = singleDimStrategy.generateDataset(1);

      assertEquals(2 * 20, dataset.size()); // 2 vertices × 20 points

      for (DataPoint point : dataset) {
        assertEquals(1, point.getNumDimensions());
      }
    }

    @Test
    @DisplayName("High dimensional dataset generation")
    void testHighDimensions() {
      FuzzyXorStrategy highDimStrategy =
          new FuzzyXorStrategy(5, new double[] {0.1, 0.1, 0.1, 0.1, 0.1}, 42L);
      List<DataPoint> dataset = highDimStrategy.generateDataset(5);

      int expectedSize = (int) Math.pow(2, 5) * 5; // 2⁵ vertices × 5 points
      assertEquals(expectedSize, dataset.size());

      for (DataPoint point : dataset) {
        assertEquals(5, point.getNumDimensions());
      }
    }
  }

  @Nested
  @DisplayName("Mathematical Correctness Tests")
  class MathematicalCorrectnessTests {

    @Test
    @DisplayName("3D dataset generation with matching variance dimensions")
    void test3DDatasetGeneration() {
      double[] variance = {0.1, 0.2, 0.3};
      int cardinality = 15;

      FuzzyXorStrategy testStrategy = new FuzzyXorStrategy(cardinality, variance, 42L);
      List<DataPoint> dataset = testStrategy.generateDataset(3);

      // Should have 2³ × 15 = 120 points
      assertEquals(120, dataset.size());

      // All points should have 3 dimensions
      for (DataPoint point : dataset) {
        assertEquals(3, point.getNumDimensions());
      }

      // Verify that variance array dimension matches dataset dimension
      assertEquals(variance.length, 3, "Variance array should have 3 dimensions for 3D dataset");
    }

    @Test
    @DisplayName("Gaussian noise properties")
    void testGaussianNoiseProperties() {
      // Test that noise is centered around vertices
      FuzzyXorStrategy testStrategy = new FuzzyXorStrategy(100, new double[] {0.25}, 42L);
      List<DataPoint> dataset = testStrategy.generateDataset(1);

      // Separate points by label
      List<Double> label0Coords =
          dataset.stream().filter(p -> p.label() == 0).map(p -> p.getCoordinate(0)).toList();

      List<Double> label1Coords =
          dataset.stream().filter(p -> p.label() == 1).map(p -> p.getCoordinate(0)).toList();

      // Points with label 0 should be around vertex 0
      double meanLabel0 =
          label0Coords.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
      assertTrue(Math.abs(meanLabel0) < 0.5, "Points with label 0 should cluster around 0");

      // Points with label 1 should be around vertex 1
      double meanLabel1 =
          label1Coords.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
      assertTrue(Math.abs(meanLabel1 - 1.0) < 0.5, "Points with label 1 should cluster around 1");
    }
  }
}
