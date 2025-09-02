/*
 * Copyright (c) 2025 Hieronim Kubica
 * Licensed under the MIT License. See LICENSE file for full terms.
 */
package data.strategy;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import data.DataPoint;

@DisplayName("ExactXorStrategy Tests")
class ExactXorStrategyTest {

  private ExactXorStrategy strategy;

  @BeforeEach
  void setUp() {
    strategy = new ExactXorStrategy();
  }

  @Nested
  @DisplayName("Dataset Generation Tests")
  class DatasetGenerationTests {

    @Test
    @DisplayName("Generate 1D dataset")
    void testGenerate1DDataset() {
      List<DataPoint> dataset = strategy.generateDataset(1);

      assertEquals(2, dataset.size()); // 2¹ = 2 vertices

      // Should have vertices at (0) and (1)
      Set<Double> coordinates =
          dataset.stream().map(p -> p.getCoordinate(0)).collect(Collectors.toSet());

      assertTrue(coordinates.contains(0.0), "Should contain vertex at 0");
      assertTrue(coordinates.contains(1.0), "Should contain vertex at 1");
    }

    @Test
    @DisplayName("Generate 2D dataset")
    void testGenerate2DDataset() {
      List<DataPoint> dataset = strategy.generateDataset(2);

      assertEquals(4, dataset.size()); // 2² = 4 vertices

      // Should have vertices at (0,0), (0,1), (1,0), (1,1)
      // Use a more flexible approach that doesn't depend on order
      boolean found00 = false, found01 = false, found10 = false, found11 = false;

      for (DataPoint point : dataset) {
        double x = point.getCoordinate(0);
        double y = point.getCoordinate(1);

        if (x == 0.0 && y == 0.0) found00 = true;
        else if (x == 0.0 && y == 1.0) found01 = true;
        else if (x == 1.0 && y == 0.0) found10 = true;
        else if (x == 1.0 && y == 1.0) found11 = true;
      }

      assertTrue(found00, "Should contain vertex (0,0)");
      assertTrue(found01, "Should contain vertex (0,1)");
      assertTrue(found10, "Should contain vertex (1,0)");
      assertTrue(found11, "Should contain vertex (1,1)");
    }

    @Test
    @DisplayName("Generate 3D dataset")
    void testGenerate3DDataset() {
      List<DataPoint> dataset = strategy.generateDataset(3);

      assertEquals(8, dataset.size()); // 2³ = 8 vertices

      // Check that all coordinates are 0 or 1
      for (DataPoint point : dataset) {
        assertEquals(3, point.getNumDimensions());
        for (int i = 0; i < 3; i++) {
          double coord = point.getCoordinate(i);
          assertTrue(
              coord == 0.0 || coord == 1.0, "Coordinate " + i + " should be 0 or 1, got: " + coord);
        }
      }
    }

    @Test
    @DisplayName("Generate high dimensional dataset")
    void testGenerateHighDDataset() {
      List<DataPoint> dataset = strategy.generateDataset(5);

      assertEquals(32, dataset.size()); // 2⁵ = 32 vertices

      for (DataPoint point : dataset) {
        assertEquals(5, point.getNumDimensions());
        for (int i = 0; i < 5; i++) {
          double coord = point.getCoordinate(i);
          assertTrue(
              coord == 0.0 || coord == 1.0, "Coordinate " + i + " should be 0 or 1, got: " + coord);
        }
      }
    }
  }

  @Nested
  @DisplayName("Labeling Logic Tests")
  class LabelingLogicTests {

    @Test
    @DisplayName("1D vertex labeling")
    void test1DVertexLabeling() {
      List<DataPoint> dataset = strategy.generateDataset(1);

      // Vertex (0) should have label 0 (even number of 1s)
      // Vertex (1) should have label 1 (odd number of 1s)
      for (DataPoint point : dataset) {
        if (point.getCoordinate(0) == 0.0) {
          assertEquals(0, point.label(), "Vertex (0) should have label 0");
        } else {
          assertEquals(1, point.label(), "Vertex (1) should have label 1");
        }
      }
    }

    @Test
    @DisplayName("2D vertex labeling")
    void test2DVertexLabeling() {
      List<DataPoint> dataset = strategy.generateDataset(2);

      for (DataPoint point : dataset) {
        double x = point.getCoordinate(0);
        double y = point.getCoordinate(1);
        int expectedLabel = calculateExpectedLabel(x, y);

        assertEquals(
            expectedLabel,
            point.label(),
            String.format("Vertex (%.0f,%.0f) should have label %d", x, y, expectedLabel));
      }
    }

    @Test
    @DisplayName("3D vertex labeling")
    void test3DVertexLabeling() {
      List<DataPoint> dataset = strategy.generateDataset(3);

      for (DataPoint point : dataset) {
        double x = point.getCoordinate(0);
        double y = point.getCoordinate(1);
        double z = point.getCoordinate(2);
        int expectedLabel = calculateExpectedLabel(x, y, z);

        assertEquals(
            expectedLabel,
            point.label(),
            String.format("Vertex (%.0f,%.0f,%.0f) should have label %d", x, y, z, expectedLabel));
      }
    }

    @Test
    @DisplayName("Label consistency across dimensions")
    void testLabelConsistency() {
      // Test that the same coordinate pattern gives the same label across dimensions
      List<DataPoint> dataset2D = strategy.generateDataset(2);
      List<DataPoint> dataset3D = strategy.generateDataset(3);

      // Find vertices with pattern (1,1) in 2D and (1,1,0) in 3D
      DataPoint vertex2D =
          dataset2D.stream()
              .filter(p -> p.getCoordinate(0) == 1.0 && p.getCoordinate(1) == 1.0)
              .findFirst()
              .orElse(null);

      DataPoint vertex3D =
          dataset3D.stream()
              .filter(
                  p ->
                      p.getCoordinate(0) == 1.0
                          && p.getCoordinate(1) == 1.0
                          && p.getCoordinate(2) == 0.0)
              .findFirst()
              .orElse(null);

      assertNotNull(vertex2D, "Should find vertex (1,1) in 2D");
      assertNotNull(vertex3D, "Should find vertex (1,1,0) in 3D");

      // Both should have the same label (odd number of 1s = label 1)
      assertEquals(
          vertex2D.label(),
          vertex3D.label(),
          "Vertices with same pattern of 1s should have same label");
    }

    private int calculateExpectedLabel(double... coordinates) {
      int countOnes = 0;
      for (double coord : coordinates) {
        if (coord == 1.0) {
          countOnes++;
        }
      }
      return countOnes % 2; // Even count = 0, odd count = 1
    }
  }

  @Nested
  @DisplayName("Data Integrity Tests")
  class DataIntegrityTests {

    @Test
    @DisplayName("All vertices are unique")
    void testAllVerticesUnique() {
      List<DataPoint> dataset = strategy.generateDataset(4);

      // Convert to string representation for easy comparison
      Set<String> vertexStrings =
          dataset.stream()
              .map(
                  p -> {
                    StringBuilder sb = new StringBuilder();
                    for (int i = 0; i < p.getNumDimensions(); i++) {
                      sb.append(String.format("%.0f", p.getCoordinate(i)));
                    }
                    return sb.toString();
                  })
              .collect(Collectors.toSet());

      assertEquals(dataset.size(), vertexStrings.size(), "All vertices should be unique");
    }

    @Test
    @DisplayName("No duplicate coordinates")
    void testNoDuplicateCoordinates() {
      List<DataPoint> dataset = strategy.generateDataset(3);

      for (int i = 0; i < dataset.size(); i++) {
        for (int j = i + 1; j < dataset.size(); j++) {
          DataPoint p1 = dataset.get(i);
          DataPoint p2 = dataset.get(j);

          boolean coordinatesEqual = true;
          for (int k = 0; k < p1.getNumDimensions(); k++) {
            if (p1.getCoordinate(k) != p2.getCoordinate(k)) {
              coordinatesEqual = false;
              break;
            }
          }

          assertTrue(
              !coordinatesEqual,
              "Vertices " + i + " and " + j + " should have different coordinates");
        }
      }
    }

    @Test
    @DisplayName("Valid label values")
    void testValidLabelValues() {
      List<DataPoint> dataset = strategy.generateDataset(4);

      for (DataPoint point : dataset) {
        int label = point.label();
        assertTrue(label == 0 || label == 1, "Label should be 0 or 1, got: " + label);
      }
    }

    @Test
    @DisplayName("Coordinate bounds")
    void testCoordinateBounds() {
      List<DataPoint> dataset = strategy.generateDataset(5);

      for (DataPoint point : dataset) {
        for (int i = 0; i < point.getNumDimensions(); i++) {
          double coord = point.getCoordinate(i);
          assertTrue(
              coord >= 0.0 && coord <= 1.0,
              "Coordinate " + i + " should be in [0,1], got: " + coord);
        }
      }
    }
  }

  @Nested
  @DisplayName("Edge Cases Tests")
  class EdgeCasesTests {

    @Test
    @DisplayName("Zero dimensions throws exception")
    void testZeroDimensions() {
      assertThrows(IllegalArgumentException.class, () -> strategy.generateDataset(0));
    }

    @Test
    @DisplayName("Negative dimensions throws exception")
    void testNegativeDimensions() {
      assertThrows(IllegalArgumentException.class, () -> strategy.generateDataset(-1));
    }

    @Test
    @DisplayName("Very high dimensions")
    void testVeryHighDimensions() {
      // Test with a reasonable high dimension (10)
      List<DataPoint> dataset = strategy.generateDataset(10);

      assertEquals(1024, dataset.size()); // 2¹⁰ = 1024 vertices

      for (DataPoint point : dataset) {
        assertEquals(10, point.getNumDimensions());

        // Check that all coordinates are binary
        for (int i = 0; i < 10; i++) {
          double coord = point.getCoordinate(i);
          assertTrue(
              coord == 0.0 || coord == 1.0, "Coordinate " + i + " should be 0 or 1, got: " + coord);
        }
      }
    }
  }

  @Nested
  @DisplayName("Mathematical Properties Tests")
  class MathematicalPropertiesTests {

    @Test
    @DisplayName("XOR property verification")
    void testXORProperty() {
      List<DataPoint> dataset = strategy.generateDataset(2);

      // Find vertices (0,0), (0,1), (1,0), (1,1)
      DataPoint v00 = findVertex(dataset, 0.0, 0.0);
      DataPoint v01 = findVertex(dataset, 0.0, 1.0);
      DataPoint v10 = findVertex(dataset, 1.0, 0.0);
      DataPoint v11 = findVertex(dataset, 1.0, 1.0);

      assertNotNull(v00, "Should find vertex (0,0)");
      assertNotNull(v01, "Should find vertex (0,1)");
      assertNotNull(v10, "Should find vertex (1,0)");
      assertNotNull(v11, "Should find vertex (1,1)");

      // Verify XOR property: label should be 1 if exactly one input is 1
      assertEquals(0, v00.label(), "Vertex (0,0) should have label 0");
      assertEquals(1, v01.label(), "Vertex (0,1) should have label 1");
      assertEquals(1, v10.label(), "Vertex (1,0) should have label 1");
      assertEquals(0, v11.label(), "Vertex (1,1) should have label 0");
    }

    @Test
    @DisplayName("Parity property verification")
    void testParityProperty() {
      List<DataPoint> dataset = strategy.generateDataset(4);

      for (DataPoint point : dataset) {
        int actualLabel = point.label();
        int expectedLabel =
            calculateExpectedLabel(
                point.getCoordinate(0),
                point.getCoordinate(1),
                point.getCoordinate(2),
                point.getCoordinate(3));

        assertEquals(expectedLabel, actualLabel, "Label should match parity of 1s in coordinates");
      }
    }

    private DataPoint findVertex(List<DataPoint> dataset, double... coordinates) {
      return dataset.stream()
          .filter(
              p -> {
                if (p.getNumDimensions() != coordinates.length) return false;
                for (int i = 0; i < coordinates.length; i++) {
                  if (p.getCoordinate(i) != coordinates[i]) return false;
                }
                return true;
              })
          .findFirst()
          .orElse(null);
    }

    private int calculateExpectedLabel(double... coordinates) {
      int countOnes = 0;
      for (double coord : coordinates) {
        if (coord == 1.0) {
          countOnes++;
        }
      }
      return countOnes % 2; // Even count = 0, odd count = 1
    }
  }
}
