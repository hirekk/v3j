/*
 * Copyright (c) 2025 Hieronim Kubica. Licensed under the MIT License. See LICENSE file for full
 * terms.
 */

package data;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import data.strategy.GenerationStrategy;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class DatasetGeneratorTest {

  private DatasetGenerator generator;
  private MockGenerationStrategy mockStrategy;

  @BeforeEach
  void setUp() {
    generator = new DatasetGenerator();
    mockStrategy = new MockGenerationStrategy();
  }

  @Test
  void testWithNumDimensions() {
    DatasetGenerator result = generator.withNumDimensions(5);

    assertSame(generator, result);
    assertEquals(5, generator.getNumDimensions());
  }

  @Test
  void testWithStrategy() {
    DatasetGenerator result = generator.withStrategy(mockStrategy);

    assertSame(generator, result);
  }

  @Test
  void testGenerateDatasetWithoutStrategyThrowsException() {
    generator.withNumDimensions(3);

    assertThrows(
        IllegalStateException.class,
        () -> {
          generator.generateDataset();
        });
  }

  @Test
  void testGenerateDatasetWithStrategy() {
    generator.withNumDimensions(3).withStrategy(mockStrategy);

    List<DataPoint> dataset = generator.generateDataset();

    assertNotNull(dataset);
    assertEquals(3, dataset.size());
    assertEquals(3, generator.getNumDimensions());
  }

  @Test
  void testFluentInterfaceChaining() {
    DatasetGenerator result = generator.withNumDimensions(4).withStrategy(mockStrategy);

    assertSame(generator, result);
    assertEquals(4, generator.getNumDimensions());

    List<DataPoint> dataset = generator.generateDataset();
    assertNotNull(dataset);
  }

  @Test
  void testGetNumDimensions() {
    assertEquals(0, generator.getNumDimensions());

    generator.withNumDimensions(7);
    assertEquals(7, generator.getNumDimensions());
  }

  @Test
  void testMultipleStrategyChanges() {
    MockGenerationStrategy strategy1 = new MockGenerationStrategy();
    MockGenerationStrategy strategy2 = new MockGenerationStrategy();

    generator.withNumDimensions(2).withStrategy(strategy1);
    List<DataPoint> dataset1 = generator.generateDataset();

    generator.withStrategy(strategy2);
    List<DataPoint> dataset2 = generator.generateDataset();

    assertNotNull(dataset1);
    assertNotNull(dataset2);
  }

  private static class MockGenerationStrategy implements GenerationStrategy {
    @Override
    public List<DataPoint> generateDataset(int numDimensions) {
      List<DataPoint> points = new java.util.ArrayList<>();

      for (int i = 0; i < 3; i++) {
        double[] coords = new double[numDimensions];
        for (int j = 0; j < numDimensions; j++) {
          coords[j] = i + j * 0.1;
        }
        points.add(new DataPoint(coords, i % 2));
      }

      return points;
    }
  }
}
