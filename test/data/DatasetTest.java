/*
 * Copyright (c) 2025 Hieronim Kubica. Licensed under the MIT License. See LICENSE file for full
 * terms.
 */

package data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class DatasetTest {

  private Dataset dataset;
  private List<DataPoint> sampleData;

  @BeforeEach
  void setUp() {
    sampleData = new ArrayList<>();
    sampleData.add(new DataPoint(new double[] {1.0, 2.0}, 1));
    sampleData.add(new DataPoint(new double[] {3.0, 4.0}, 0));
    sampleData.add(new DataPoint(new double[] {5.0, 6.0}, 1));

    dataset = new Dataset(sampleData, 2);
  }

  @Test
  void testValidConstruction() {
    assertNotNull(dataset);
    assertEquals(2, dataset.numDimensions);
    assertEquals(3, dataset.getSize());
    assertFalse(dataset.isEmpty());
  }

  @Test
  void testConstructionWithNullData() {
    Dataset emptyDataset = new Dataset(null, 3);
    assertNotNull(emptyDataset.data);
    assertTrue(emptyDataset.data.isEmpty());
    assertEquals(3, emptyDataset.numDimensions);
  }

  @Test
  void testConstructionWithEmptyData() {
    Dataset emptyDataset = new Dataset(new ArrayList<>(), 2);
    assertTrue(emptyDataset.isEmpty());
    assertEquals(0, emptyDataset.getSize());
  }

  @Test
  void testInvalidDimensionsThrowsException() {
    assertThrows(
        IllegalArgumentException.class,
        () -> {
          new Dataset(sampleData, 0);
        });

    assertThrows(
        IllegalArgumentException.class,
        () -> {
          new Dataset(sampleData, -1);
        });
  }

  @Test
  void testDataImmutability() {
    List<DataPoint> originalData = dataset.data;
    originalData.add(new DataPoint(new double[] {7.0, 8.0}, 0));

    assertEquals(4, dataset.getSize());
  }

  @Test
  void testGetSize() {
    assertEquals(3, dataset.getSize());

    Dataset emptyDataset = new Dataset(2);
    assertEquals(0, emptyDataset.getSize());
  }

  @Test
  void testIsEmpty() {
    assertFalse(dataset.isEmpty());

    Dataset emptyDataset = new Dataset(2);
    assertTrue(emptyDataset.isEmpty());
  }

  @Test
  void testShuffle() {
    List<DataPoint> originalOrder = new ArrayList<>(dataset.data);

    // Shuffle multiple times to ensure we get a different order
    // While it's theoretically possible for shuffle to produce the same order multiple times,
    // the probability is astronomically low, making this test reliable
    boolean orderChanged = false;
    for (int i = 0; i < 100; i++) {
      dataset.shuffle();
      if (!dataset.data.equals(originalOrder)) {
        orderChanged = true;
        break;
      }
    }

    assertTrue(orderChanged, "Shuffle should produce different order after multiple attempts");
    assertEquals(originalOrder.size(), dataset.data.size());
    assertTrue(dataset.data.containsAll(originalOrder));
    assertTrue(originalOrder.containsAll(dataset.data));
  }

  @Test
  void testSubsetValidIndices() {
    Dataset subset = dataset.subset(1, 3);

    assertEquals(2, subset.getSize());
    assertEquals(2, subset.numDimensions);
    assertEquals(3.0, subset.data.get(0).getCoordinate(0));
    assertEquals(5.0, subset.data.get(1).getCoordinate(0));
  }

  @Test
  void testSubsetInvalidIndicesThrowsException() {
    assertThrows(
        IllegalArgumentException.class,
        () -> {
          dataset.subset(-1, 2);
        });

    assertThrows(
        IllegalArgumentException.class,
        () -> {
          dataset.subset(1, 5);
        });

    assertThrows(
        IllegalArgumentException.class,
        () -> {
          dataset.subset(2, 2);
        });

    assertThrows(
        IllegalArgumentException.class,
        () -> {
          dataset.subset(3, 1);
        });
  }

  @Test
  void testGetInputs() {
    double[] inputs = dataset.getInputs(0);
    assertArrayEquals(new double[] {1.0, 2.0}, inputs);

    inputs = dataset.getInputs(1);
    assertArrayEquals(new double[] {3.0, 4.0}, inputs);
  }

  @Test
  void testGetLabel() {
    assertEquals(1, dataset.getLabel(0));
    assertEquals(0, dataset.getLabel(1));
    assertEquals(1, dataset.getLabel(2));
  }

  @Test
  void testExportToCSV(@TempDir Path tempDir) throws IOException {
    Path csvFile = tempDir.resolve("test.csv");

    dataset.exportToCSV(csvFile);

    assertTrue(Files.exists(csvFile));
    List<String> lines = Files.readAllLines(csvFile);
    assertEquals(3, lines.size());

    String firstLine = lines.get(0);
    assertTrue(firstLine.contains("1.0"));
    assertTrue(firstLine.contains("2.0"));
    assertTrue(firstLine.contains("1"));
  }

  @Test
  void testExportToCSVWithPrecision(@TempDir Path tempDir) throws IOException {
    Path csvFile = tempDir.resolve("test_precision.csv");

    dataset.exportToCSV(csvFile, 2);

    assertTrue(Files.exists(csvFile));
    List<String> lines = Files.readAllLines(csvFile);

    String firstLine = lines.get(0);
    assertTrue(firstLine.startsWith("1.00,2.00,1"));
  }

  @Test
  void testExportEmptyDatasetThrowsException(@TempDir Path tempDir) {
    Dataset emptyDataset = new Dataset(2);
    Path csvFile = tempDir.resolve("empty.csv");

    assertThrows(
        IllegalStateException.class,
        () -> {
          emptyDataset.exportToCSV(csvFile);
        });
  }

  @Test
  void testFromGenerator() {
    DatasetGenerator mockGenerator = new DatasetGenerator();
    mockGenerator.withNumDimensions(3);
    mockGenerator.withStrategy(new MockGenerationStrategy());

    Dataset generatedDataset = Dataset.fromGenerator(mockGenerator);

    assertNotNull(generatedDataset);
    assertEquals(3, generatedDataset.numDimensions);
    assertEquals(2, generatedDataset.getSize());
  }

  @Test
  void testFromCsv(@TempDir Path tempDir) throws IOException {
    Path csvFile = tempDir.resolve("test_import.csv");
    List<String> csvContent = List.of("1.0,2.0,1", "3.0,4.0,0", "5.0,6.0,1");
    Files.write(csvFile, csvContent);

    Dataset importedDataset = Dataset.fromCsv(csvFile);

    assertNotNull(importedDataset);
    assertEquals(2, importedDataset.numDimensions);
    assertEquals(3, importedDataset.getSize());
    assertEquals(1.0, importedDataset.getInputs(0)[0]);
    assertEquals(0, importedDataset.getLabel(1));
  }

  @Test
  void testFromCsvInvalidFileThrowsException(@TempDir Path tempDir) {
    Path nonExistentFile = tempDir.resolve("nonexistent.csv");

    assertThrows(
        IOException.class,
        () -> {
          Dataset.fromCsv(nonExistentFile);
        });
  }

  @Test
  void testFromCsvInvalidDataThrowsException(@TempDir Path tempDir) throws IOException {
    Path invalidCsvFile = tempDir.resolve("invalid.csv");
    List<String> invalidContent = List.of("1.0,2.0,1", "3.0,invalid,0", "5.0,6.0,1");
    Files.write(invalidCsvFile, invalidContent);

    assertThrows(
        NumberFormatException.class,
        () -> {
          Dataset.fromCsv(invalidCsvFile);
        });
  }

  private static class MockGenerationStrategy implements data.strategy.GenerationStrategy {
    @Override
    public List<DataPoint> generateDataset(int numDimensions) {
      List<DataPoint> points = new ArrayList<>();
      points.add(new DataPoint(new double[] {1.0, 2.0, 3.0}, 1));
      points.add(new DataPoint(new double[] {4.0, 5.0, 6.0}, 0));
      return points;
    }
  }
}
