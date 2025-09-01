/*
 * Copyright (c) 2024 Hieronim Kubica. Licensed under the MIT License. See LICENSE file for full
 * terms.
 */

package data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Container for N-dimensional data points with CSV import/export capabilities.
 *
 * <p>This class provides a mutable container for DataPoint objects, supporting operations like
 * shuffling, subsetting, and CSV serialization. The dataset maintains a fixed dimensionality that
 * applies to all contained data points.
 *
 * <p>The class supports both programmatic construction and loading from CSV files. CSV files are
 * expected to have N coordinate columns followed by a binary label column.
 */
public final class Dataset {
  /** The list of data points in this dataset */
  public final List<DataPoint> data;

  /** The dimensionality of all data points in this dataset */
  public final int numDimensions;

  /**
   * Constructs a dataset with the specified data points and dimensionality.
   *
   * @param data the list of data points (can be null for empty dataset)
   * @param numDimensions the dimensionality of the dataset (must be positive)
   * @throws IllegalArgumentException if numDimensions is not positive
   */
  public Dataset(List<DataPoint> data, int numDimensions) {
    if (numDimensions <= 0) {
      throw new IllegalArgumentException("Number of dimensions must be a positive integer");
    }
    this.numDimensions = numDimensions;

    if (data != null) {
      this.data = new ArrayList<>(data);
    } else {
      this.data = new ArrayList<>();
    }
  }

  /**
   * Constructs an empty dataset with the specified dimensionality.
   *
   * @param numDimensions the dimensionality of the dataset (must be positive)
   * @throws IllegalArgumentException if numDimensions is not positive
   */
  public Dataset(int numDimensions) {
    this.numDimensions = numDimensions;
    this.data = new ArrayList<>();
  }

  /**
   * Returns the number of data points in this dataset.
   *
   * @return the size of the dataset
   */
  public int getSize() {
    return data.size();
  }

  /**
   * Checks if this dataset contains no data points.
   *
   * @return true if the dataset is empty, false otherwise
   */
  public boolean isEmpty() {
    return data.isEmpty();
  }

  // Data initialization methods

  /**
   * Validates that a CSV file exists, is readable, and is not empty.
   *
   * @param filePath path to the CSV file to validate
   * @throws IOException if the file does not exist, is not readable, or is empty
   */
  private static void validateCsvFile(Path filePath) throws IOException {
    if (!Files.exists(filePath)) {
      String errorMsg = String.format("File does not exist %s", filePath);
      throw new IOException(errorMsg);
    }
    if (!Files.isReadable(filePath)) {
      throw new IOException("File is not readable: " + filePath);
    }
    if (Files.size(filePath) == 0) {
      throw new IOException("CSV file is empty");
    }
  }

  /**
   * Reads a CSV file and returns the raw string arrays for each line.
   *
   * @param filePath path to the CSV file to read
   * @return list of string arrays, each representing a CSV line split by commas
   * @throws IOException if an I/O error occurs while reading the file
   */
  private static List<String[]> readCsv(Path filePath) throws IOException {
    List<String[]> contents = new ArrayList<>();

    try (BufferedReader reader = Files.newBufferedReader(filePath)) {
      String line;

      while ((line = reader.readLine()) != null) {
        if (!line.trim().isEmpty()) {
          contents.add(line.split(","));
        }
      }
    }

    return contents;
  }

  /**
   * Parses raw CSV string arrays into DataPoint objects.
   *
   * @param csvContents the raw CSV content as a list of string arrays
   * @return list of DataPoint objects parsed from the CSV data
   * @throws NumberFormatException if the CSV contains invalid numeric data
   */
  private static List<DataPoint> parseCsvData(List<String[]> csvContents) {
    List<DataPoint> csvData = new ArrayList<>(csvContents.size());

    for (int rowIndex = 0; rowIndex < csvContents.size(); rowIndex++) {
      String[] values = csvContents.get(rowIndex);
      int numRowCoordinateDimensions = values.length - 1;

      double[] coordinates = new double[numRowCoordinateDimensions];
      for (int i = 0; i < numRowCoordinateDimensions; i++) {
        coordinates[i] = Double.parseDouble(values[i].trim());
      }

      int label = Integer.parseInt(values[numRowCoordinateDimensions]);

      csvData.add(new DataPoint(coordinates, label));
    }

    return csvData;
  }

  // Static factory methods

  /**
   * Creates a dataset by loading data from a CSV file.
   *
   * @param filePath path to the CSV file to load
   * @return a new Dataset instance containing the loaded data
   * @throws IOException if file operations fail
   * @throws NumberFormatException if CSV contains invalid numeric data
   */
  public static Dataset fromCsv(Path filePath) throws IOException {
    validateCsvFile(filePath);

    List<String[]> csvContents = readCsv(filePath);

    List<DataPoint> csvData = parseCsvData(csvContents);
    int numDimensions = csvData.get(0).getNumDimensions();

    return new Dataset(csvData, numDimensions);
  }

  /**
   * Creates a dataset using a DatasetGenerator.
   *
   * @param generator the DatasetGenerator to use for creating the dataset
   * @return a new Dataset instance containing the generated data
   */
  public static Dataset fromGenerator(DatasetGenerator generator) {
    int numDimensions = generator.getNumDimensions();
    List<DataPoint> data = generator.generateDataset();

    return new Dataset(data, numDimensions);
  }

  /**
   * Shuffles the data points in this dataset in-place.
   *
   * <p>This method modifies the internal data list by randomizing the order of data points using
   * Collections.shuffle().
   */
  public void shuffle() {
    Collections.shuffle(data);
  }

  /**
   * Exports this dataset to a CSV file with default precision.
   *
   * @param filePath path to the output CSV file
   * @throws IOException if an I/O error occurs during writing
   * @throws IllegalStateException if the dataset is empty
   */
  public void exportToCSV(Path filePath) throws IOException {
    if (data.isEmpty()) {
      throw new IllegalStateException("Cannot export empty dataset");
    }
    writeDatasetToCSV(data, filePath, -1);
  }

  /**
   * Exports this dataset to a CSV file with specified precision.
   *
   * @param filePath path to the output CSV file
   * @param precision the number of decimal places for coordinate values (ignored if <= 0)
   * @throws IOException if an I/O error occurs during writing
   * @throws IllegalStateException if the dataset is empty
   */
  public void exportToCSV(Path filePath, int precision) throws IOException {
    if (data.isEmpty()) {
      throw new IllegalStateException("Cannot export empty dataset");
    }
    writeDatasetToCSV(data, filePath, precision);
  }

  /**
   * Writes dataset data to a CSV file with the specified precision.
   *
   * @param data the list of data points to write
   * @param filePath path to the output CSV file
   * @param precision the number of decimal places for coordinate values
   * @throws IOException if an I/O error occurs during writing
   */
  private void writeDatasetToCSV(List<DataPoint> data, Path filePath, int precision)
      throws IOException {

    String formatString = precisionToFormatString(precision);

    try (BufferedWriter writer = Files.newBufferedWriter(filePath)) {
      for (DataPoint point : data) {
        for (int i = 0; i < numDimensions; i++) {
          if (i > 0) writer.write(",");
          writer.write(String.format(formatString, point.getCoordinate(i)));
        }
        writer.write("," + point.label() + "\n");
      }
    }
  }

  /**
   * Converts precision value to a format string for coordinate formatting.
   *
   * @param precision the number of decimal places (ignored if <= 0)
   * @return a format string suitable for String.format()
   */
  private String precisionToFormatString(int precision) {
    String formatString = "%";
    if (precision > 0) {
      formatString += "." + precision;
    }
    return formatString + "f";
  }

  /**
   * Creates a subset of this dataset containing data points from startIndex to endIndex.
   *
   * @param startIndex the starting index (inclusive)
   * @param endIndex the ending index (exclusive)
   * @return a new Dataset instance containing the subset of data points
   * @throws IllegalArgumentException if the indices are invalid
   */
  public Dataset subset(int startIndex, int endIndex) {
    if (startIndex < 0 || endIndex > data.size() || startIndex >= endIndex) {
      throw new IllegalArgumentException("Invalid subset indices");
    }
    List<DataPoint> subsetData = data.subList(startIndex, endIndex);
    return new Dataset(subsetData, numDimensions);
  }

  /**
   * Gets the input coordinates for the data point at the specified index.
   *
   * @param index the index of the data point
   * @return array of coordinate values for the specified data point
   * @throws IndexOutOfBoundsException if the index is out of bounds
   */
  public double[] getInputs(int index) {
    return data.get(index).coordinates();
  }

  /**
   * Gets the binary label for the data point at the specified index.
   *
   * @param index the index of the data point
   * @return the binary label (0 or 1) for the specified data point
   * @throws IndexOutOfBoundsException if the index is out of bounds
   */
  public double getLabel(int index) {
    return data.get(index).label();
  }
}
