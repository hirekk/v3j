package data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class Dataset {
  public final List<DataPoint> data;
  public final int numDimensions;

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

  public Dataset(int numDimensions) {
    this.numDimensions = numDimensions;
    this.data = new ArrayList<>();
  }

  public int getSize() {
    return data.size();
  }

  public boolean isEmpty() {
    return data.isEmpty();
  }

  // Data initialization methods

  /**
   * Validates file exists and is readable and nonempty.
   *
   * @param filePath Path to CSV file
   * @throws IOException if file does not exist, is not readable, or is empty.
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
   * Reads CSV file and returns raw string arrays.
   *
   * @param filePath Path to CSV file
   * @return list of string arrays, each representing a CSV line
   * @throws IOException if an I/O error occurs opening the file
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
   * Parses raw CSV lines into DataPoint objects.
   *
   * @param csvContents contents of CSV file - lines split by comma.
   * @return list of DataPoints, each representing a CSV line
   * @throws NumberFormatException if CSV contains invalid numeric data
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
   * Loads data from CSV file
   *
   * @param filePath Path to CSV file
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
   * Creates dataset from generator
   *
   * @param generator DatasetGenerator to use
   * @return New Dataset instance
   */
  public static Dataset fromGenerator(DatasetGenerator generator) {
    int numDimensions = generator.getNumDimensions();
    List<DataPoint> data = generator.generateDataset();

    return new Dataset(data, numDimensions);
  }

  // Shuffling - modifies the dataset in-place
  public void shuffle() {
    Collections.shuffle(data);
  }

  // Serialization
  public void exportToCSV(Path filePath) throws IOException {
    if (data.isEmpty()) {
      throw new IllegalStateException("Cannot export empty dataset");
    }
    writeDatasetToCSV(data, filePath, -1);
  }

  public void exportToCSV(Path filePath, int precision) throws IOException {
    if (data.isEmpty()) {
      throw new IllegalStateException("Cannot export empty dataset");
    }
    writeDatasetToCSV(data, filePath, precision);
  }

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

  private String precisionToFormatString(int precision) {
    String formatString = "%";
    if (precision > 0) {
      formatString += "." + precision;
    }
    return formatString + "f";
  }

  // Utility methods
  public Dataset subset(int startIndex, int endIndex) {
    if (startIndex < 0 || endIndex > data.size() || startIndex >= endIndex) {
      throw new IllegalArgumentException("Invalid subset indices");
    }
    List<DataPoint> subsetData = data.subList(startIndex, endIndex);
    return new Dataset(subsetData, numDimensions);
  }

  // Training loop convenience methods
  public double[] getInputs(int index) {
    return data.get(index).coordinates();
  }

  public double getLabel(int index) {
    return data.get(index).label();
  }
}
