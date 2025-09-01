/*
 * Copyright (c) 2025 Hieronim Kubica. Licensed under the MIT License. See LICENSE file for full
 * terms.
 */

import data.Dataset;
import data.DatasetGenerator;
import data.strategy.ExactXorStrategy;
import data.strategy.FuzzyXorStrategy;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import math.Quaternion;
import ml.QuaternionPerceptron;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

/**
 * Command-line interface for generating XOR-style datasets.
 *
 * <p>This class provides a CLI with two subcommands for generating different types of XOR datasets:
 * exact hypercube vertices and fuzzy data clouds. The interface uses Picocli for argument parsing
 * and provides help options for each command.
 *
 * <p>The main class acts as a command dispatcher, while the actual dataset generation logic is
 * implemented in the subcommand classes.
 *
 * @see ExactCommand
 * @see FuzzyCommand
 */
@Command(
    name = "v3j",
    mixinStandardHelpOptions = true,
    subcommands = {GenerateCommand.class, TrainCommand.class})
public class Main {

  /**
   * Main entry point for the XOR dataset generator CLI.
   *
   * <p>Parses command-line arguments and executes the appropriate subcommand. Exits with the return
   * code from the executed command.
   *
   * @param args command-line arguments to parse
   */
  public static void main(String[] args) {
    int exitCode = new CommandLine(new Main()).execute(args);
    System.exit(exitCode);
  }
}

/** Command for generating XOR datasets. */
@Command(
    name = "generate",
    mixinStandardHelpOptions = true,
    description = "Generate XOR dataset",
    subcommands = {ExactCommand.class, FuzzyCommand.class})
class GenerateCommand implements Runnable {

  @Override
  public void run() {
    // This command just shows help for its subcommands
    System.out.println("Use 'v3j generate exact' or 'v3j generate fuzzy' to generate datasets.");
  }
}

/**
 * Subcommand for generating exact XOR datasets.
 *
 * <p>Generates datasets containing exactly the vertices of an N-dimensional hypercube. Each vertex
 * is labeled based on the parity of 1s in its coordinates (odd = 1, even = 0). The dataset size is
 * always 2^N where N is the number of dimensions.
 */
@Command(
    name = "exact",
    mixinStandardHelpOptions = true,
    description = "Generate exact XOR dataset")
class ExactCommand implements Runnable {

  /** Number of dimensions for the hypercube */
  @Option(
      names = {"--num-dimensions", "-d"},
      required = true,
      description = "Number of dimensions")
  private int numDimensions;

  /** Output file path for the generated CSV dataset */
  @Option(
      names = {"--output-path", "-o"},
      description = "Output CSV file path")
  private String outputPath = "data/xor/exact.csv";

  /**
   * Executes the exact XOR dataset generation.
   *
   * <p>Creates a DatasetGenerator with ExactXorStrategy, generates the dataset, and exports it to
   * the specified CSV file.
   *
   * @throws RuntimeException if dataset generation or export fails
   */
  @Override
  public void run() {
    DatasetGenerator generator =
        new DatasetGenerator()
            .withNumDimensions(numDimensions)
            .withStrategy(new ExactXorStrategy());

    Dataset dataset = Dataset.fromGenerator(generator);
    try {
      dataset.exportToCSV(Path.of(outputPath));
    } catch (IOException e) {
      System.err.printf("Error saving dataset: %s%n", e.getMessage());
      System.exit(1);
    }

    System.out.printf(
        "Generated exact XOR dataset with %d dimensions, saved to %s%n", numDimensions, outputPath);
  }
}

/**
 * Subcommand for generating fuzzy XOR datasets.
 *
 * <p>Generates datasets with data clouds around each hypercube vertex. Each vertex has a
 * configurable number of points distributed with Gaussian noise around it. The dataset size is 2^N
 * × cardinality where N is the number of dimensions.
 */
@Command(
    name = "fuzzy",
    mixinStandardHelpOptions = true,
    description = "Generate fuzzy XOR data clouds")
class FuzzyCommand implements Runnable {

  /** Number of dimensions for the hypercube */
  @Option(
      names = {"--num-dimensions", "-d"},
      required = true,
      description = "Number of dimensions")
  private int numDimensions;

  /** Number of points to generate around each vertex */
  @Option(
      names = {"--blob-cardinality", "-c"},
      required = true,
      description = "Number of points per vertex cloud")
  private int blobCardinality;

  /** Standard deviation for the Gaussian distribution around vertices */
  @Option(
      names = {"--blob-variance", "-v"},
      required = true,
      description = "Blob variance")
  private double blobVariance;

  /** Random seed for reproducible generation (optional) */
  @Option(
      names = {"--seed", "-s"},
      required = false,
      description = "Random seed")
  private Long seed = 0L;

  /** Output file path for the generated CSV dataset */
  @Option(
      names = {"--output-path", "-o"},
      description = "Output CSV file path")
  private String outputPath = "data/xor/fuzzy.csv";

  /**
   * Executes the fuzzy XOR dataset generation.
   *
   * <p>Creates a DatasetGenerator with FuzzyXorStrategy, generates the dataset with configurable
   * cardinality and variance, and exports it to the specified CSV file.
   *
   * @throws RuntimeException if dataset generation or export fails
   */
  @Override
  public void run() {
    double[] blobVarianceArray = new double[numDimensions];
    Arrays.fill(blobVarianceArray, blobVariance);
    DatasetGenerator generator =
        new DatasetGenerator()
            .withNumDimensions(numDimensions)
            .withStrategy(new FuzzyXorStrategy(blobCardinality, blobVarianceArray, seed));

    Dataset dataset = Dataset.fromGenerator(generator);
    try {
      dataset.exportToCSV(Path.of(outputPath));
    } catch (IOException e) {
      System.err.printf("Error saving dataset: %s%n", e.getMessage());
      System.exit(1);
    }

    System.out.printf(
        "Generated fuzzy XOR dataset with %d dimensions, saved to %s%n", numDimensions, outputPath);
  }
}

/** Subcommand for training the QuaternionPerceptron on a dataset. */
@Command(
    name = "train",
    mixinStandardHelpOptions = true,
    description = "Train quaternion perceptron on dataset")
class TrainCommand implements Runnable {

  @Option(
      names = {"--data", "-d"},
      description = "Input dataset file path")
  private String dataPath = "data/xor/exact2d.csv";

  @Override
  public void run() {
    try {
      trainModel(dataPath);
    } catch (Exception e) {
      System.err.printf("Error during training: %s%n", e.getMessage());
      e.printStackTrace();
      System.exit(1);
    }
  }

  private void trainModel(String dataPath) throws Exception {
    System.out.println("Loading dataset from CSV: " + dataPath);
    Dataset dataset = Dataset.fromCsv(Path.of(dataPath));
    System.out.println(
        "Dataset loaded: "
            + dataset.data.size()
            + " samples, "
            + dataset.numDimensions
            + " dimensions");

    List<Quaternion> inputOrientations =
        dataset.data.stream()
            .map(
                dp -> {
                  double x = dp.getCoordinate(0);
                  double y = dp.getCoordinate(1);
                  // Replace 0s with -1s and add z-offset to ensure distinct quaternion
                  // representations
                  double quatX = (Math.abs(x) < 1e-10) ? -1.0 : x;
                  double quatY = (Math.abs(y) < 1e-10) ? -1.0 : y;

                  // Create quaternion with w=0 to represent pure quaternions, then normalize
                  return new Quaternion(0.0, quatX, quatY, 0.5).normalize();
                })
            .collect(Collectors.toList());

    List<Integer> binaryLabels =
        dataset.data.stream().map(dp -> dp.label()).collect(Collectors.toList());

    System.out.println("Initializing QuaternionPerceptron...");
    QuaternionPerceptron perceptron = new QuaternionPerceptron(42L);

    System.out.println("Initial bias rotation: " + perceptron.getBiasRotation());
    System.out.println("Initial action rotation: " + perceptron.getActionRotation());

    System.out.println("\nStarting training loop...");
    int epochs = 100;

    for (int epoch = 0; epoch < epochs; epoch++) {
      dataset.shuffle();
      perceptron.step(inputOrientations, binaryLabels);

      if (epoch % 10 == 0) {
        // Compute training accuracy
        int correctPredictions = 0;
        double totalErrorMagnitude = 0.0;

        for (int i = 0; i < inputOrientations.size(); i++) {
          Quaternion input = inputOrientations.get(i);
          int targetLabel = binaryLabels.get(i);
          int predicted = perceptron.classify(input);

          if (predicted == targetLabel) {
            correctPredictions++;
          }

          // Compute error magnitude using geodesic distance
          Quaternion predictedQuat = perceptron.forward(input);
          // Use the same target quaternions as the perceptron for consistency
          Quaternion targetQuat =
              targetLabel == 1
                  ? Quaternion.fromAxisAngle(Math.PI, new double[] {0.0, 0.0, 1.0})
                  : // 180°
                  // rotation
                  // around
                  // Z
                  // for
                  // label
                  // 1
                  Quaternion.ONE; // Identity for label 0

          double errorMagnitude = predictedQuat.geodesicDistance(targetQuat);
          totalErrorMagnitude += errorMagnitude;
        }

        double accuracy = (double) correctPredictions / inputOrientations.size();
        double avgErrorMagnitude = totalErrorMagnitude / inputOrientations.size();

        System.out.printf(
            "Epoch %3d: Training Accuracy: %.2f%%, Average Error Magnitude: %.6f%n",
            epoch, accuracy * 100.0, avgErrorMagnitude);
      }
    }

    System.out.println("\nTraining completed!");
    System.out.println("Final bias rotation: " + perceptron.getBiasRotation());
    System.out.println("Final action rotation: " + perceptron.getActionRotation());

    System.out.println("\nTesting on first few samples:");
    for (int i = 0; i < Math.min(5, inputOrientations.size()); i++) {
      Quaternion input = inputOrientations.get(i);
      int targetLabel = binaryLabels.get(i);
      int predicted = perceptron.classify(input);

      System.out.printf(
          "Sample %d: Input=(%.2f, %.2f), Target=%d, Predicted=%d%n",
          i, input.getX(), input.getY(), targetLabel, predicted);
    }
  }
}
