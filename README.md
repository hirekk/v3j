# V3J - Octonion based neural network architecture (actually, only quaternions for now)

A minimal Java project for generating N-dimensional XOR-style datasets using modern Java practices.

## 📁 Structure

```
v3j/
├── build.gradle             # Gradle config with Java 17, JUnit, Picocli, Spotless
├── gradlew                  # Gradle wrapper
├── src/
│   └── data/               # Data generation package
│       ├── Main.java        # CLI entry point with subcommands
│       ├── Dataset.java     # Dataset container with CSV I/O
│       ├── DatasetGenerator.java # Dataset generation orchestration
│       └── strategy/        # Generation strategies
│           ├── ExactXorStrategy.java    # Exact hypercube vertices
│           └── FuzzyXorStrategy.java    # Fuzzy data clouds
└── build/                   # Build output (auto-generated)
```

## 🚀 **Quick Start**

### **Build the Project**
```bash
./gradlew build
```

### **Get Help and Learn Commands**

**General help (shows available subcommands):**
```bash
./gradlew run --args="--help"
```

**Help for exact XOR generation:**
```bash
./gradlew run --args="exact --help"
```

**Help for fuzzy XOR generation:**
```bash
./gradlew run --args="fuzzy --help"
```

**Help for specific command with examples:**
```bash
./gradlew run --args="fuzzy --help"
```

### **Generate Datasets**

**Exact XOR (hypercube vertices):**
```bash
# 2D exact XOR
./gradlew run --args="exact -d 2 -o data/xor/exact2d.csv"

# 3D exact XOR
./gradlew run --args="exact -d 3 -o data/xor/exact3d.csv"

# 4D exact XOR
./gradlew run --args="exact -d 4 -o data/xor/exact4d.csv"
```

**Fuzzy XOR (data clouds):**
```bash
# 2D fuzzy XOR with 10 points per vertex, variance 0.2
./gradlew run --args="fuzzy -d 2 -c 10 -v 0.2 -o data/xor/fuzzy2d.csv"

# 3D fuzzy XOR with 15 points per vertex, variance 0.15
./gradlew run --args="fuzzy -d 3 -c 15 -v 0.15 -o data/xor/fuzzy3d.csv"

# 3D fuzzy XOR with custom seed
./gradlew run --args="fuzzy -d 3 -c 2 -v 0.01 -s 42 -o data/xor/fuzzy3d.csv"
```

### **Get Help**
```bash
# General help
./gradlew run --args="--help"

# Command-specific help
./gradlew run --args="exact --help"
./gradlew run --args="fuzzy --help"
```

## 💡 **What Each Component Does**

- **`Main.java`**: CLI entry point with `exact` and `fuzzy` subcommands
- **`Dataset.java`**: Holds data points, handles CSV import/export, provides utility methods
- **`DatasetGenerator.java`**: Orchestrates dataset generation using strategy pattern
- **`ExactXorStrategy.java`**: Generates exact hypercube vertices (2^N points)
- **`FuzzyXorStrategy.java`**: Generates fuzzy data clouds around vertices with configurable cardinality and variance

## 🛠️ **System Requirements**

- **Java 17+** - JDK required
- **Gradle** - Included via wrapper (`./gradlew`)

## 📥 **Installing Java on macOS:**

```bash
# Homebrew (Recommended)
brew install openjdk@17

# Or download from:
# https://adoptium.net/ (Eclipse Temurin)
# https://www.oracle.com/java/technologies/downloads/ (Oracle)
```

## 🔧 **Build Commands**

```bash
./gradlew build      # Build everything
./gradlew test       # Run tests
./gradlew clean      # Clean build artifacts
./gradlew run --args="exact -d 2"  # Run with arguments
```

## 🎯 **Dataset Types**

### **Exact XOR**
- Generates vertices of an N-dimensional hypercube
- Dataset size: 2^N points
- Each point is exactly at a vertex coordinate
- Labels: 1 for odd number of 1s, 0 for even

### **Fuzzy XOR**
- Generates data clouds around hypercube vertices
- Dataset size: 2^N × cardinality points
- Points are distributed around vertices with Gaussian noise
- Variance controls spread of the clouds

## 📝 **Example Usage**

**Create a 3D exact XOR dataset:**
```bash
./gradlew run --args="exact -d 3 -o data/xor/exact3d.csv"
```

**This generates:**
- 8 data points (2³ vertices)
- 4 columns: 3 input coordinates + 1 label
- CSV file at `data/xor/exact3d.csv`

**Create a 2D fuzzy XOR dataset:**
```bash
./gradlew run --args="fuzzy -d 2 -c 20 -v 0.1 -o data/xor/fuzzy2d.csv"
```

**This generates:**
- 80 data points (2² vertices × 20 points per vertex)
- 3 columns: 2 input coordinates + 1 label
- Points distributed around vertices with 0.1 variance

**Create a 3D fuzzy XOR dataset with custom seed:**
```bash
./gradlew run --args="fuzzy -d 3 -c 2 -v 0.01 -s 42 -o data/xor/fuzzy3d.csv"
```

**This generates:**
- 16 data points (2³ vertices × 2 points per vertex)
- 4 columns: 3 input coordinates + 1 label
- Reproducible results with seed 42

## 🎯 **CLI Options Reference**

### **Exact Command**
- `-d, --num-dimensions` (required): Number of dimensions
- `-o, --output-path`: Output CSV file path (default: `data/xor/exact.csv`)

### **Fuzzy Command**
- `-d, --num-dimensions` (required): Number of dimensions
- `-c, --blob-cardinality` (required): Number of points per vertex cloud
- `-v, --blob-variance` (required): Blob variance
- `-s, --seed`: Random seed (default: 0)
- `-o, --output-path`: Output CSV file path (default: `data/xor/fuzzy.csv`)

## 🏗️ **Project Features**

- **Modern Java 17** with records, pattern matching
- **Strategy Pattern** for different generation approaches
- **Builder Pattern** for fluent API design
- **Robust CSV I/O** with proper error handling
- **CLI interface** using Picocli with short aliases
- **Code formatting** with Spotless
- **JUnit 5** testing framework
