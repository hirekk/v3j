# V3J - Octonion neural network architecture (actually, only quaternions for now)

An attempt to train octonion-valued neural network without backpropagation...

- ... but it's actually quaternions--for now.
- aaand it doesn't really work... (for now?)

## The idea

Leverage the rich geometry of S^7 (or S^3 for the quaternion PoC) to do away with backpropagation.

Eventually, take advantage of non-associativity of octonion multiplication to get rid of arbitrary activation functions.

## Project Structure

```
v3j/
├── build.gradle             # Gradle config with Java 17, JUnit, Picocli, Spotless
├── gradlew                  # Gradle wrapper
├── src/
│   ├── data/               # Data generation package
│   │   └── README.md       # Detailed data package documentation
│   └── math/               # Mathematical abstractions package
│       └── README.md       # Detailed math package documentation
└── build/                   # Build output (auto-generated)
```

## Quick Start

### Build the Project
```bash
./gradlew build
```

### Get Help and Learn Commands

**General help (shows available subcommands):**
```bash
./gradlew run --args="--help"
```

**Help for specific commands:**
```bash
./gradlew run --args="exact --help"
./gradlew run --args="fuzzy --help"
```

### Basic Usage Examples

**Generate 2D exact XOR dataset:**
```bash
./gradlew run --args="exact --num-dimensions 2 --output-path data/xor/exact2d.csv"
```

**Generate 3D fuzzy XOR dataset:**
```bash
./gradlew run --args="fuzzy --num-dimensions 3 --blob-cardinality 10 --blob-variance 0.15 --output-path data/xor/fuzzy3d.csv"
```

For detailed usage instructions and examples, see the package-specific READMEs:
- [Data Package Documentation](src/data/README.md)
- [Math Package Documentation](src/math/README.md)

## System Requirements

- **Java 24+** - JDK required
- **Gradle** - Included via wrapper (`./gradlew`)

## Installing Java on macOS

```bash
# Homebrew (Recommended)
brew install openjdk@24

# Or download from:
# https://adoptium.net/ (Eclipse Temurin)
# https://www.oracle.com/java/technologies/downloads/ (Oracle)
```

## Build Commands

```bash
./gradlew build      # Build everything
./gradlew test       # Run tests
./gradlew clean      # Clean build artifacts
./gradlew run --args="exact -d 2"  # Run with arguments
```

## Project Features

- **Modern Java 24** with records, pattern matching
- **Strategy Pattern** for different generation approaches
- **Builder Pattern** for fluent API design
- **Custom CSV I/O** without external dependency based on known data format
- **CLI interface** using Picocli with short aliases
- **Code formatting** with Spotless
- **JUnit 5** testing framework
