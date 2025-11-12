#include "cli_parser.h"
#include "config.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>

CLIParser::CLIParser(int argc, char** argv) {
    for (int i = 0; i < argc; i++) {
        args_.push_back(std::string(argv[i]));
    }
}

CLIOptions CLIParser::parse() {
    parseArguments();
    return options_;
}

void CLIParser::parseArguments() {
    for (size_t i = 1; i < args_.size(); i++) {
        std::string arg = args_[i];
        
        if (arg == "--help" || arg == "-h") {
            options_.show_help = true;
        } else if (arg == "--version" || arg == "-v") {
            options_.show_version = true;
        } else if (arg == "--mode" || arg == "-m") {
            if (i + 1 >= args_.size()) {
                throw std::runtime_error("--mode requires an argument");
            }
            options_.mode = parseMode(args_[++i]);
        } else if (arg == "--config" || arg == "-c") {
            if (i + 1 >= args_.size()) {
                throw std::runtime_error("--config requires an argument");
            }
            options_.config_file = args_[++i];
        } else if (arg == "--data" || arg == "-d") {
            if (i + 1 >= args_.size()) {
                throw std::runtime_error("--data requires an argument");
            }
            options_.data_file = args_[++i];
        } else if (arg == "--threads" || arg == "-t") {
            if (i + 1 >= args_.size()) {
                throw std::runtime_error("--threads requires an argument");
            }
            options_.num_threads = std::stoi(args_[++i]);
            if (options_.num_threads <= 0) {
                throw std::runtime_error("Number of threads must be positive");
            }
        } else if (arg == "--factors" || arg == "-f") {
            if (i + 1 >= args_.size()) {
                throw std::runtime_error("--factors requires an argument");
            }
            options_.num_factors = std::stoi(args_[++i]);
            if (options_.num_factors <= 0) {
                throw std::runtime_error("Number of factors must be positive");
            }
        } else if (arg == "--epochs" || arg == "-e") {
            if (i + 1 >= args_.size()) {
                throw std::runtime_error("--epochs requires an argument");
            }
            options_.max_epochs = std::stoi(args_[++i]);
            if (options_.max_epochs <= 0) {
                throw std::runtime_error("Number of epochs must be positive");
            }
        } else if (arg == "--learning-rate" || arg == "-lr") {
            if (i + 1 >= args_.size()) {
                throw std::runtime_error("--learning-rate requires an argument");
            }
            options_.learning_rate = std::stod(args_[++i]);
            if (options_.learning_rate <= 0.0 || options_.learning_rate > 1.0) {
                throw std::runtime_error("Learning rate must be between 0 and 1");
            }
        } else if (arg == "--regularization" || arg == "-reg") {
            if (i + 1 >= args_.size()) {
                throw std::runtime_error("--regularization requires an argument");
            }
            options_.regularization = std::stod(args_[++i]);
            if (options_.regularization < 0.0) {
                throw std::runtime_error("Regularization must be non-negative");
            }
        } else if (arg == "--top-n" || arg == "-n") {
            if (i + 1 >= args_.size()) {
                throw std::runtime_error("--top-n requires an argument");
            }
            options_.top_n = std::stoi(args_[++i]);
            if (options_.top_n <= 0) {
                throw std::runtime_error("Top N must be positive");
            }
        } else if (arg == "--benchmark" || arg == "-b") {
            options_.benchmark_mode = true;
        } else if (arg == "--verbose") {
            options_.verbose = true;
        } else if (arg == "--quiet" || arg == "-q") {
            options_.quiet = true;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    
    // Validate conflicting options
    if (options_.verbose && options_.quiet) {
        throw std::runtime_error("Cannot specify both --verbose and --quiet");
    }
}

ExecutionMode CLIParser::parseMode(const std::string& mode_str) {
    std::string mode_lower = mode_str;
    std::transform(mode_lower.begin(), mode_lower.end(), mode_lower.begin(), ::tolower);
    
    if (mode_lower == "serial" || mode_lower == "s") {
        return ExecutionMode::SERIAL;
    } else if (mode_lower == "openmp" || mode_lower == "omp") {
        return ExecutionMode::OPENMP;
    } else if (mode_lower == "mpi" || mode_lower == "m") {
        return ExecutionMode::MPI;
    } else if (mode_lower == "hybrid" || mode_lower == "h") {
        return ExecutionMode::HYBRID;
    } else {
        throw std::runtime_error("Invalid execution mode: " + mode_str + 
                                ". Valid modes: serial, openmp, mpi, hybrid");
    }
}

void CLIParser::applyToConfiguration(Configuration& config) const {
    // Apply command-line overrides to configuration
    if (options_.num_threads > 0) {
        config.setNumThreads(options_.num_threads);
    }
    if (options_.num_factors > 0) {
        config.setNumFactors(options_.num_factors);
    }
    if (options_.max_epochs > 0) {
        config.setMaxEpochs(options_.max_epochs);
    }
    if (options_.learning_rate > 0.0) {
        config.setLearningRate(options_.learning_rate);
    }
    if (options_.regularization >= 0.0) {
        config.setRegularization(options_.regularization);
    }
    if (options_.top_n > 0) {
        config.setTopNRecommendations(options_.top_n);
    }
    if (!options_.data_file.empty()) {
        config.setInputFile(options_.data_file);
    }
    
    // Set MPI usage based on execution mode
    config.setUseMPI(options_.mode == ExecutionMode::MPI || options_.mode == ExecutionMode::HYBRID);
}

bool CLIParser::validateMode() const {
    return isModeSupported(options_.mode);
}

void CLIParser::printHelp(const std::string& program_name) {
    printUsage(program_name);
    printModeHelp();
}

void CLIParser::printUsage(const std::string& program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]" << std::endl;
    std::cout << std::endl;
    std::cout << "Parallel Matrix Factorization Recommendation System" << std::endl;
    std::cout << "Version: " << PROJECT_VERSION << std::endl;
    std::cout << std::endl;
    std::cout << "OPTIONS:" << std::endl;
    std::cout << "  -h, --help                    Show this help message" << std::endl;
    std::cout << "  -v, --version                 Show version information" << std::endl;
    std::cout << "  -m, --mode MODE               Execution mode (serial, openmp, mpi, hybrid)" << std::endl;
    std::cout << "  -c, --config FILE             Configuration file (JSON format)" << std::endl;
    std::cout << "  -d, --data FILE               Input data file (.dat format)" << std::endl;
    std::cout << "  -t, --threads N               Number of OpenMP threads" << std::endl;
    std::cout << "  -f, --factors N               Number of latent factors" << std::endl;
    std::cout << "  -e, --epochs N                Maximum number of epochs" << std::endl;
    std::cout << "  -lr, --learning-rate RATE     Learning rate (0.0 to 1.0)" << std::endl;
    std::cout << "  -reg, --regularization REG    Regularization parameter" << std::endl;
    std::cout << "  -n, --top-n N                 Number of top recommendations" << std::endl;
    std::cout << "  -b, --benchmark               Run performance benchmark" << std::endl;
    std::cout << "  --verbose                     Enable verbose output" << std::endl;
    std::cout << "  -q, --quiet                   Suppress non-essential output" << std::endl;
    std::cout << std::endl;
}

void CLIParser::printVersion() {
    std::cout << PROJECT_NAME << " version " << PROJECT_VERSION << std::endl;
    std::cout << "Build configuration:" << std::endl;
    std::cout << "  OpenMP support: " << (OPENMP_ENABLED ? "enabled" : "disabled") << std::endl;
    std::cout << "  MPI support: " << (MPI_ENABLED ? "enabled" : "disabled") << std::endl;
}

void CLIParser::printModeHelp() {
    std::cout << "EXECUTION MODES:" << std::endl;
    std::cout << "  serial                        Single-threaded execution" << std::endl;
    std::cout << "  openmp                        Multi-threaded with OpenMP" << std::endl;
    std::cout << "  mpi                           Distributed with MPI" << std::endl;
    std::cout << "  hybrid                        Combined OpenMP + MPI" << std::endl;
    std::cout << std::endl;
    std::cout << "EXAMPLES:" << std::endl;
    std::cout << "  # Run with default configuration" << std::endl;
    std::cout << "  " << args_[0] << std::endl;
    std::cout << std::endl;
    std::cout << "  # Run OpenMP with 8 threads" << std::endl;
    std::cout << "  " << args_[0] << " --mode openmp --threads 8" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Run with custom configuration file" << std::endl;
    std::cout << "  " << args_[0] << " --config my_config.json --mode openmp" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Run MPI with 4 processes" << std::endl;
    std::cout << "  mpirun -np 4 " << args_[0] << " --mode mpi" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Run benchmark with custom parameters" << std::endl;
    std::cout << "  " << args_[0] << " --benchmark --factors 100 --epochs 50" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Run with custom data file" << std::endl;
    std::cout << "  " << args_[0] << " --data my_ratings.dat --mode openmp --threads 4" << std::endl;
    std::cout << std::endl;
}

// Utility functions
std::string executionModeToString(ExecutionMode mode) {
    switch (mode) {
        case ExecutionMode::SERIAL: return "serial";
        case ExecutionMode::OPENMP: return "openmp";
        case ExecutionMode::MPI: return "mpi";
        case ExecutionMode::HYBRID: return "hybrid";
        default: return "unknown";
    }
}

bool isModeSupported(ExecutionMode mode) {
    switch (mode) {
        case ExecutionMode::SERIAL:
            return true;
        case ExecutionMode::OPENMP:
            return OPENMP_ENABLED;
        case ExecutionMode::MPI:
            return MPI_ENABLED;
        case ExecutionMode::HYBRID:
            return OPENMP_ENABLED && MPI_ENABLED;
        default:
            return false;
    }
}