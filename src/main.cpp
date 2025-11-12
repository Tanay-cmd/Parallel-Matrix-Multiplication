// Main entry point for parallel matrix factorization recommendation system
#include "config.h"
#include "rating_matrix.h"
#include "sgd_factorizer.h"
#include "json_config.h"
#include "cli_parser.h"
#ifdef USE_MPI
#include "mpi_coordinator.h"
#endif
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>

int main(int argc, char** argv) {
#ifdef USE_MPI
    // Initialize MPI coordinator for MPI and hybrid modes
    MPICoordinator* mpi_coordinator = nullptr;
#endif
    
    try {
        // Parse command line arguments
        CLIParser cli_parser(argc, argv);
        CLIOptions options = cli_parser.parse();
        
#ifdef USE_MPI
        // Initialize MPI for MPI and hybrid modes
        if (options.mode == ExecutionMode::MPI || options.mode == ExecutionMode::HYBRID) {
            mpi_coordinator = new MPICoordinator();
            mpi_coordinator->initialize(argc, argv);
        }
#endif
        
        // Handle help and version requests
        if (options.show_help) {
            cli_parser.printHelp(argv[0]);
            return 0;
        }
        
        if (options.show_version) {
            std::cout << PROJECT_NAME << " version " << PROJECT_VERSION << std::endl;
            std::cout << "Build configuration:" << std::endl;
            std::cout << "  OpenMP support: " << (OPENMP_ENABLED ? "enabled" : "disabled") << std::endl;
            std::cout << "  MPI support: " << (MPI_ENABLED ? "enabled" : "disabled") << std::endl;
            return 0;
        }
        
        // Validate execution mode is supported
        if (!cli_parser.validateMode()) {
            std::cerr << "Error: Execution mode '" << executionModeToString(options.mode) 
                      << "' is not supported in this build." << std::endl;
            std::cerr << "Available modes depend on compile-time configuration (OpenMP/MPI)." << std::endl;
            return 1;
        }
        
        // Load configuration
        Configuration config;
        
        // Load from config file if specified
        if (!options.config_file.empty()) {
            if (!options.quiet) {
                std::cout << "Loading configuration from: " << options.config_file << std::endl;
            }
            config.loadFromFile(options.config_file);
        } else {
            // Try to load default config file if it exists
            try {
                config.loadFromFile("config/default_config.json");
                if (!options.quiet) {
                    std::cout << "Loaded default configuration file." << std::endl;
                }
            } catch (const std::exception&) {
                // Default config file doesn't exist or is invalid, use defaults
                if (options.verbose) {
                    std::cout << "Using default configuration (no config file found)." << std::endl;
                }
            }
        }
        
        // Apply command-line overrides
        cli_parser.applyToConfiguration(config);
        
        // Print configuration if verbose
        if (options.verbose) {
            config.printConfig();
        }
        
        std::cout << "Parallel Matrix Factorization Recommendation System" << std::endl;
        std::cout << "Version: " << PROJECT_VERSION << std::endl;
        std::cout << "Execution Mode: " << executionModeToString(options.mode) << std::endl;
        
        if (options.benchmark_mode) {
            std::cout << "\n=== Running Performance Benchmark ===" << std::endl;
            
            // Generate synthetic data for benchmarking
            std::vector<Rating> synthetic_ratings;
            int num_users = 200;
            int num_items = 100;
            int num_ratings = 5000;
            
            if (!options.quiet) {
                std::cout << "Generating synthetic dataset..." << std::endl;
                std::cout << "Users: " << num_users << ", Items: " << num_items << ", Ratings: " << num_ratings << std::endl;
            }
            
            // Simple synthetic data generation
            for (int i = 0; i < num_ratings; i++) {
                int user_id = rand() % num_users;
                int item_id = rand() % num_items;
                double rating = 1.0 + (rand() % 5);
                synthetic_ratings.push_back({user_id, item_id, rating, 0});
            }
            
            // Configure SGD from loaded configuration
            SGDConfig sgd_config;
            sgd_config.num_factors = config.getAlgorithm().num_factors;
            sgd_config.max_epochs = config.getAlgorithm().max_epochs;
            sgd_config.num_threads = config.getParallelism().num_threads;
            sgd_config.learning_rate = config.getAlgorithm().learning_rate;
            sgd_config.regularization = config.getAlgorithm().regularization;
            sgd_config.convergence_threshold = config.getAlgorithm().convergence_threshold;
            
            // Create factorizer and run benchmark based on execution mode
            SGDFactorizer factorizer(num_users, num_items, sgd_config);
            
            switch (options.mode) {
                case ExecutionMode::SERIAL:
                    if (!options.quiet) std::cout << "Running serial SGD training..." << std::endl;
                    factorizer.trainSerial(synthetic_ratings);
                    break;
                case ExecutionMode::OPENMP:
                    if (!options.quiet) std::cout << "Running OpenMP SGD training..." << std::endl;
                    {
                        PerformanceMetrics metrics = factorizer.benchmarkSerialVsOpenMP(synthetic_ratings);
                        if (config.getOutput().performance_report) {
                            factorizer.printPerformanceReport(metrics);
                        }
                    }
                    break;
                case ExecutionMode::MPI:
#ifdef USE_MPI
                    if (mpi_coordinator && mpi_coordinator->isMaster()) {
                        std::cout << "Running MPI distributed benchmark..." << std::endl;
                        std::cout << "Note: MPI benchmark requires multiple processes. Use: mpirun -np N ./main.exe --benchmark --mode mpi" << std::endl;
                    }
                    // MPI benchmark would need to be run with mpirun for meaningful results
#else
                    std::cout << "MPI benchmark requires MPI support, but MPI is not available in this build." << std::endl;
#endif
                    break;
                case ExecutionMode::HYBRID:
#ifdef USE_MPI
                    if (mpi_coordinator && mpi_coordinator->isMaster()) {
                        std::cout << "Running hybrid OpenMP+MPI benchmark..." << std::endl;
                        
                        // Distribute synthetic data across MPI processes
                        RatingMatrix synthetic_matrix;
                        // Note: In a real implementation, we'd need to create a proper matrix from synthetic_ratings
                        // For now, we'll indicate that hybrid benchmark needs proper data distribution
                        std::cout << "Hybrid benchmark requires proper data distribution implementation." << std::endl;
                    }
#else
                    std::cout << "Hybrid mode requires MPI support, but MPI is not available in this build." << std::endl;
#endif
                    break;
            }
            
        } else {
            if (!options.quiet) {
                std::cout << "\nData format: MovieLens .dat files with :: delimiters" << std::endl;
                std::cout << "Expected files: ratings.dat, movies.dat, users.dat" << std::endl;
                std::cout << "\nTo run performance benchmark: " << argv[0] << " --benchmark" << std::endl;
                std::cout << "For help: " << argv[0] << " --help" << std::endl;
            }
            
            // Try to load actual data if available
            RatingMatrix matrix;
            try {
                std::string data_file = config.getData().input_file;
                if (!options.quiet) {
                    std::cout << "\nAttempting to load " << data_file << "..." << std::endl;
                }
                matrix.loadFromDat(data_file);
                
                if (!options.quiet) {
                    std::cout << "Data loaded successfully!" << std::endl;
                    std::cout << "Users: " << matrix.getNumUsers() << std::endl;
                    std::cout << "Items: " << matrix.getNumItems() << std::endl;
                    std::cout << "Ratings: " << matrix.getRatings().size() << std::endl;
                }
                
                // Configure SGD from loaded configuration
                SGDConfig sgd_config;
                sgd_config.num_factors = config.getAlgorithm().num_factors;
                sgd_config.max_epochs = config.getAlgorithm().max_epochs;
                sgd_config.num_threads = config.getParallelism().num_threads;
                sgd_config.learning_rate = config.getAlgorithm().learning_rate;
                sgd_config.regularization = config.getAlgorithm().regularization;
                sgd_config.convergence_threshold = config.getAlgorithm().convergence_threshold;
                
                // Split data into training and validation sets if validation is enabled
                std::vector<Rating> train_ratings, validation_ratings;
                if (config.getData().validation_split) {
                    const auto& all_ratings = matrix.getRatings();
                    for (size_t i = 0; i < all_ratings.size(); i++) {
                        if (i % 5 == 0) { // 20% for validation
                            validation_ratings.push_back(all_ratings[i]);
                        } else { // 80% for training
                            train_ratings.push_back(all_ratings[i]);
                        }
                    }
                    if (!options.quiet) {
                        std::cout << "Train/Validation Split: " << train_ratings.size() << " / " << validation_ratings.size() << " ratings" << std::endl;
                    }
                } else {
                    train_ratings = matrix.getRatings();
                }
                
                SGDFactorizer factorizer(matrix.getNumUsers(), matrix.getNumItems(), sgd_config);
                
                // Run training based on execution mode
                double train_rmse = 0.0;
                switch (options.mode) {
                    case ExecutionMode::SERIAL:
                        if (!options.quiet) std::cout << "\nRunning serial SGD training..." << std::endl;
                        train_rmse = factorizer.trainSerial(train_ratings);
                        break;
                    case ExecutionMode::OPENMP:
                        if (!options.quiet) std::cout << "\nRunning OpenMP SGD training..." << std::endl;
                        train_rmse = factorizer.trainOpenMP(train_ratings);
                        break;
                    case ExecutionMode::MPI:
#ifdef USE_MPI
                        if (mpi_coordinator) {
                            if (mpi_coordinator->isMaster()) {
                                std::cout << "\nRunning MPI distributed training..." << std::endl;
                            }
                            
                            // Distribute data across MPI processes
                            mpi_coordinator->distributeData(matrix);
                            std::vector<Rating> local_ratings = mpi_coordinator->getLocalUserBlock(matrix);
                            
                            // Create factorizer for local user block
                            SGDFactorizer factorizer(mpi_coordinator->getLocalUserCount(), matrix.getNumItems(), sgd_config);
                            
                            // Run MPI training
                            train_rmse = factorizer.trainMPI(local_ratings, *mpi_coordinator, matrix.getNumUsers(), matrix.getNumItems());
                            
                            if (mpi_coordinator->isMaster()) {
                                std::cout << "MPI training completed. Local RMSE: " << train_rmse << std::endl;
                            }
                        } else {
                            std::cerr << "MPI coordinator not initialized for MPI mode." << std::endl;
                            return 1;
                        }
#else
                        std::cout << "MPI execution mode requires MPI support, but MPI is not available in this build." << std::endl;
                        return 1;
#endif
                        break;
                    case ExecutionMode::HYBRID:
#ifdef USE_MPI
                        if (mpi_coordinator) {
                            if (mpi_coordinator->isMaster()) {
                                std::cout << "\nRunning hybrid OpenMP+MPI training..." << std::endl;
                            }
                            
                            // Distribute data across MPI processes
                            mpi_coordinator->distributeData(matrix);
                            std::vector<Rating> local_ratings = mpi_coordinator->getLocalUserBlock(matrix);
                            
                            // Create factorizer for local user block
                            SGDFactorizer factorizer(mpi_coordinator->getLocalUserCount(), matrix.getNumItems(), sgd_config);
                            
                            // Run hybrid training
                            train_rmse = factorizer.trainHybrid(local_ratings, *mpi_coordinator, matrix.getNumUsers(), matrix.getNumItems());
                            
                            if (mpi_coordinator->isMaster()) {
                                std::cout << "Hybrid training completed. Local RMSE: " << train_rmse << std::endl;
                            }
                        } else {
                            std::cerr << "MPI coordinator not initialized for hybrid mode." << std::endl;
                            return 1;
                        }
#else
                        std::cout << "Hybrid execution mode requires MPI support, but MPI is not available in this build." << std::endl;
                        return 1;
#endif
                        break;
                }
                
                // Calculate validation RMSE if validation split was used
                if (config.getData().validation_split && !validation_ratings.empty()) {
                    double validation_rmse = factorizer.calculateRMSE(validation_ratings);
                    if (!options.quiet) {
                        std::cout << "Training completed." << std::endl;
                        std::cout << "Training RMSE: " << std::fixed << std::setprecision(6) << train_rmse << std::endl;
                        std::cout << "Validation RMSE: " << std::fixed << std::setprecision(6) << validation_rmse << std::endl;
                    }
                } else {
                    if (!options.quiet) {
                        std::cout << "Training completed. Final RMSE: " << train_rmse << std::endl;
                    }
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Could not load data file: " << e.what() << std::endl;
                std::cerr << "Use --benchmark to run with synthetic data" << std::endl;
                return 1;
            }
        }
        
        
#ifdef USE_MPI
        // Cleanup MPI coordinator
        if (mpi_coordinator) {
            mpi_coordinator->finalize();
            delete mpi_coordinator;
        }
#endif
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        
#ifdef USE_MPI
        // Cleanup MPI coordinator in case of exception
        if (mpi_coordinator) {
            mpi_coordinator->finalize();
            delete mpi_coordinator;
        }
#endif
        
        return 1;
    }
}