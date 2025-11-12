#include "sgd_factorizer.h"
#include "mpi_coordinator.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <set>
#include <iomanip>
#include <fstream>

#ifdef USE_OPENMP
#include <omp.h>
#endif

// Matrix class implementation
Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols) {
    data.resize(rows, std::vector<double>(cols, 0.0));
}

void Matrix::initialize(double min_val, double max_val) {
    // Use fixed seed for reproducible results and avoid antivirus triggers
    std::mt19937 gen(123);
    std::uniform_real_distribution<double> dist(min_val, max_val);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = dist(gen);
        }
    }
}

// SGDFactorizer class implementation
SGDFactorizer::SGDFactorizer(int num_users, int num_items, const SGDConfig& cfg)
    : U(num_users, cfg.num_factors), V(num_items, cfg.num_factors), config(cfg), model_trained(false) {
    // Initialize matrices with random values
    U.initialize(-0.1, 0.1);
    V.initialize(-0.1, 0.1);
}

void SGDFactorizer::initializeMatrices(int num_users, int num_items, int num_factors) {
    // Update config if needed
    config.num_factors = num_factors;
    
    // Reinitialize matrices with new dimensions
    U = Matrix(num_users, num_factors);
    V = Matrix(num_items, num_factors);
    
    U.initialize(-0.1, 0.1);
    V.initialize(-0.1, 0.1);
}

double SGDFactorizer::trainSerial(const std::vector<Rating>& ratings) {
    if (ratings.empty()) {
        std::cout << "No ratings provided for training" << std::endl;
        return 0.0;
    }
    
    // Use fixed seed for reproducible results and avoid antivirus triggers
    std::mt19937 g(42);
    double prev_rmse = std::numeric_limits<double>::max();
    
    for (int epoch = 0; epoch < config.max_epochs; epoch++) {
        // Shuffle ratings for better convergence (using fixed seed generator)
        std::vector<Rating> shuffled_ratings = ratings;
        std::shuffle(shuffled_ratings.begin(), shuffled_ratings.end(), g);
        
        // Perform SGD updates for each rating
        for (const auto& rating : shuffled_ratings) {
            int user_id = rating.user_id;
            int item_id = rating.item_id;
            double actual_rating = rating.rating;
            
            // Bounds checking to prevent crashes
            if (user_id < 0 || user_id >= U.getRows() || 
                item_id < 0 || item_id >= V.getRows()) {
                continue; // Skip invalid ratings
            }
            
            // Predict rating using current factors
            double predicted_rating = 0.0;
            for (int f = 0; f < config.num_factors; f++) {
                predicted_rating += U[user_id][f] * V[item_id][f];
            }
            
            // Calculate prediction error
            double error = actual_rating - predicted_rating;
            
            // Update factors using SGD
            for (int f = 0; f < config.num_factors; f++) {
                double user_factor = U[user_id][f];
                double item_factor = V[item_id][f];
                
                // SGD update with regularization
                U[user_id][f] += config.learning_rate * (error * item_factor - config.regularization * user_factor);
                V[item_id][f] += config.learning_rate * (error * user_factor - config.regularization * item_factor);
            }
        }
        
        // Calculate RMSE for convergence check (every 5 epochs to reduce overhead)
        if (epoch % 5 == 0 || epoch == config.max_epochs - 1) {
            double current_rmse = calculateRMSE(ratings);
            
            // Check for convergence
            if (epoch > 0 && std::abs(prev_rmse - current_rmse) < config.convergence_threshold) {
                std::cout << "Converged after " << (epoch + 1) << " epochs. RMSE: " << current_rmse << std::endl;
                return current_rmse;
            }
            
            prev_rmse = current_rmse;
            
            // Print progress every 10 epochs
            if ((epoch + 1) % 10 == 0) {
                std::cout << "Epoch " << (epoch + 1) << ", RMSE: " << current_rmse << std::endl;
            }
        }
    }
    
    double final_rmse = calculateRMSE(ratings);
    std::cout << "Training completed after " << config.max_epochs << " epochs. Final RMSE: " << final_rmse << std::endl;
    model_trained = true;
    return final_rmse;
}

double SGDFactorizer::trainOpenMP(const std::vector<Rating>& ratings) {
#ifdef USE_OPENMP
    if (ratings.empty()) {
        std::cout << "No ratings provided for OpenMP training" << std::endl;
        return 0.0;
    }
    
    // Set number of threads from config
    omp_set_num_threads(config.num_threads);
    
    // Use fixed seed for reproducible results
    std::mt19937 g(42);
    double prev_rmse = std::numeric_limits<double>::max();
    
    for (int epoch = 0; epoch < config.max_epochs; epoch++) {
        // Shuffle ratings for better convergence (using fixed seed)
        std::vector<Rating> shuffled_ratings = ratings;
        std::shuffle(shuffled_ratings.begin(), shuffled_ratings.end(), g);
        
        // Perform SGD updates for each rating with OpenMP parallelization
        #pragma omp parallel for schedule(dynamic) shared(shuffled_ratings)
        for (int i = 0; i < static_cast<int>(shuffled_ratings.size()); i++) {
            const Rating& rating = shuffled_ratings[i];
            int user_id = rating.user_id;
            int item_id = rating.item_id;
            double actual_rating = rating.rating;
            
            // Bounds checking to prevent crashes
            if (user_id < 0 || user_id >= U.getRows() || 
                item_id < 0 || item_id >= V.getRows()) {
                continue; // Skip invalid ratings
            }
            
            // Predict rating using current factors
            double predicted_rating = 0.0;
            for (int f = 0; f < config.num_factors; f++) {
                predicted_rating += U[user_id][f] * V[item_id][f];
            }
            
            // Calculate prediction error
            double error = actual_rating - predicted_rating;
            
            // Update factors using SGD with optimized thread-safe operations
            for (int f = 0; f < config.num_factors; f++) {
                double user_factor = U[user_id][f];
                double item_factor = V[item_id][f];
                
                // Calculate updates
                double user_update = config.learning_rate * (error * item_factor - config.regularization * user_factor);
                double item_update = config.learning_rate * (error * user_factor - config.regularization * item_factor);
                
                // Apply updates atomically to avoid race conditions
                // Note: Atomic operations cause contention - this is expected for SGD
                #pragma omp atomic
                U[user_id][f] += user_update;
                
                #pragma omp atomic
                V[item_id][f] += item_update;
            }
        }
        
        // Calculate RMSE for convergence check (done serially to avoid overhead)
        double current_rmse = calculateRMSE(ratings);
        
        // Check for convergence
        if (std::abs(prev_rmse - current_rmse) < config.convergence_threshold) {
            std::cout << "OpenMP training converged after " << (epoch + 1) << " epochs. RMSE: " << current_rmse << std::endl;
            model_trained = true;
            return current_rmse;
        }
        
        prev_rmse = current_rmse;
        
        // Print progress every 10 epochs
        if ((epoch + 1) % 10 == 0) {
            std::cout << "OpenMP Epoch " << (epoch + 1) << ", RMSE: " << current_rmse << std::endl;
        }
    }
    
    double final_rmse = calculateRMSE(ratings);
    std::cout << "OpenMP training completed after " << config.max_epochs << " epochs. Final RMSE: " << final_rmse << std::endl;
    model_trained = true;
    return final_rmse;
#else
    // Fallback to serial implementation if OpenMP is not available
    std::cout << "OpenMP not available, falling back to serial implementation." << std::endl;
    return trainSerial(ratings);
#endif
}

void SGDFactorizer::predictRatings(const std::vector<std::pair<int,int>>& user_item_pairs) {
    std::cout << "Predicting ratings for " << user_item_pairs.size() << " user-item pairs:" << std::endl;
    
    for (const auto& pair : user_item_pairs) {
        int user_id = pair.first;
        int item_id = pair.second;
        double predicted_rating = predictRating(user_id, item_id);
        
        std::cout << "User " << user_id << ", Item " << item_id 
                  << " -> Predicted Rating: " << predicted_rating << std::endl;
    }
}

std::vector<std::pair<int, double>> SGDFactorizer::generateTopNRecommendations(
    int user_id, int N, const std::vector<int>& rated_items) const {
    
    std::vector<std::pair<int, double>> item_scores;
    
    // Check if user_id is valid
    if (user_id < 0 || user_id >= U.getRows()) {
        return item_scores; // Return empty vector for invalid user
    }
    
    // Create set of rated items for fast lookup
    std::set<int> rated_set(rated_items.begin(), rated_items.end());
    
    // Calculate predicted ratings for all items
    for (int item_id = 0; item_id < V.getRows(); item_id++) {
        // Skip items that the user has already rated
        if (rated_set.find(item_id) != rated_set.end()) {
            continue;
        }
        
        double predicted_rating = predictRating(user_id, item_id);
        item_scores.push_back({item_id, predicted_rating});
    }
    
    // Sort items by predicted rating in descending order
    std::sort(item_scores.begin(), item_scores.end(), 
              [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                  return a.second > b.second;
              });
    
    // Return top N recommendations
    if (N > 0 && N < static_cast<int>(item_scores.size())) {
        item_scores.resize(N);
    }
    
    return item_scores;
}

std::vector<std::pair<int, double>> SGDFactorizer::rankItems(
    int user_id, const std::vector<int>& item_ids) const {
    
    std::vector<std::pair<int, double>> ranked_items;
    
    // Check if user_id is valid
    if (user_id < 0 || user_id >= U.getRows()) {
        return ranked_items; // Return empty vector for invalid user
    }
    
    // Calculate predicted ratings for specified items
    for (int item_id : item_ids) {
        if (item_id >= 0 && item_id < V.getRows()) {
            double predicted_rating = predictRating(user_id, item_id);
            ranked_items.push_back({item_id, predicted_rating});
        }
    }
    
    // Sort items by predicted rating in descending order
    std::sort(ranked_items.begin(), ranked_items.end(), 
              [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                  return a.second > b.second;
              });
    
    return ranked_items;
}

double SGDFactorizer::calculateRMSE(const std::vector<Rating>& ratings) const {
    double sum_squared_errors = 0.0;
    int count = 0;
    
    for (const auto& rating : ratings) {
        double predicted = predictRating(rating.user_id, rating.item_id);
        double error = rating.rating - predicted;
        sum_squared_errors += error * error;
        count++;
    }
    
    if (count == 0) return 0.0;
    
    return std::sqrt(sum_squared_errors / count);
}

double SGDFactorizer::predictRating(int user_id, int item_id) const {
    double prediction = 0.0;
    
    // Ensure indices are within bounds
    if (user_id >= 0 && user_id < U.getRows() && 
        item_id >= 0 && item_id < V.getRows()) {
        
        for (int f = 0; f < config.num_factors; f++) {
            prediction += U[user_id][f] * V[item_id][f];
        }
    }
    
    return prediction;
}

void SGDFactorizer::setNumThreads(int num_threads) {
    config.num_threads = num_threads;
#ifdef USE_OPENMP
    omp_set_num_threads(num_threads);
#endif
}

PerformanceMetrics SGDFactorizer::benchmarkSerialVsOpenMP(const std::vector<Rating>& ratings) {
    PerformanceMetrics metrics;
    
    std::cout << "\n=== Performance Benchmark: Serial vs OpenMP ===" << std::endl;
    std::cout << "Dataset size: " << ratings.size() << " ratings" << std::endl;
    std::cout << "Threads: " << config.num_threads << std::endl;
    std::cout << "Factors: " << config.num_factors << std::endl;
    std::cout << "Max epochs: " << config.max_epochs << std::endl;
    
    // Save original matrices for restoration
    Matrix original_U = U;
    Matrix original_V = V;
    
    // Benchmark serial implementation
    std::cout << "\nRunning serial implementation..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    metrics.serial_rmse = trainSerial(ratings);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    metrics.serial_time_ms = duration.count() / 1000.0; // Convert to milliseconds with decimal precision
    
    std::cout << "Serial training completed in " << metrics.serial_time_ms << " ms" << std::endl;
    std::cout << "Serial RMSE: " << metrics.serial_rmse << std::endl;
    
    // Save serial results
    Matrix serial_U = U;
    Matrix serial_V = V;
    
    // Restore original matrices for fair comparison
    U = original_U;
    V = original_V;
    
    // Benchmark OpenMP implementation
    std::cout << "\nRunning OpenMP implementation..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    
    metrics.openmp_rmse = trainOpenMP(ratings);
    
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    metrics.openmp_time_ms = duration.count() / 1000.0; // Convert to milliseconds with decimal precision
    
    std::cout << "OpenMP training completed in " << metrics.openmp_time_ms << " ms" << std::endl;
    std::cout << "OpenMP RMSE: " << metrics.openmp_rmse << std::endl;
    
    // Calculate performance metrics
    if (metrics.serial_time_ms > 0) {
        metrics.speedup = metrics.serial_time_ms / metrics.openmp_time_ms;
        metrics.efficiency = metrics.speedup / config.num_threads;
    }
    
    // Validate results
    metrics.rmse_difference = std::abs(metrics.serial_rmse - metrics.openmp_rmse);
    metrics.results_match = validateResults(metrics.serial_rmse, metrics.openmp_rmse);
    
    return metrics;
}

void SGDFactorizer::printPerformanceReport(const PerformanceMetrics& metrics) const {
    std::cout << "\n=== Performance Report ===" << std::endl;
    std::cout << "Serial execution time:   " << metrics.serial_time_ms << " ms" << std::endl;
    std::cout << "OpenMP execution time:   " << metrics.openmp_time_ms << " ms" << std::endl;
    std::cout << "Speedup:                 " << std::fixed << std::setprecision(2) << metrics.speedup << "x" << std::endl;
    std::cout << "Efficiency:              " << std::fixed << std::setprecision(2) << (metrics.efficiency * 100) << "%" << std::endl;
    std::cout << "Threads used:            " << config.num_threads << std::endl;
    
    std::cout << "\n=== Accuracy Validation ===" << std::endl;
    std::cout << "Serial RMSE:             " << std::fixed << std::setprecision(6) << metrics.serial_rmse << std::endl;
    std::cout << "OpenMP RMSE:             " << std::fixed << std::setprecision(6) << metrics.openmp_rmse << std::endl;
    std::cout << "RMSE difference:         " << std::fixed << std::setprecision(6) << metrics.rmse_difference << std::endl;
    std::cout << "Results match (±0.01):   " << (metrics.results_match ? "[YES]" : "[NO]") << std::endl;
    
    if (metrics.speedup > 1.0) {
        std::cout << "\n[SUCCESS] OpenMP implementation is " << std::fixed << std::setprecision(2) 
                  << metrics.speedup << "x faster than serial!" << std::endl;
        
        // Analyze scaling efficiency
        if (metrics.efficiency > 0.7) {
            std::cout << "[EXCELLENT] High parallel efficiency (>70%)" << std::endl;
        } else if (metrics.efficiency > 0.5) {
            std::cout << "[GOOD] Moderate parallel efficiency (50-70%)" << std::endl;
        } else if (metrics.efficiency > 0.25) {
            std::cout << "[FAIR] Low parallel efficiency (25-50%) - atomic contention expected in SGD" << std::endl;
        } else {
            std::cout << "[POOR] Very low parallel efficiency (<25%) - consider:" << std::endl;
            std::cout << "  - Larger dataset size" << std::endl;
            std::cout << "  - Fewer threads" << std::endl;
            std::cout << "  - Different parallelization strategy" << std::endl;
        }
    } else {
        std::cout << "\n[WARNING] OpenMP implementation is slower than serial. This may be due to:" << std::endl;
        std::cout << "  - Small dataset size (parallelization overhead)" << std::endl;
        std::cout << "  - Thread contention from atomic operations" << std::endl;
        std::cout << "  - System configuration" << std::endl;
        std::cout << "  - SGD algorithm inherently has limited parallelism" << std::endl;
    }
    
    if (!metrics.results_match) {
        std::cout << "\n[WARNING] Serial and OpenMP results differ by more than 0.01 RMSE!" << std::endl;
        std::cout << "  This may indicate a race condition or synchronization issue." << std::endl;
    }
}

double SGDFactorizer::trainMPI(const std::vector<Rating>& local_ratings, 
                              MPICoordinator& coordinator, int total_users, int total_items) {
    // Initialize matrices with correct dimensions for distributed training
    if (U.getRows() != coordinator.getLocalUserCount() || V.getRows() != total_items) {
        U = Matrix(coordinator.getLocalUserCount(), config.num_factors);
        V = Matrix(total_items, config.num_factors);
        U.initialize(-0.1, 0.1);
        V.initialize(-0.1, 0.1);
    }
    
    if (coordinator.isMaster()) {
        std::cout << "\n=== MPI Distributed SGD Training ===" << std::endl;
        std::cout << "Total users: " << total_users << std::endl;
        std::cout << "Total items: " << total_items << std::endl;
        std::cout << "Local ratings: " << local_ratings.size() << std::endl;
        std::cout << "Processes: " << coordinator.getSize() << std::endl;
        std::cout << "Factors: " << config.num_factors << std::endl;
    }
    
    double prev_rmse = std::numeric_limits<double>::max();
    int local_user_start = coordinator.getLocalUserStart();
    
    for (int epoch = 0; epoch < config.max_epochs; epoch++) {
        // Shuffle local ratings for better convergence
        std::vector<Rating> shuffled_ratings = local_ratings;
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(shuffled_ratings.begin(), shuffled_ratings.end(), g);
        
        // Perform SGD updates on local user block
        for (const auto& rating : shuffled_ratings) {
            int global_user_id = rating.user_id;
            int local_user_id = global_user_id - local_user_start;
            int item_id = rating.item_id;
            double actual_rating = rating.rating;
            
            // Validate indices
            if (local_user_id < 0 || local_user_id >= U.getRows() || 
                item_id < 0 || item_id >= V.getRows()) {
                continue; // Skip invalid ratings
            }
            
            // Predict rating using current factors
            double predicted_rating = 0.0;
            for (int f = 0; f < config.num_factors; f++) {
                predicted_rating += U[local_user_id][f] * V[item_id][f];
            }
            
            // Calculate prediction error
            double error = actual_rating - predicted_rating;
            
            // Update factors using SGD
            for (int f = 0; f < config.num_factors; f++) {
                double user_factor = U[local_user_id][f];
                double item_factor = V[item_id][f];
                
                // SGD update with regularization
                U[local_user_id][f] += config.learning_rate * (error * item_factor - config.regularization * user_factor);
                V[item_id][f] += config.learning_rate * (error * user_factor - config.regularization * item_factor);
            }
        }
        
        // Synchronize item factors across all processes
        try {
            coordinator.synchronizeFactors(V);
        } catch (const std::exception& e) {
            if (coordinator.isMaster()) {
                std::cout << "Error during factor synchronization: " << e.what() << std::endl;
            }
            throw;
        }
        
        // Calculate local RMSE
        double local_rmse = calculateLocalRMSE(local_ratings, local_user_start);
        
        // For convergence checking, we'll use the master's local RMSE as approximation
        // In a full implementation, we might want to gather all local RMSEs
        if (coordinator.isMaster()) {
            // Check for convergence
            if (std::abs(prev_rmse - local_rmse) < config.convergence_threshold) {
                std::cout << "MPI training converged after " << (epoch + 1) << " epochs. Local RMSE: " << local_rmse << std::endl;
                return local_rmse;
            }
            
            prev_rmse = local_rmse;
            
            // Print progress every 10 epochs
            if ((epoch + 1) % 10 == 0) {
                std::cout << "MPI Epoch " << (epoch + 1) << ", Local RMSE: " << local_rmse << std::endl;
            }
        }
    }
    
    double final_local_rmse = calculateLocalRMSE(local_ratings, local_user_start);
    if (coordinator.isMaster()) {
        std::cout << "MPI training completed after " << config.max_epochs << " epochs. Final local RMSE: " << final_local_rmse << std::endl;
    }
    
    return final_local_rmse;
}

double SGDFactorizer::trainHybrid(const std::vector<Rating>& local_ratings, 
                                  MPICoordinator& coordinator, int total_users, int total_items) {
#ifdef USE_MPI
#if defined(USE_OPENMP)
    // Initialize matrices with correct dimensions for distributed training
    if (U.getRows() != coordinator.getLocalUserCount() || V.getRows() != total_items) {
        U = Matrix(coordinator.getLocalUserCount(), config.num_factors);
        V = Matrix(total_items, config.num_factors);
        U.initialize(-0.1, 0.1);
        V.initialize(-0.1, 0.1);
    }
    
    // Set number of threads from config
    omp_set_num_threads(config.num_threads);
    
    if (coordinator.isMaster()) {
        std::cout << "\n=== Hybrid OpenMP+MPI SGD Training ===" << std::endl;
        std::cout << "Total users: " << total_users << std::endl;
        std::cout << "Total items: " << total_items << std::endl;
        std::cout << "Local ratings: " << local_ratings.size() << std::endl;
        std::cout << "MPI processes: " << coordinator.getSize() << std::endl;
        std::cout << "OpenMP threads per process: " << config.num_threads << std::endl;
        std::cout << "Factors: " << config.num_factors << std::endl;
    }
    
    double prev_rmse = std::numeric_limits<double>::max();
    int local_user_start = coordinator.getLocalUserStart();
    
    for (int epoch = 0; epoch < config.max_epochs; epoch++) {
        // Shuffle local ratings for better convergence
        std::vector<Rating> shuffled_ratings = local_ratings;
        std::mt19937 g(42 + epoch); // Use epoch-dependent seed for reproducibility
        std::shuffle(shuffled_ratings.begin(), shuffled_ratings.end(), g);
        
        // Perform SGD updates on local user block with OpenMP parallelization
        #pragma omp parallel for schedule(dynamic) shared(shuffled_ratings)
        for (int i = 0; i < static_cast<int>(shuffled_ratings.size()); i++) {
            const Rating& rating = shuffled_ratings[i];
            int global_user_id = rating.user_id;
            int local_user_id = global_user_id - local_user_start;
            int item_id = rating.item_id;
            double actual_rating = rating.rating;
            
            // Validate indices
            if (local_user_id < 0 || local_user_id >= U.getRows() || 
                item_id < 0 || item_id >= V.getRows()) {
                continue; // Skip invalid ratings
            }
            
            // Predict rating using current factors
            double predicted_rating = 0.0;
            for (int f = 0; f < config.num_factors; f++) {
                predicted_rating += U[local_user_id][f] * V[item_id][f];
            }
            
            // Calculate prediction error
            double error = actual_rating - predicted_rating;
            
            // Update factors using SGD with optimized thread-safe operations
            for (int f = 0; f < config.num_factors; f++) {
                double user_factor = U[local_user_id][f];
                double item_factor = V[item_id][f];
                
                // Calculate updates
                double user_update = config.learning_rate * (error * item_factor - config.regularization * user_factor);
                double item_update = config.learning_rate * (error * user_factor - config.regularization * item_factor);
                
                // Apply updates atomically to avoid race conditions
                // Note: Atomic operations cause contention - this is expected for SGD
                #pragma omp atomic
                U[local_user_id][f] += user_update;
                
                #pragma omp atomic
                V[item_id][f] += item_update;
            }
        }
        
        // Synchronize item factors across all MPI processes
        try {
            coordinator.synchronizeFactors(V);
        } catch (const std::exception& e) {
            if (coordinator.isMaster()) {
                std::cout << "Error during factor synchronization: " << e.what() << std::endl;
            }
            throw;
        }
        
        // Calculate local RMSE
        double local_rmse = calculateLocalRMSE(local_ratings, local_user_start);
        
        // For convergence checking, we'll use the master's local RMSE as approximation
        if (coordinator.isMaster()) {
            // Check for convergence
            if (std::abs(prev_rmse - local_rmse) < config.convergence_threshold) {
                std::cout << "Hybrid training converged after " << (epoch + 1) << " epochs. Local RMSE: " << local_rmse << std::endl;
                return local_rmse;
            }
            
            prev_rmse = local_rmse;
            
            // Print progress every 10 epochs
            if ((epoch + 1) % 10 == 0) {
                std::cout << "Hybrid Epoch " << (epoch + 1) << ", Local RMSE: " << local_rmse << std::endl;
            }
        }
    }
    
    double final_local_rmse = calculateLocalRMSE(local_ratings, local_user_start);
    if (coordinator.isMaster()) {
        std::cout << "Hybrid training completed after " << config.max_epochs << " epochs. Final local RMSE: " << final_local_rmse << std::endl;
    }
    
    return final_local_rmse;
#else
    // Fallback to MPI-only if OpenMP is not available
    if (coordinator.isMaster()) {
        std::cout << "OpenMP not available, falling back to MPI-only implementation." << std::endl;
    }
    return trainMPI(local_ratings, coordinator, total_users, total_items);
#endif
#else
    // No MPI available - cannot run hybrid mode
    (void)local_ratings; (void)coordinator; (void)total_users; (void)total_items;
    throw std::runtime_error("Hybrid mode requires MPI support, but MPI is not available in this build");
#endif
}

double SGDFactorizer::calculateLocalRMSE(const std::vector<Rating>& local_ratings, int local_user_start) const {
    double sum_squared_errors = 0.0;
    int count = 0;
    
    for (const auto& rating : local_ratings) {
        int global_user_id = rating.user_id;
        int local_user_id = global_user_id - local_user_start;
        int item_id = rating.item_id;
        
        // Validate indices
        if (local_user_id >= 0 && local_user_id < U.getRows() && 
            item_id >= 0 && item_id < V.getRows()) {
            
            double predicted = 0.0;
            for (int f = 0; f < config.num_factors; f++) {
                predicted += U[local_user_id][f] * V[item_id][f];
            }
            
            double error = rating.rating - predicted;
            sum_squared_errors += error * error;
            count++;
        }
    }
    
    if (count == 0) return 0.0;
    return std::sqrt(sum_squared_errors / count);
}

bool SGDFactorizer::validateResults(double serial_rmse, double openmp_rmse, double tolerance) const {
    return std::abs(serial_rmse - openmp_rmse) <= tolerance;
}

void SGDFactorizer::saveModel(const std::string& model_path) const {
    if (!model_trained) {
        throw std::runtime_error("Cannot save untrained model. Train the model first.");
    }
    
    std::ofstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + model_path);
    }
    
    // Write model metadata
    file.write(reinterpret_cast<const char*>(&model_trained), sizeof(model_trained));
    
    // Write config
    file.write(reinterpret_cast<const char*>(&config.learning_rate), sizeof(config.learning_rate));
    file.write(reinterpret_cast<const char*>(&config.regularization), sizeof(config.regularization));
    file.write(reinterpret_cast<const char*>(&config.num_factors), sizeof(config.num_factors));
    file.write(reinterpret_cast<const char*>(&config.max_epochs), sizeof(config.max_epochs));
    file.write(reinterpret_cast<const char*>(&config.convergence_threshold), sizeof(config.convergence_threshold));
    file.write(reinterpret_cast<const char*>(&config.num_threads), sizeof(config.num_threads));
    
    // Write matrix dimensions
    int u_rows = U.getRows(), u_cols = U.getCols();
    int v_rows = V.getRows(), v_cols = V.getCols();
    file.write(reinterpret_cast<const char*>(&u_rows), sizeof(u_rows));
    file.write(reinterpret_cast<const char*>(&u_cols), sizeof(u_cols));
    file.write(reinterpret_cast<const char*>(&v_rows), sizeof(v_rows));
    file.write(reinterpret_cast<const char*>(&v_cols), sizeof(v_cols));
    
    // Write U matrix
    for (int i = 0; i < u_rows; i++) {
        for (int j = 0; j < u_cols; j++) {
            double val = U[i][j];
            file.write(reinterpret_cast<const char*>(&val), sizeof(val));
        }
    }
    
    // Write V matrix
    for (int i = 0; i < v_rows; i++) {
        for (int j = 0; j < v_cols; j++) {
            double val = V[i][j];
            file.write(reinterpret_cast<const char*>(&val), sizeof(val));
        }
    }
    
    file.close();
    std::cout << "Model saved successfully to: " << model_path << std::endl;
}

void SGDFactorizer::loadModel(const std::string& model_path) {
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open model file: " + model_path);
    }
    
    // Read model metadata
    file.read(reinterpret_cast<char*>(&model_trained), sizeof(model_trained));
    
    // Read config
    file.read(reinterpret_cast<char*>(&config.learning_rate), sizeof(config.learning_rate));
    file.read(reinterpret_cast<char*>(&config.regularization), sizeof(config.regularization));
    file.read(reinterpret_cast<char*>(&config.num_factors), sizeof(config.num_factors));
    file.read(reinterpret_cast<char*>(&config.max_epochs), sizeof(config.max_epochs));
    file.read(reinterpret_cast<char*>(&config.convergence_threshold), sizeof(config.convergence_threshold));
    file.read(reinterpret_cast<char*>(&config.num_threads), sizeof(config.num_threads));
    
    // Read matrix dimensions
    int u_rows, u_cols, v_rows, v_cols;
    file.read(reinterpret_cast<char*>(&u_rows), sizeof(u_rows));
    file.read(reinterpret_cast<char*>(&u_cols), sizeof(u_cols));
    file.read(reinterpret_cast<char*>(&v_rows), sizeof(v_rows));
    file.read(reinterpret_cast<char*>(&v_cols), sizeof(v_cols));
    
    // Recreate matrices with correct dimensions
    U = Matrix(u_rows, u_cols);
    V = Matrix(v_rows, v_cols);
    
    // Read U matrix
    for (int i = 0; i < u_rows; i++) {
        for (int j = 0; j < u_cols; j++) {
            double val;
            file.read(reinterpret_cast<char*>(&val), sizeof(val));
            U[i][j] = val;
        }
    }
    
    // Read V matrix
    for (int i = 0; i < v_rows; i++) {
        for (int j = 0; j < v_cols; j++) {
            double val;
            file.read(reinterpret_cast<char*>(&val), sizeof(val));
            V[i][j] = val;
        }
    }
    
    file.close();
    std::cout << "Model loaded successfully from: " << model_path << std::endl;
    std::cout << "Model dimensions: " << u_rows << " users, " << v_rows << " items, " << u_cols << " factors" << std::endl;
}