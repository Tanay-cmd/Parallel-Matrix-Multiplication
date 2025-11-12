#include "mpi_coordinator.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <functional>
#include <iomanip>
#include <cmath>
#include <fstream>

MPICoordinator::MPICoordinator() : rank(0), size(1) {
}

MPICoordinator::~MPICoordinator() {
}

void MPICoordinator::initialize(int argc, char** argv) {
#ifdef USE_MPI
    int provided;
    int result = MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    if (result != MPI_SUCCESS) {
        throw std::runtime_error("Failed to initialize MPI");
    }
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (isMaster()) {
        std::cout << "MPI initialized successfully" << std::endl;
        std::cout << "Number of MPI processes: " << size << std::endl;
        std::cout << "Thread support level: ";
        switch (provided) {
            case MPI_THREAD_SINGLE:
                std::cout << "MPI_THREAD_SINGLE" << std::endl;
                break;
            case MPI_THREAD_FUNNELED:
                std::cout << "MPI_THREAD_FUNNELED" << std::endl;
                break;
            case MPI_THREAD_SERIALIZED:
                std::cout << "MPI_THREAD_SERIALIZED" << std::endl;
                break;
            case MPI_THREAD_MULTIPLE:
                std::cout << "MPI_THREAD_MULTIPLE" << std::endl;
                break;
            default:
                std::cout << "Unknown" << std::endl;
        }
    }
#else
    // Non-MPI mode - single process
    rank = 0;
    size = 1;
    if (argc > 0 && argv != nullptr) {
        std::cout << "MPI not available - running in single process mode" << std::endl;
    }
#endif
}

void MPICoordinator::distributeData(const RatingMatrix& matrix) {
    int num_users = matrix.getNumUsers();
    const std::vector<Rating>& ratings = matrix.getRatings();
    
    if (isMaster()) {
        std::cout << "=== MPI Data Distribution ===" << std::endl;
        std::cout << "Total users: " << num_users << std::endl;
        std::cout << "Total ratings: " << ratings.size() << std::endl;
        std::cout << "MPI processes: " << size << std::endl;
    }
    
    // Calculate user block boundaries using static block partitioning
    user_blocks.clear();
    user_blocks.resize(size + 1);
    
    // Simple block partitioning: divide users evenly across processes
    int users_per_process = num_users / size;
    int remaining_users = num_users % size;
    
    user_blocks[0] = 0; // Start with user 0
    for (int i = 0; i < size; i++) {
        int block_size = users_per_process;
        if (i < remaining_users) {
            block_size++; // Distribute remaining users to first processes
        }
        user_blocks[i + 1] = user_blocks[i] + block_size;
    }
    
    // Ensure the last block covers all users
    user_blocks[size] = num_users;
    
    if (isMaster()) {
        std::cout << "User block partitioning:" << std::endl;
        for (int i = 0; i < size; i++) {
            int start_user = user_blocks[i];
            int end_user = user_blocks[i + 1] - 1;
            int block_size = end_user - start_user + 1;
            std::cout << "  Process " << i << ": users " << start_user 
                      << " to " << end_user << " (size: " << block_size << ")" << std::endl;
        }
        
        // Verify load balancing
        verifyLoadBalancing(ratings);
    }
    
#ifdef USE_MPI
    // Broadcast user block boundaries to all processes
    MPI_Bcast(user_blocks.data(), user_blocks.size(), MPI_INT, 0, MPI_COMM_WORLD);
    
    if (!isMaster()) {
        std::cout << "Process " << rank << " received user block: " 
                  << user_blocks[rank] << " to " << (user_blocks[rank + 1] - 1) << std::endl;
    }
#endif
}

void MPICoordinator::verifyLoadBalancing(const std::vector<Rating>& ratings) {
    if (!isMaster()) return;
    
    std::cout << "\n=== Load Balancing Verification ===" << std::endl;
    
    // Count ratings per process block
    std::vector<int> ratings_per_process(size, 0);
    
    for (const auto& rating : ratings) {
        // Find which process this user belongs to
        for (int i = 0; i < size; i++) {
            if (rating.user_id >= user_blocks[i] && rating.user_id < user_blocks[i + 1]) {
                ratings_per_process[i]++;
                break;
            }
        }
    }
    
    // Calculate statistics
    int total_ratings = ratings.size();
    int min_ratings = *std::min_element(ratings_per_process.begin(), ratings_per_process.end());
    int max_ratings = *std::max_element(ratings_per_process.begin(), ratings_per_process.end());
    double avg_ratings = static_cast<double>(total_ratings) / size;
    
    std::cout << "Ratings distribution across processes:" << std::endl;
    for (int i = 0; i < size; i++) {
        double percentage = (static_cast<double>(ratings_per_process[i]) / total_ratings) * 100.0;
        double deviation = ((ratings_per_process[i] - avg_ratings) / avg_ratings) * 100.0;
        std::cout << "  Process " << i << ": " << ratings_per_process[i] 
                  << " ratings (" << percentage << "%, deviation: " 
                  << (deviation >= 0 ? "+" : "") << deviation << "%)" << std::endl;
    }
    
    // Load balancing metrics
    double load_imbalance = (static_cast<double>(max_ratings - min_ratings) / avg_ratings) * 100.0;
    std::cout << "\nLoad balancing metrics:" << std::endl;
    std::cout << "  Average ratings per process: " << avg_ratings << std::endl;
    std::cout << "  Min ratings: " << min_ratings << std::endl;
    std::cout << "  Max ratings: " << max_ratings << std::endl;
    std::cout << "  Load imbalance: " << load_imbalance << "%" << std::endl;
    
    if (load_imbalance > 20.0) {
        std::cout << "  WARNING: High load imbalance detected (> 20%)" << std::endl;
    } else {
        std::cout << "  Load balancing is acceptable (< 20% imbalance)" << std::endl;
    }
}

std::vector<Rating> MPICoordinator::getLocalUserBlock(const RatingMatrix& matrix) {
    if (user_blocks.empty()) {
        throw std::runtime_error("Data distribution not performed. Call distributeData() first.");
    }
    
    int start_user = user_blocks[rank];
    int end_user = user_blocks[rank + 1] - 1;
    
    std::vector<Rating> local_ratings;
    const std::vector<Rating>& all_ratings = matrix.getRatings();
    
    // Extract ratings for this process's user block
    for (const auto& rating : all_ratings) {
        if (rating.user_id >= start_user && rating.user_id <= end_user) {
            local_ratings.push_back(rating);
        }
    }
    
    std::cout << "Process " << rank << " extracted " << local_ratings.size() 
              << " ratings for users " << start_user << " to " << end_user << std::endl;
    
    return local_ratings;
}

int MPICoordinator::getLocalUserStart() const {
    if (user_blocks.empty()) {
        throw std::runtime_error("Data distribution not performed. Call distributeData() first.");
    }
    return user_blocks[rank];
}

int MPICoordinator::getLocalUserEnd() const {
    if (user_blocks.empty()) {
        throw std::runtime_error("Data distribution not performed. Call distributeData() first.");
    }
    return user_blocks[rank + 1] - 1;
}

int MPICoordinator::getLocalUserCount() const {
    if (user_blocks.empty()) {
        throw std::runtime_error("Data distribution not performed. Call distributeData() first.");
    }
    return user_blocks[rank + 1] - user_blocks[rank];
}

void MPICoordinator::synchronizeFactors(Matrix& item_factors) {
#ifdef USE_MPI
    int num_items = item_factors.getRows();
    int num_factors = item_factors.getCols();
    
    // Flatten the item factor matrix for MPI communication
    std::vector<double> local_factors(num_items * num_factors);
    std::vector<double> global_factors(num_items * num_factors);
    
    // Copy local item factors to flat array
    for (int i = 0; i < num_items; i++) {
        for (int f = 0; f < num_factors; f++) {
            local_factors[i * num_factors + f] = item_factors[i][f];
        }
    }
    
    // Synchronize item factors across all processes using MPI_Allreduce
    int result = MPI_Allreduce(local_factors.data(), global_factors.data(), 
                              num_items * num_factors, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    if (result != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Allreduce failed during factor synchronization");
    }
    
    // Average the factors (since we summed across processes)
    for (int i = 0; i < num_items * num_factors; i++) {
        global_factors[i] /= size;
    }
    
    // Copy averaged factors back to the matrix
    for (int i = 0; i < num_items; i++) {
        for (int f = 0; f < num_factors; f++) {
            item_factors[i][f] = global_factors[i * num_factors + f];
        }
    }
    
    if (isMaster()) {
        std::cout << "Item factors synchronized across " << size << " processes" << std::endl;
    }
#else
    // Single process mode - no synchronization needed
    if (rank == 0) {
        std::cout << "Single process mode - no factor synchronization needed" << std::endl;
    }
#endif
}

void MPICoordinator::gatherResults(const Matrix& local_user_factors, Matrix& global_user_factors) {
#ifdef USE_MPI
    int local_user_count = getLocalUserCount();
    int num_factors = local_user_factors.getCols();
    
    // Prepare local user factors for gathering
    std::vector<double> local_data(local_user_count * num_factors);
    for (int u = 0; u < local_user_count; u++) {
        for (int f = 0; f < num_factors; f++) {
            local_data[u * num_factors + f] = local_user_factors[u][f];
        }
    }
    
    // Calculate receive counts and displacements for MPI_Gatherv
    std::vector<int> recv_counts(size);
    std::vector<int> displacements(size);
    
    // Each process sends its local user count
    int local_count = local_user_count * num_factors;
    MPI_Gather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (isMaster()) {
        // Calculate displacements
        displacements[0] = 0;
        for (int i = 1; i < size; i++) {
            displacements[i] = displacements[i-1] + recv_counts[i-1];
        }
        
        // Prepare buffer for all user factors
        int total_elements = 0;
        for (int i = 0; i < size; i++) {
            total_elements += recv_counts[i];
        }
        
        std::vector<double> all_user_data(total_elements);
        
        // Gather all user factors at master
        int result = MPI_Gatherv(local_data.data(), local_count, MPI_DOUBLE,
                                all_user_data.data(), recv_counts.data(), 
                                displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        if (result != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Gatherv failed during result gathering");
        }
        
        // Copy gathered data to global user factors matrix
        int data_index = 0;
        for (int proc = 0; proc < size; proc++) {
            int proc_user_start = user_blocks[proc];
            int proc_user_count = user_blocks[proc + 1] - user_blocks[proc];
            
            for (int u = 0; u < proc_user_count; u++) {
                int global_user_id = proc_user_start + u;
                for (int f = 0; f < num_factors; f++) {
                    global_user_factors[global_user_id][f] = all_user_data[data_index++];
                }
            }
        }
        
        std::cout << "User factors gathered from " << size << " processes" << std::endl;
    } else {
        // Worker processes just send their data
        MPI_Gatherv(local_data.data(), local_count, MPI_DOUBLE,
                   nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
#else
    // Single process mode - just copy local to global
    if (rank == 0) {
        for (int u = 0; u < local_user_factors.getRows(); u++) {
            for (int f = 0; f < local_user_factors.getCols(); f++) {
                global_user_factors[u][f] = local_user_factors[u][f];
            }
        }
        std::cout << "Single process mode - copied local user factors to global" << std::endl;
    }
#endif
}

bool MPICoordinator::validateResults(const SGDFactorizer& mpi_factorizer, 
                                     const SGDFactorizer& serial_baseline,
                                     const std::vector<Rating>& validation_ratings,
                                     double tolerance) {
    if (!isMaster()) {
        return true; // Only master validates results
    }
    
    std::cout << "\n=== MPI Result Validation ===" << std::endl;
    
    // Calculate RMSE for both implementations
    double mpi_rmse = mpi_factorizer.calculateRMSE(validation_ratings);
    double serial_rmse = serial_baseline.calculateRMSE(validation_ratings);
    
    double rmse_difference = std::abs(mpi_rmse - serial_rmse);
    double relative_error = (serial_rmse > 0) ? (rmse_difference / serial_rmse) * 100.0 : 0.0;
    
    std::cout << "Serial baseline RMSE:    " << std::fixed << std::setprecision(6) << serial_rmse << std::endl;
    std::cout << "MPI distributed RMSE:    " << std::fixed << std::setprecision(6) << mpi_rmse << std::endl;
    std::cout << "RMSE difference:         " << std::fixed << std::setprecision(6) << rmse_difference << std::endl;
    std::cout << "Relative error:          " << std::fixed << std::setprecision(2) << relative_error << "%" << std::endl;
    std::cout << "Tolerance:               " << tolerance << std::endl;
    
    bool results_match = rmse_difference <= tolerance;
    
    if (results_match) {
        std::cout << "✓ MPI results match serial baseline within tolerance" << std::endl;
    } else {
        std::cout << "✗ MPI results differ from serial baseline by more than tolerance" << std::endl;
        std::cout << "  This may indicate:" << std::endl;
        std::cout << "  - Synchronization issues in distributed training" << std::endl;
        std::cout << "  - Different convergence due to parallel updates" << std::endl;
        std::cout << "  - Numerical precision differences" << std::endl;
    }
    
    return results_match;
}

MPIPerformanceMetrics MPICoordinator::measurePerformance(
    const std::function<double()>& serial_training,
    const std::function<double()>& mpi_training) {
    
    MPIPerformanceMetrics metrics;
    
    if (!isMaster()) {
        // Worker processes just participate in MPI training
        mpi_training();
        return metrics;
    }
    
    std::cout << "\n=== MPI Performance Measurement ===" << std::endl;
    
    // Measure serial baseline performance
    std::cout << "Running serial baseline..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    metrics.serial_rmse = serial_training();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    metrics.serial_time_ms = duration.count() / 1000.0;
    
    std::cout << "Serial training completed in " << metrics.serial_time_ms << " ms" << std::endl;
    
    // Measure MPI distributed performance
    std::cout << "Running MPI distributed training..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    
    metrics.mpi_rmse = mpi_training();
    
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    metrics.mpi_time_ms = duration.count() / 1000.0;
    
    std::cout << "MPI training completed in " << metrics.mpi_time_ms << " ms" << std::endl;
    
    // Calculate performance metrics
    if (metrics.serial_time_ms > 0) {
        metrics.speedup = metrics.serial_time_ms / metrics.mpi_time_ms;
        metrics.efficiency = metrics.speedup / size;
    }
    
    metrics.rmse_difference = std::abs(metrics.mpi_rmse - metrics.serial_rmse);
    metrics.results_match = metrics.rmse_difference <= 0.01; // Default tolerance
    
    return metrics;
}

MPIPerformanceMetrics MPICoordinator::measureHybridPerformance(
    const std::function<double()>& serial_training,
    const std::function<double()>& hybrid_training) {
    
    MPIPerformanceMetrics metrics;
    
    if (!isMaster()) {
        // Worker processes just participate in hybrid training
        hybrid_training();
        return metrics;
    }
    
    std::cout << "\n=== Hybrid OpenMP+MPI Performance Measurement ===" << std::endl;
    
    // Measure serial baseline performance
    std::cout << "Running serial baseline..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    metrics.serial_rmse = serial_training();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    metrics.serial_time_ms = duration.count() / 1000.0;
    
    std::cout << "Serial training completed in " << metrics.serial_time_ms << " ms" << std::endl;
    
    // Measure hybrid distributed performance
    std::cout << "Running hybrid OpenMP+MPI training..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    
    metrics.mpi_rmse = hybrid_training();
    
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    metrics.mpi_time_ms = duration.count() / 1000.0;
    
    std::cout << "Hybrid training completed in " << metrics.mpi_time_ms << " ms" << std::endl;
    
    // Calculate performance metrics
    if (metrics.serial_time_ms > 0) {
        metrics.speedup = metrics.serial_time_ms / metrics.mpi_time_ms;
        metrics.efficiency = metrics.speedup / size;
    }
    
    metrics.rmse_difference = std::abs(metrics.mpi_rmse - metrics.serial_rmse);
    metrics.results_match = metrics.rmse_difference <= 0.01; // Default tolerance
    
    return metrics;
}

void MPICoordinator::printPerformanceReport(const MPIPerformanceMetrics& metrics) const {
    if (!isMaster()) return;
    
    std::cout << "\n=== MPI Performance Report ===" << std::endl;
    std::cout << "Serial execution time:   " << std::fixed << std::setprecision(2) << metrics.serial_time_ms << " ms" << std::endl;
    std::cout << "MPI execution time:      " << std::fixed << std::setprecision(2) << metrics.mpi_time_ms << " ms" << std::endl;
    std::cout << "Speedup:                 " << std::fixed << std::setprecision(2) << metrics.speedup << "x" << std::endl;
    std::cout << "Efficiency:              " << std::fixed << std::setprecision(2) << (metrics.efficiency * 100) << "%" << std::endl;
    std::cout << "MPI processes:           " << size << std::endl;
    
    std::cout << "\n=== Accuracy Comparison ===" << std::endl;
    std::cout << "Serial RMSE:             " << std::fixed << std::setprecision(6) << metrics.serial_rmse << std::endl;
    std::cout << "MPI RMSE:                " << std::fixed << std::setprecision(6) << metrics.mpi_rmse << std::endl;
    std::cout << "RMSE difference:         " << std::fixed << std::setprecision(6) << metrics.rmse_difference << std::endl;
    std::cout << "Results match (±0.01):   " << (metrics.results_match ? "✓ YES" : "✗ NO") << std::endl;
    
    // Performance analysis
    std::cout << "\n=== Performance Analysis ===" << std::endl;
    if (metrics.speedup > 1.0) {
        std::cout << "✓ MPI implementation achieved " << std::fixed << std::setprecision(2) 
                  << metrics.speedup << "x speedup!" << std::endl;
        
        if (metrics.efficiency > 0.7) {
            std::cout << "✓ Excellent efficiency (" << std::fixed << std::setprecision(1) 
                      << (metrics.efficiency * 100) << "%)" << std::endl;
        } else if (metrics.efficiency > 0.5) {
            std::cout << "⚠ Good efficiency (" << std::fixed << std::setprecision(1) 
                      << (metrics.efficiency * 100) << "%)" << std::endl;
        } else {
            std::cout << "⚠ Low efficiency (" << std::fixed << std::setprecision(1) 
                      << (metrics.efficiency * 100) << "%) - consider:" << std::endl;
            std::cout << "  - Larger dataset size" << std::endl;
            std::cout << "  - Fewer processes" << std::endl;
            std::cout << "  - Optimizing communication overhead" << std::endl;
        }
    } else {
        std::cout << "⚠ MPI implementation is slower than serial. Possible causes:" << std::endl;
        std::cout << "  - Communication overhead exceeds computation benefits" << std::endl;
        std::cout << "  - Dataset too small for effective parallelization" << std::endl;
        std::cout << "  - Load imbalance across processes" << std::endl;
        std::cout << "  - Synchronization bottlenecks" << std::endl;
    }
    
    if (!metrics.results_match) {
        std::cout << "\n⚠ WARNING: MPI and serial results differ significantly!" << std::endl;
        std::cout << "  This may indicate correctness issues in the distributed implementation." << std::endl;
    }
}

void MPICoordinator::printHybridPerformanceReport(const MPIPerformanceMetrics& metrics, int threads_per_process) const {
    if (!isMaster()) return;
    
    int total_cores = size * threads_per_process;
    
    std::cout << "\n=== Hybrid OpenMP+MPI Performance Report ===" << std::endl;
    std::cout << "Serial execution time:   " << std::fixed << std::setprecision(2) << metrics.serial_time_ms << " ms" << std::endl;
    std::cout << "Hybrid execution time:   " << std::fixed << std::setprecision(2) << metrics.mpi_time_ms << " ms" << std::endl;
    std::cout << "Speedup:                 " << std::fixed << std::setprecision(2) << metrics.speedup << "x" << std::endl;
    std::cout << "Efficiency:              " << std::fixed << std::setprecision(2) << (metrics.efficiency * 100) << "%" << std::endl;
    std::cout << "MPI processes:           " << size << std::endl;
    std::cout << "OpenMP threads/process:  " << threads_per_process << std::endl;
    std::cout << "Total cores used:        " << total_cores << std::endl;
    
    std::cout << "\n=== Accuracy Comparison ===" << std::endl;
    std::cout << "Serial RMSE:             " << std::fixed << std::setprecision(6) << metrics.serial_rmse << std::endl;
    std::cout << "Hybrid RMSE:             " << std::fixed << std::setprecision(6) << metrics.mpi_rmse << std::endl;
    std::cout << "RMSE difference:         " << std::fixed << std::setprecision(6) << metrics.rmse_difference << std::endl;
    std::cout << "Results match (±0.01):   " << (metrics.results_match ? "✓ YES" : "✗ NO") << std::endl;
    
    // Performance analysis
    std::cout << "\n=== Hybrid Performance Analysis ===" << std::endl;
    if (metrics.speedup > 1.0) {
        std::cout << "✓ Hybrid implementation achieved " << std::fixed << std::setprecision(2) 
                  << metrics.speedup << "x speedup using " << total_cores << " cores!" << std::endl;
        
        double theoretical_max_speedup = total_cores;
        double parallel_efficiency = metrics.speedup / theoretical_max_speedup;
        
        std::cout << "Parallel efficiency:     " << std::fixed << std::setprecision(2) 
                  << (parallel_efficiency * 100) << "% (vs theoretical max)" << std::endl;
        
        if (parallel_efficiency > 0.7) {
            std::cout << "✓ Excellent parallel efficiency!" << std::endl;
        } else if (parallel_efficiency > 0.5) {
            std::cout << "⚠ Good parallel efficiency" << std::endl;
        } else {
            std::cout << "⚠ Low parallel efficiency - consider:" << std::endl;
            std::cout << "  - Larger dataset size" << std::endl;
            std::cout << "  - Fewer total cores" << std::endl;
            std::cout << "  - Optimizing OpenMP thread synchronization" << std::endl;
            std::cout << "  - Reducing MPI communication overhead" << std::endl;
        }
        
        // Compare with pure MPI efficiency
        if (metrics.efficiency > 0.7) {
            std::cout << "✓ Excellent MPI process efficiency (" << std::fixed << std::setprecision(1) 
                      << (metrics.efficiency * 100) << "%)" << std::endl;
        } else if (metrics.efficiency > 0.5) {
            std::cout << "⚠ Good MPI process efficiency (" << std::fixed << std::setprecision(1) 
                      << (metrics.efficiency * 100) << "%)" << std::endl;
        } else {
            std::cout << "⚠ Low MPI process efficiency (" << std::fixed << std::setprecision(1) 
                      << (metrics.efficiency * 100) << "%)" << std::endl;
        }
    } else {
        std::cout << "⚠ Hybrid implementation is slower than serial. Possible causes:" << std::endl;
        std::cout << "  - Combined OpenMP+MPI overhead exceeds computation benefits" << std::endl;
        std::cout << "  - Dataset too small for effective hybrid parallelization" << std::endl;
        std::cout << "  - Thread contention within MPI processes" << std::endl;
        std::cout << "  - MPI communication bottlenecks" << std::endl;
        std::cout << "  - Load imbalance across processes or threads" << std::endl;
    }
    
    if (!metrics.results_match) {
        std::cout << "\n⚠ WARNING: Hybrid and serial results differ significantly!" << std::endl;
        std::cout << "  This may indicate correctness issues in the hybrid implementation." << std::endl;
    }
    
    std::cout << "\n=== Hybrid Mode Recommendations ===" << std::endl;
    std::cout << "For optimal hybrid performance:" << std::endl;
    std::cout << "  - Use fewer MPI processes with more OpenMP threads per process" << std::endl;
    std::cout << "  - Ensure MPI processes are bound to separate NUMA nodes" << std::endl;
    std::cout << "  - Balance computation vs communication workload" << std::endl;
    std::cout << "  - Consider dataset size relative to total core count" << std::endl;
}

MPICoordinator::ScalabilityMetrics MPICoordinator::analyzeScalability(
    const std::vector<int>& process_counts,
    const std::function<double()>& serial_training,
    const std::function<double(int)>& mpi_training_with_processes) {
    
    ScalabilityMetrics metrics;
    
    if (!isMaster()) {
        // Worker processes just participate in training
        for (int proc_count : process_counts) {
            if (proc_count <= size) {
                mpi_training_with_processes(proc_count);
            }
        }
        return metrics;
    }
    
    std::cout << "\n=== MPI Scalability Analysis ===" << std::endl;
    std::cout << "Testing process counts: ";
    for (size_t i = 0; i < process_counts.size(); i++) {
        std::cout << process_counts[i];
        if (i < process_counts.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    
    // Measure serial baseline
    std::cout << "\nMeasuring serial baseline..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    metrics.serial_baseline_rmse = serial_training();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    metrics.serial_baseline_time_ms = duration.count() / 1000.0;
    
    std::cout << "Serial baseline: " << metrics.serial_baseline_time_ms << " ms, RMSE: " 
              << std::fixed << std::setprecision(6) << metrics.serial_baseline_rmse << std::endl;
    
    // Test each process count
    for (int proc_count : process_counts) {
        if (proc_count > size) {
            std::cout << "\nSkipping " << proc_count << " processes (only " << size << " available)" << std::endl;
            continue;
        }
        
        std::cout << "\nTesting with " << proc_count << " processes..." << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
        
        double mpi_rmse = mpi_training_with_processes(proc_count);
        
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double mpi_time_ms = duration.count() / 1000.0;
        
        // Calculate performance metrics
        double speedup = (metrics.serial_baseline_time_ms > 0) ? 
                        (metrics.serial_baseline_time_ms / mpi_time_ms) : 0.0;
        double efficiency = (proc_count > 0) ? (speedup / proc_count) : 0.0;
        
        // Store results
        metrics.process_counts.push_back(proc_count);
        metrics.execution_times_ms.push_back(mpi_time_ms);
        metrics.speedups.push_back(speedup);
        metrics.efficiencies.push_back(efficiency);
        metrics.rmse_values.push_back(mpi_rmse);
        
        std::cout << "  Time: " << std::fixed << std::setprecision(2) << mpi_time_ms << " ms" << std::endl;
        std::cout << "  RMSE: " << std::fixed << std::setprecision(6) << mpi_rmse << std::endl;
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        std::cout << "  Efficiency: " << std::fixed << std::setprecision(1) << (efficiency * 100) << "%" << std::endl;
    }
    
    return metrics;
}

void MPICoordinator::printScalabilityReport(const ScalabilityMetrics& metrics) const {
    if (!isMaster()) return;
    
    std::cout << "\n=== MPI Scalability Report ===" << std::endl;
    std::cout << "Serial baseline: " << std::fixed << std::setprecision(2) 
              << metrics.serial_baseline_time_ms << " ms, RMSE: " 
              << std::setprecision(6) << metrics.serial_baseline_rmse << std::endl;
    
    std::cout << "\nScalability Results:" << std::endl;
    std::cout << std::setw(10) << "Processes" 
              << std::setw(12) << "Time (ms)" 
              << std::setw(10) << "Speedup" 
              << std::setw(12) << "Efficiency" 
              << std::setw(12) << "RMSE" << std::endl;
    std::cout << std::string(56, '-') << std::endl;
    
    for (size_t i = 0; i < metrics.process_counts.size(); i++) {
        std::cout << std::setw(10) << metrics.process_counts[i]
                  << std::setw(12) << std::fixed << std::setprecision(2) << metrics.execution_times_ms[i]
                  << std::setw(10) << std::fixed << std::setprecision(2) << metrics.speedups[i] << "x"
                  << std::setw(11) << std::fixed << std::setprecision(1) << (metrics.efficiencies[i] * 100) << "%"
                  << std::setw(12) << std::fixed << std::setprecision(6) << metrics.rmse_values[i] << std::endl;
    }
    
    // Analysis
    std::cout << "\n=== Scalability Analysis ===" << std::endl;
    
    // Find best performance
    if (!metrics.speedups.empty()) {
        auto max_speedup_it = std::max_element(metrics.speedups.begin(), metrics.speedups.end());
        size_t best_idx = std::distance(metrics.speedups.begin(), max_speedup_it);
        
        std::cout << "Best speedup: " << std::fixed << std::setprecision(2) << *max_speedup_it 
                  << "x with " << metrics.process_counts[best_idx] << " processes" << std::endl;
        std::cout << "Best efficiency: " << std::fixed << std::setprecision(1) 
                  << (metrics.efficiencies[best_idx] * 100) << "%" << std::endl;
    }
    
    // Analyze scaling behavior
    if (metrics.speedups.size() >= 2) {
        bool good_scaling = true;
        for (size_t i = 1; i < metrics.speedups.size(); i++) {
            if (metrics.speedups[i] < metrics.speedups[i-1] * 0.8) {
                good_scaling = false;
                break;
            }
        }
        
        if (good_scaling) {
            std::cout << "✓ Good scaling behavior observed" << std::endl;
        } else {
            std::cout << "⚠ Scaling degradation detected - consider:" << std::endl;
            std::cout << "  - Communication overhead increasing with process count" << std::endl;
            std::cout << "  - Load imbalance becoming more significant" << std::endl;
            std::cout << "  - Dataset size may be too small for higher process counts" << std::endl;
        }
    }
    
    // RMSE consistency check
    double max_rmse_diff = 0.0;
    for (double rmse : metrics.rmse_values) {
        double diff = std::abs(rmse - metrics.serial_baseline_rmse);
        max_rmse_diff = std::max(max_rmse_diff, diff);
    }
    
    if (max_rmse_diff <= 0.01) {
        std::cout << "✓ RMSE consistency maintained across all process counts" << std::endl;
    } else {
        std::cout << "⚠ RMSE varies significantly across process counts (max diff: " 
                  << std::fixed << std::setprecision(6) << max_rmse_diff << ")" << std::endl;
        std::cout << "  This may indicate convergence differences in distributed training" << std::endl;
    }
}

void MPICoordinator::generatePerformanceCSV(const ScalabilityMetrics& metrics, const std::string& filename) const {
    if (!isMaster()) return;
    
    std::ofstream csv_file(filename);
    if (!csv_file.is_open()) {
        std::cout << "Warning: Could not create CSV file: " << filename << std::endl;
        return;
    }
    
    // Write CSV header
    csv_file << "Processes,Time_ms,Speedup,Efficiency,RMSE" << std::endl;
    
    // Write serial baseline
    csv_file << "1," << std::fixed << std::setprecision(6) 
             << metrics.serial_baseline_time_ms << ",1.0,1.0," 
             << metrics.serial_baseline_rmse << std::endl;
    
    // Write MPI results
    for (size_t i = 0; i < metrics.process_counts.size(); i++) {
        csv_file << metrics.process_counts[i] << ","
                 << std::fixed << std::setprecision(6) << metrics.execution_times_ms[i] << ","
                 << metrics.speedups[i] << ","
                 << metrics.efficiencies[i] << ","
                 << metrics.rmse_values[i] << std::endl;
    }
    
    csv_file.close();
    std::cout << "Performance data saved to: " << filename << std::endl;
}

void MPICoordinator::finalize() {
#ifdef USE_MPI
    if (isMaster()) {
        std::cout << "Finalizing MPI..." << std::endl;
    }
    MPI_Finalize();
#else
    if (rank == 0) {
        std::cout << "MPI finalization - single process mode" << std::endl;
    }
#endif
}