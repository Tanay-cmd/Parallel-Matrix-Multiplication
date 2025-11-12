#ifndef MPI_COORDINATOR_H
#define MPI_COORDINATOR_H

#ifdef USE_MPI
#include <mpi.h>
#else
// Mock MPI definitions for testing without MPI
#include <cstring>
#define MPI_COMM_WORLD 0
#define MPI_SUCCESS 0
#define MPI_DOUBLE 0
#define MPI_INT 0
#define MPI_SUM 0
#define MPI_THREAD_SINGLE 0
#define MPI_THREAD_FUNNELED 1
#define MPI_THREAD_SERIALIZED 2
#define MPI_THREAD_MULTIPLE 3
typedef int MPI_Comm;
inline int MPI_Init_thread(int*, char***, int, int*) { return 0; }
inline int MPI_Comm_rank(MPI_Comm, int* rank) { *rank = 0; return 0; }
inline int MPI_Comm_size(MPI_Comm, int* size) { *size = 1; return 0; }
inline int MPI_Bcast(void*, int, int, int, MPI_Comm) { return 0; }
inline int MPI_Allreduce(void* send, void* recv, int count, int, int, MPI_Comm) { 
    memcpy(recv, send, count * sizeof(double)); return 0; 
}
inline int MPI_Gather(void*, int, int, void*, int*, int, int, MPI_Comm) { return 0; }
inline int MPI_Gatherv(void*, int, int, void*, int*, int*, int, int, MPI_Comm) { return 0; }
inline int MPI_Finalize() { return 0; }
#endif

#include "rating_matrix.h"
#include "sgd_factorizer.h"
#include <vector>
#include <functional>

struct MPIPerformanceMetrics {
    double serial_time_ms = 0.0;
    double mpi_time_ms = 0.0;
    double serial_rmse = 0.0;
    double mpi_rmse = 0.0;
    double speedup = 0.0;
    double efficiency = 0.0;
    bool results_match = false;
    double rmse_difference = 0.0;
};

class MPICoordinator {
private:
    int rank;
    int size;
    std::vector<int> user_blocks;
    
    void verifyLoadBalancing(const std::vector<Rating>& ratings);

public:
    MPICoordinator();
    ~MPICoordinator();
    
    void initialize(int argc, char** argv);
    void distributeData(const RatingMatrix& matrix);
    void synchronizeFactors(Matrix& item_factors);
    void gatherResults(const Matrix& local_user_factors, Matrix& global_user_factors);
    bool validateResults(const SGDFactorizer& mpi_factorizer, 
                        const SGDFactorizer& serial_baseline,
                        const std::vector<Rating>& validation_ratings,
                        double tolerance = 0.01);
    
    MPIPerformanceMetrics measurePerformance(
        const std::function<double()>& serial_training,
        const std::function<double()>& mpi_training);
    
    MPIPerformanceMetrics measureHybridPerformance(
        const std::function<double()>& serial_training,
        const std::function<double()>& hybrid_training);
    
    void printPerformanceReport(const MPIPerformanceMetrics& metrics) const;
    void printHybridPerformanceReport(const MPIPerformanceMetrics& metrics, int threads_per_process) const;
    
    // Enhanced performance analysis methods
    struct ScalabilityMetrics {
        std::vector<int> process_counts;
        std::vector<double> execution_times_ms;
        std::vector<double> speedups;
        std::vector<double> efficiencies;
        std::vector<double> rmse_values;
        double serial_baseline_time_ms = 0.0;
        double serial_baseline_rmse = 0.0;
    };
    
    ScalabilityMetrics analyzeScalability(
        const std::vector<int>& process_counts,
        const std::function<double()>& serial_training,
        const std::function<double(int)>& mpi_training_with_processes);
    
    void printScalabilityReport(const ScalabilityMetrics& metrics) const;
    void generatePerformanceCSV(const ScalabilityMetrics& metrics, const std::string& filename) const;
    void finalize();
    
    // Data access methods for local user blocks
    std::vector<Rating> getLocalUserBlock(const RatingMatrix& matrix);
    int getLocalUserStart() const;
    int getLocalUserEnd() const;
    int getLocalUserCount() const;
    
    int getRank() const { return rank; }
    int getSize() const { return size; }
    bool isMaster() const { return rank == 0; }
};

#endif // MPI_COORDINATOR_H