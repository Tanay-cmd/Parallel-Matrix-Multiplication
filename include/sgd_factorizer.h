#ifndef SGD_FACTORIZER_H
#define SGD_FACTORIZER_H

#include "rating_matrix.h"
#include <vector>
#include <chrono>

struct SGDConfig {
    double learning_rate = 0.01;
    double regularization = 0.01;
    int num_factors = 50;
    int max_epochs = 100;
    double convergence_threshold = 0.001;
    int num_threads = 4;
};

struct PerformanceMetrics {
    double serial_time_ms = 0.0;
    double openmp_time_ms = 0.0;
    double serial_rmse = 0.0;
    double openmp_rmse = 0.0;
    double speedup = 0.0;
    double efficiency = 0.0;
    bool results_match = false;
    double rmse_difference = 0.0;
};

class Matrix {
private:
    std::vector<std::vector<double>> data;
    int rows, cols;

public:
    Matrix(int rows, int cols);
    void initialize(double min_val = -0.1, double max_val = 0.1);
    std::vector<double>& operator[](int row) { return data[row]; }
    const std::vector<double>& operator[](int row) const { return data[row]; }
    int getRows() const { return rows; }
    int getCols() const { return cols; }
};

class SGDFactorizer {
private:
    Matrix U; // User latent factors
    Matrix V; // Item latent factors
    SGDConfig config;
    bool model_trained;

public:
    SGDFactorizer(int num_users, int num_items, const SGDConfig& cfg);
    
    void initializeMatrices(int num_users, int num_items, int num_factors);
    double trainSerial(const std::vector<Rating>& ratings);
    double trainOpenMP(const std::vector<Rating>& ratings);
    double trainMPI(const std::vector<Rating>& local_ratings, class MPICoordinator& coordinator, int total_users, int total_items);
    double trainHybrid(const std::vector<Rating>& local_ratings, class MPICoordinator& coordinator, int total_users, int total_items);
    void predictRatings(const std::vector<std::pair<int,int>>& user_item_pairs);
    std::vector<std::pair<int, double>> generateTopNRecommendations(int user_id, int N, const std::vector<int>& rated_items = {}) const;
    std::vector<std::pair<int, double>> rankItems(int user_id, const std::vector<int>& item_ids) const;
    double calculateRMSE(const std::vector<Rating>& ratings) const;
    double calculateLocalRMSE(const std::vector<Rating>& local_ratings, int local_user_start) const;
    double predictRating(int user_id, int item_id) const;
    
    // Configuration methods
    void setNumThreads(int num_threads);
    int getNumThreads() const { return config.num_threads; }
    
    // Performance measurement and validation
    PerformanceMetrics benchmarkSerialVsOpenMP(const std::vector<Rating>& ratings);
    void printPerformanceReport(const PerformanceMetrics& metrics) const;
    bool validateResults(double serial_rmse, double openmp_rmse, double tolerance = 0.01) const;
    
    const Matrix& getUserMatrix() const { return U; }
    const Matrix& getItemMatrix() const { return V; }
    
    // Model persistence
    void saveModel(const std::string& model_path) const;
    void loadModel(const std::string& model_path);
    bool isModelTrained() const { return model_trained; }
    

};

#endif // SGD_FACTORIZER_H