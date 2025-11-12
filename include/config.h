#ifndef CONFIG_H
#define CONFIG_H

// Project configuration constants
#define PROJECT_NAME "ParallelMatrixRecommendation"
#define PROJECT_VERSION "1.0.0"

// Build configuration
#ifdef USE_OPENMP
#define OPENMP_ENABLED true
#else
#define OPENMP_ENABLED false
#endif

#ifdef USE_MPI
#define MPI_ENABLED true
#else
#define MPI_ENABLED false
#endif

// Default algorithm parameters
#define DEFAULT_NUM_FACTORS 50
#define DEFAULT_LEARNING_RATE 0.01
#define DEFAULT_REGULARIZATION 0.01
#define DEFAULT_MAX_EPOCHS 100
#define DEFAULT_CONVERGENCE_THRESHOLD 0.001
#define DEFAULT_NUM_THREADS 4

// Data file paths
#define DEFAULT_RATINGS_FILE "data/ratings.dat"
#define DEFAULT_MOVIES_FILE "data/movies.dat"
#define DEFAULT_USERS_FILE "data/users.dat"
#define DEFAULT_MODEL_FILE "trained_model.bin"

#endif // CONFIG_H