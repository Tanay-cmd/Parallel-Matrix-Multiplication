#ifndef CLI_PARSER_H
#define CLI_PARSER_H

#include "json_config.h"
#include <string>
#include <vector>

// Execution modes
enum class ExecutionMode {
    SERIAL,
    OPENMP,
    MPI,
    HYBRID
};

// Command line options structure
struct CLIOptions {
    ExecutionMode mode = ExecutionMode::SERIAL;
    std::string config_file = "";
    std::string data_file = "";
    int num_threads = 0;  // 0 means use config file value
    int num_factors = 0;  // 0 means use config file value
    int max_epochs = 0;   // 0 means use config file value
    double learning_rate = 0.0;  // 0.0 means use config file value
    double regularization = 0.0; // 0.0 means use config file value
    int top_n = 0;        // 0 means use config file value
    bool show_help = false;
    bool show_version = false;
    bool benchmark_mode = false;
    bool verbose = false;
    bool quiet = false;
};

// Command line parser class
class CLIParser {
private:
    std::vector<std::string> args_;
    CLIOptions options_;
    
    void parseArguments();
    ExecutionMode parseMode(const std::string& mode_str);
    void printUsage(const std::string& program_name);
    void printVersion();
    void printModeHelp();

public:
    CLIParser(int argc, char** argv);
    
    // Parse command line arguments and return options
    CLIOptions parse();
    
    // Apply CLI options to configuration
    void applyToConfiguration(Configuration& config) const;
    
    // Validate that the selected mode is supported
    bool validateMode() const;
    
    // Print help information
    void printHelp(const std::string& program_name);
};

// Utility functions
std::string executionModeToString(ExecutionMode mode);
bool isModeSupported(ExecutionMode mode);

#endif // CLI_PARSER_H