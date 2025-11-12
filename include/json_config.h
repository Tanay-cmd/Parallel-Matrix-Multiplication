#ifndef JSON_CONFIG_H
#define JSON_CONFIG_H

#include <string>
#include <map>
#include <vector>
#include <stdexcept>

// Simple JSON value types
enum class JsonType {
    STRING,
    NUMBER,
    BOOLEAN,
    OBJECT,
    ARRAY,
    NULL_VALUE
};

// Simple JSON value class
class JsonValue {
private:
    JsonType type_;
    std::string string_value_;
    double number_value_;
    bool boolean_value_;
    std::map<std::string, JsonValue> object_value_;
    std::vector<JsonValue> array_value_;

public:
    JsonValue() : type_(JsonType::NULL_VALUE) {}
    JsonValue(const std::string& value) : type_(JsonType::STRING), string_value_(value) {}
    JsonValue(double value) : type_(JsonType::NUMBER), number_value_(value) {}
    JsonValue(bool value) : type_(JsonType::BOOLEAN), boolean_value_(value) {}
    
    JsonType getType() const { return type_; }
    
    // Getters with type checking
    std::string asString() const;
    double asNumber() const;
    int asInt() const;
    bool asBool() const;
    
    // Object access
    bool hasKey(const std::string& key) const;
    const JsonValue& operator[](const std::string& key) const;
    JsonValue& operator[](const std::string& key);
    
    // Array access
    size_t size() const;
    const JsonValue& operator[](size_t index) const;
    
    // Set values
    void setObject();
    void setArray();
};

// Simple JSON parser
class JsonParser {
private:
    std::string json_text_;
    size_t pos_;
    
    void skipWhitespace();
    char peek();
    char next();
    std::string parseString();
    double parseNumber();
    bool parseBoolean();
    JsonValue parseValue();
    JsonValue parseObject();
    JsonValue parseArray();
    
public:
    JsonValue parse(const std::string& json_text);
    JsonValue parseFile(const std::string& filename);
};

// Configuration structures
struct AlgorithmConfig {
    double learning_rate = 0.01;
    double regularization = 0.01;
    int num_factors = 50;
    int max_epochs = 100;
    double convergence_threshold = 0.001;
};

struct ParallelismConfig {
    int num_threads = 4;
    bool use_mpi = false;
};

struct DataConfig {
    std::string input_file = "data/ratings.dat";
    std::string movies_file = "data/movies.dat";
    std::string users_file = "data/users.dat";
    double train_ratio = 0.8;
    bool validation_split = true;
};

struct OutputConfig {
    bool save_factors = false;
    bool performance_report = true;
    int top_n_recommendations = 10;
};

// Main configuration class
class Configuration {
private:
    AlgorithmConfig algorithm_;
    ParallelismConfig parallelism_;
    DataConfig data_;
    OutputConfig output_;
    
    void loadDefaults();
    void validateConfig();
    void loadFromJsonValue(const JsonValue& root);

public:
    Configuration();
    
    // Load configuration from JSON file
    void loadFromFile(const std::string& filename);
    
    // Load configuration from JSON string
    void loadFromJson(const std::string& json_text);
    
    // Getters
    const AlgorithmConfig& getAlgorithm() const { return algorithm_; }
    const ParallelismConfig& getParallelism() const { return parallelism_; }
    const DataConfig& getData() const { return data_; }
    const OutputConfig& getOutput() const { return output_; }
    
    // Setters for command-line overrides
    void setLearningRate(double rate) { algorithm_.learning_rate = rate; }
    void setRegularization(double reg) { algorithm_.regularization = reg; }
    void setNumFactors(int factors) { algorithm_.num_factors = factors; }
    void setMaxEpochs(int epochs) { algorithm_.max_epochs = epochs; }
    void setConvergenceThreshold(double threshold) { algorithm_.convergence_threshold = threshold; }
    void setNumThreads(int threads) { parallelism_.num_threads = threads; }
    void setUseMPI(bool use_mpi) { parallelism_.use_mpi = use_mpi; }
    void setInputFile(const std::string& file) { data_.input_file = file; }
    void setTrainRatio(double ratio) { data_.train_ratio = ratio; }
    void setValidationSplit(bool split) { data_.validation_split = split; }
    void setTopNRecommendations(int n) { output_.top_n_recommendations = n; }
    
    // Print configuration
    void printConfig() const;
};

#endif // JSON_CONFIG_H