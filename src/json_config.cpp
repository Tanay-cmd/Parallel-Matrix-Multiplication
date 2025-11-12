#include "json_config.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cctype>
#include <stdexcept>

// JsonValue implementation
std::string JsonValue::asString() const {
    if (type_ != JsonType::STRING) {
        throw std::runtime_error("JsonValue is not a string");
    }
    return string_value_;
}

double JsonValue::asNumber() const {
    if (type_ != JsonType::NUMBER) {
        throw std::runtime_error("JsonValue is not a number");
    }
    return number_value_;
}

int JsonValue::asInt() const {
    if (type_ != JsonType::NUMBER) {
        throw std::runtime_error("JsonValue is not a number");
    }
    return static_cast<int>(number_value_);
}

bool JsonValue::asBool() const {
    if (type_ != JsonType::BOOLEAN) {
        throw std::runtime_error("JsonValue is not a boolean");
    }
    return boolean_value_;
}

bool JsonValue::hasKey(const std::string& key) const {
    if (type_ != JsonType::OBJECT) {
        return false;
    }
    return object_value_.find(key) != object_value_.end();
}

const JsonValue& JsonValue::operator[](const std::string& key) const {
    if (type_ != JsonType::OBJECT) {
        throw std::runtime_error("JsonValue is not an object");
    }
    auto it = object_value_.find(key);
    if (it == object_value_.end()) {
        throw std::runtime_error(std::string("Key not found: ") + key);
    }
    return it->second;
}

JsonValue& JsonValue::operator[](const std::string& key) {
    if (type_ != JsonType::OBJECT) {
        throw std::runtime_error("JsonValue is not an object");
    }
    return object_value_[key];
}

size_t JsonValue::size() const {
    if (type_ == JsonType::ARRAY) {
        return array_value_.size();
    } else if (type_ == JsonType::OBJECT) {
        return object_value_.size();
    }
    return 0;
}

const JsonValue& JsonValue::operator[](size_t index) const {
    if (type_ != JsonType::ARRAY) {
        throw std::runtime_error("JsonValue is not an array");
    }
    if (index >= array_value_.size()) {
        throw std::runtime_error("Array index out of bounds");
    }
    return array_value_[index];
}

void JsonValue::setObject() {
    type_ = JsonType::OBJECT;
    object_value_.clear();
}

void JsonValue::setArray() {
    type_ = JsonType::ARRAY;
    array_value_.clear();
}

// JsonParser implementation
void JsonParser::skipWhitespace() {
    while (pos_ < json_text_.length() && std::isspace(json_text_[pos_])) {
        pos_++;
    }
}

char JsonParser::peek() {
    skipWhitespace();
    if (pos_ >= json_text_.length()) {
        return '\0';
    }
    return json_text_[pos_];
}

char JsonParser::next() {
    skipWhitespace();
    if (pos_ >= json_text_.length()) {
        return '\0';
    }
    return json_text_[pos_++];
}

std::string JsonParser::parseString() {
    if (next() != '"') {
        throw std::runtime_error("Expected '\"' at start of string");
    }
    
    std::string result;
    while (pos_ < json_text_.length() && json_text_[pos_] != '"') {
        if (json_text_[pos_] == '\\' && pos_ + 1 < json_text_.length()) {
            pos_++; // Skip backslash
            char escaped = json_text_[pos_];
            switch (escaped) {
                case 'n': result += '\n'; break;
                case 't': result += '\t'; break;
                case 'r': result += '\r'; break;
                case '\\': result += '\\'; break;
                case '"': result += '"'; break;
                default: result += escaped; break;
            }
        } else {
            result += json_text_[pos_];
        }
        pos_++;
    }
    
    if (pos_ >= json_text_.length() || json_text_[pos_] != '"') {
        throw std::runtime_error("Unterminated string");
    }
    pos_++; // Skip closing quote
    
    return result;
}

double JsonParser::parseNumber() {
    std::string number_str;
    
    // Handle negative numbers
    if (peek() == '-') {
        number_str += next();
    }
    
    // Parse digits
    while (pos_ < json_text_.length() && (std::isdigit(json_text_[pos_]) || json_text_[pos_] == '.')) {
        number_str += json_text_[pos_++];
    }
    
    if (number_str.empty() || number_str == "-") {
        throw std::runtime_error("Invalid number format");
    }
    
    return std::stod(number_str);
}

bool JsonParser::parseBoolean() {
    if (json_text_.substr(pos_, 4) == "true") {
        pos_ += 4;
        return true;
    } else if (json_text_.substr(pos_, 5) == "false") {
        pos_ += 5;
        return false;
    } else {
        throw std::runtime_error("Invalid boolean value");
    }
}

JsonValue JsonParser::parseValue() {
    char c = peek();
    
    if (c == '"') {
        return JsonValue(parseString());
    } else if (c == '{') {
        return parseObject();
    } else if (c == '[') {
        return parseArray();
    } else if (c == 't' || c == 'f') {
        return JsonValue(parseBoolean());
    } else if (c == 'n') {
        if (json_text_.substr(pos_, 4) == "null") {
            pos_ += 4;
            return JsonValue(); // null value
        } else {
            throw std::runtime_error("Invalid null value");
        }
    } else if (std::isdigit(c) || c == '-') {
        return JsonValue(parseNumber());
    } else {
        throw std::runtime_error(std::string("Unexpected character: ") + std::string(1, c));
    }
}

JsonValue JsonParser::parseObject() {
    if (next() != '{') {
        throw std::runtime_error("Expected '{' at start of object");
    }
    
    JsonValue obj;
    obj.setObject();
    
    if (peek() == '}') {
        next(); // Skip closing brace
        return obj;
    }
    
    while (true) {
        // Parse key
        if (peek() != '"') {
            throw std::runtime_error("Expected string key in object");
        }
        std::string key = parseString();
        
        // Parse colon
        if (next() != ':') {
            throw std::runtime_error("Expected ':' after object key");
        }
        
        // Parse value
        obj[key] = parseValue();
        
        char c = peek();
        if (c == '}') {
            next(); // Skip closing brace
            break;
        } else if (c == ',') {
            next(); // Skip comma
        } else {
            throw std::runtime_error("Expected ',' or '}' in object");
        }
    }
    
    return obj;
}

JsonValue JsonParser::parseArray() {
    if (next() != '[') {
        throw std::runtime_error("Expected '[' at start of array");
    }
    
    JsonValue arr;
    arr.setArray();
    
    if (peek() == ']') {
        next(); // Skip closing bracket
        return arr;
    }
    
    while (true) {
        // This is a simplified implementation - we'd need to modify JsonValue to support array operations
        // For now, we'll just skip array parsing since our config doesn't use arrays
        throw std::runtime_error("Array parsing not implemented in this simplified parser");
    }
}

JsonValue JsonParser::parse(const std::string& json_text) {
    json_text_ = json_text;
    pos_ = 0;
    return parseValue();
}

JsonValue JsonParser::parseFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error(std::string("Could not open file: ") + filename);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return parse(buffer.str());
}

// Configuration implementation
Configuration::Configuration() {
    loadDefaults();
}

void Configuration::loadDefaults() {
    // Algorithm defaults
    algorithm_.learning_rate = 0.01;
    algorithm_.regularization = 0.01;
    algorithm_.num_factors = 50;
    algorithm_.max_epochs = 100;
    algorithm_.convergence_threshold = 0.001;
    
    // Parallelism defaults
    parallelism_.num_threads = 4;
    parallelism_.use_mpi = false;
    
    // Data defaults
    data_.input_file = "data/ratings.dat";
    data_.movies_file = "data/movies.dat";
    data_.users_file = "data/users.dat";
    data_.train_ratio = 0.8;
    data_.validation_split = true;
    
    // Output defaults
    output_.save_factors = false;
    output_.performance_report = true;
    output_.top_n_recommendations = 10;
}

void Configuration::loadFromFile(const std::string& filename) {
    try {
        JsonParser parser;
        JsonValue root = parser.parseFile(filename);
        loadFromJsonValue(root);
        validateConfig();
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to load configuration from ") + filename + ": " + e.what());
    }
}

void Configuration::loadFromJson(const std::string& json_text) {
    try {
        JsonParser parser;
        JsonValue root = parser.parse(json_text);
        loadFromJsonValue(root);
        validateConfig();
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to parse JSON configuration: ") + e.what());
    }
}

void Configuration::loadFromJsonValue(const JsonValue& root) {
    // Load algorithm configuration
    if (root.hasKey("algorithm")) {
        const JsonValue& algo = root["algorithm"];
        if (algo.hasKey("learning_rate")) {
            algorithm_.learning_rate = algo["learning_rate"].asNumber();
        }
        if (algo.hasKey("regularization")) {
            algorithm_.regularization = algo["regularization"].asNumber();
        }
        if (algo.hasKey("num_factors")) {
            algorithm_.num_factors = algo["num_factors"].asInt();
        }
        if (algo.hasKey("max_epochs")) {
            algorithm_.max_epochs = algo["max_epochs"].asInt();
        }
        if (algo.hasKey("convergence_threshold")) {
            algorithm_.convergence_threshold = algo["convergence_threshold"].asNumber();
        }
    }
    
    // Load parallelism configuration
    if (root.hasKey("parallelism")) {
        const JsonValue& para = root["parallelism"];
        if (para.hasKey("num_threads")) {
            parallelism_.num_threads = para["num_threads"].asInt();
        }
        if (para.hasKey("use_mpi")) {
            parallelism_.use_mpi = para["use_mpi"].asBool();
        }
    }
    
    // Load data configuration
    if (root.hasKey("data")) {
        const JsonValue& data = root["data"];
        if (data.hasKey("input_file")) {
            data_.input_file = data["input_file"].asString();
        }
        if (data.hasKey("ratings_file")) {
            data_.input_file = data["ratings_file"].asString();
        }
        if (data.hasKey("movies_file")) {
            data_.movies_file = data["movies_file"].asString();
        }
        if (data.hasKey("users_file")) {
            data_.users_file = data["users_file"].asString();
        }
        if (data.hasKey("train_ratio")) {
            data_.train_ratio = data["train_ratio"].asNumber();
        }
        if (data.hasKey("validation_split")) {
            data_.validation_split = data["validation_split"].asBool();
        }
    }
    
    // Load output configuration
    if (root.hasKey("output")) {
        const JsonValue& output = root["output"];
        if (output.hasKey("save_factors")) {
            output_.save_factors = output["save_factors"].asBool();
        }
        if (output.hasKey("performance_report")) {
            output_.performance_report = output["performance_report"].asBool();
        }
        if (output.hasKey("top_n_recommendations")) {
            output_.top_n_recommendations = output["top_n_recommendations"].asInt();
        }
    }
}

void Configuration::validateConfig() {
    // Validate algorithm parameters
    if (algorithm_.learning_rate <= 0.0 || algorithm_.learning_rate > 1.0) {
        throw std::runtime_error("Learning rate must be between 0 and 1");
    }
    if (algorithm_.regularization < 0.0) {
        throw std::runtime_error("Regularization must be non-negative");
    }
    if (algorithm_.num_factors <= 0) {
        throw std::runtime_error("Number of factors must be positive");
    }
    if (algorithm_.max_epochs <= 0) {
        throw std::runtime_error("Maximum epochs must be positive");
    }
    if (algorithm_.convergence_threshold <= 0.0) {
        throw std::runtime_error("Convergence threshold must be positive");
    }
    
    // Validate parallelism parameters
    if (parallelism_.num_threads <= 0) {
        throw std::runtime_error("Number of threads must be positive");
    }
    
    // Validate data parameters
    if (data_.train_ratio <= 0.0 || data_.train_ratio >= 1.0) {
        throw std::runtime_error("Train ratio must be between 0 and 1");
    }
    
    // Validate output parameters
    if (output_.top_n_recommendations <= 0) {
        throw std::runtime_error("Top N recommendations must be positive");
    }
}

void Configuration::printConfig() const {
    std::cout << "=== Configuration ===" << std::endl;
    std::cout << "Algorithm:" << std::endl;
    std::cout << "  Learning Rate: " << algorithm_.learning_rate << std::endl;
    std::cout << "  Regularization: " << algorithm_.regularization << std::endl;
    std::cout << "  Num Factors: " << algorithm_.num_factors << std::endl;
    std::cout << "  Max Epochs: " << algorithm_.max_epochs << std::endl;
    std::cout << "  Convergence Threshold: " << algorithm_.convergence_threshold << std::endl;
    
    std::cout << "Parallelism:" << std::endl;
    std::cout << "  Num Threads: " << parallelism_.num_threads << std::endl;
    std::cout << "  Use MPI: " << (parallelism_.use_mpi ? "true" : "false") << std::endl;
    
    std::cout << "Data:" << std::endl;
    std::cout << "  Input File: " << data_.input_file << std::endl;
    std::cout << "  Movies File: " << data_.movies_file << std::endl;
    std::cout << "  Users File: " << data_.users_file << std::endl;
    std::cout << "  Train Ratio: " << data_.train_ratio << std::endl;
    std::cout << "  Validation Split: " << (data_.validation_split ? "true" : "false") << std::endl;
    
    std::cout << "Output:" << std::endl;
    std::cout << "  Save Factors: " << (output_.save_factors ? "true" : "false") << std::endl;
    std::cout << "  Performance Report: " << (output_.performance_report ? "true" : "false") << std::endl;
    std::cout << "  Top N Recommendations: " << output_.top_n_recommendations << std::endl;
    std::cout << "=====================" << std::endl;
}