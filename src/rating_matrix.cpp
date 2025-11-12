#include "rating_matrix.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <set>

RatingMatrix::RatingMatrix() : num_users(0), num_items(0), sparsity(0.0) {
}

RatingMatrix::~RatingMatrix() {
}

void RatingMatrix::loadFromDat(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    ratings.clear();
    std::string line;
    std::set<int> unique_users, unique_items;
    int total_lines = 0;
    int valid_lines = 0;
    int invalid_ratings = 0;
    int parse_errors = 0;
    
    std::cout << "Loading data from: " << filename << std::endl;
    
    while (std::getline(file, line)) {
        total_lines++;
        if (line.empty()) continue;
        
        // Parse MovieLens format: user_id::item_id::rating::timestamp
        // Split by "::" delimiter
        std::vector<std::string> tokens;
        size_t pos = 0;
        std::string delimiter = "::";
        std::string token;
        std::string temp_line = line;
        
        while ((pos = temp_line.find(delimiter)) != std::string::npos) {
            token = temp_line.substr(0, pos);
            tokens.push_back(token);
            temp_line.erase(0, pos + delimiter.length());
        }
        tokens.push_back(temp_line); // Add the last token
        
        if (tokens.size() >= 3) {
            try {
                Rating r;
                r.user_id = std::stoi(tokens[0]);
                r.item_id = std::stoi(tokens[1]);
                r.rating = std::stod(tokens[2]);
                r.timestamp = (tokens.size() > 3) ? std::stol(tokens[3]) : 0;
                
                // Data validation: skip invalid ratings (< 1 or > 5)
                if (r.rating >= 1.0 && r.rating <= 5.0) {
                    ratings.push_back(r);
                    unique_users.insert(r.user_id);
                    unique_items.insert(r.item_id);
                    valid_lines++;
                } else {
                    invalid_ratings++;
                }
            } catch (const std::exception& e) {
                parse_errors++;
                continue;
            }
        } else {
            parse_errors++;
        }
    }
    
    file.close();
    
    // Update matrix dimensions
    num_users = unique_users.size();
    num_items = unique_items.size();
    
    // Calculate sparsity
    long total_possible_ratings = static_cast<long>(num_users) * num_items;
    sparsity = (total_possible_ratings > 0) ? 
               (1.0 - static_cast<double>(ratings.size()) / total_possible_ratings) : 0.0;
    
    // Report loading statistics
    std::cout << "Data loading complete:" << std::endl;
    std::cout << "  Total lines processed: " << total_lines << std::endl;
    std::cout << "  Valid ratings loaded: " << valid_lines << std::endl;
    std::cout << "  Invalid ratings (< 1 or > 5): " << invalid_ratings << std::endl;
    std::cout << "  Parse errors: " << parse_errors << std::endl;
    std::cout << "  Success rate: " << (total_lines > 0 ? (valid_lines * 100.0 / total_lines) : 0) << "%" << std::endl;
}

void RatingMatrix::createSparseRepresentation() {
    // The ratings vector is already a sparse representation
    // Sort by user_id for efficient block partitioning
    std::sort(ratings.begin(), ratings.end(), 
              [](const Rating& a, const Rating& b) {
                  return a.user_id < b.user_id;
              });
}

std::vector<Rating> RatingMatrix::getUserBlock(int start_user, int end_user) {
    std::vector<Rating> block;
    
    for (const auto& rating : ratings) {
        if (rating.user_id >= start_user && rating.user_id <= end_user) {
            block.push_back(rating);
        }
    }
    
    return block;
}

void RatingMatrix::printStatistics() {
    std::cout << "=== Rating Matrix Statistics ===" << std::endl;
    std::cout << "Number of users: " << num_users << std::endl;
    std::cout << "Number of items: " << num_items << std::endl;
    std::cout << "Number of ratings: " << ratings.size() << std::endl;
    std::cout << "Matrix sparsity: " << (sparsity * 100.0) << "%" << std::endl;
    
    if (!ratings.empty()) {
        // Calculate rating distribution
        std::vector<int> rating_counts(6, 0); // Index 0 unused, 1-5 for ratings
        double sum = 0.0;
        
        for (const auto& rating : ratings) {
            int r = static_cast<int>(rating.rating);
            if (r >= 1 && r <= 5) {
                rating_counts[r]++;
            }
            sum += rating.rating;
        }
        
        double avg_rating = sum / ratings.size();
        std::cout << "Average rating: " << avg_rating << std::endl;
        std::cout << "Rating distribution:" << std::endl;
        for (int i = 1; i <= 5; i++) {
            double percentage = (static_cast<double>(rating_counts[i]) / ratings.size()) * 100.0;
            std::cout << "  " << i << " stars: " << rating_counts[i] 
                      << " (" << percentage << "%)" << std::endl;
        }
    }
}