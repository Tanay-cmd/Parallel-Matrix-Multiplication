#include "sgd_factorizer.h"
#include "movie_info.h"
#include "rating_matrix.h"
#include <iostream>
#include <iomanip>
#include <set>
#include <algorithm>
#include <limits>

class RecommendationEngine {
private:
    SGDFactorizer factorizer;
    MovieDatabase movie_db;
    RatingMatrix rating_matrix;
    bool model_loaded;
    bool data_loaded;
    
public:
    RecommendationEngine() : factorizer(1, 1, SGDConfig()), model_loaded(false), data_loaded(false) {}
    
    void loadModel(const std::string& model_path) {
        factorizer.loadModel(model_path);
        model_loaded = true;
        std::cout << "[SUCCESS] Model loaded successfully!" << std::endl;
    }
    
    void loadMovieDatabase(const std::string& movies_path) {
        movie_db.loadFromDat(movies_path);
        std::cout << "[SUCCESS] Movie database loaded!" << std::endl;
    }
    
    void loadRatingData(const std::string& ratings_path) {
        rating_matrix.loadFromDat(ratings_path);
        data_loaded = true;
        std::cout << "[SUCCESS] Rating data loaded!" << std::endl;
    }
    
    std::vector<int> getUserRatedItems(int user_id) const {
        std::vector<int> rated_items;
        if (!data_loaded) return rated_items;
        
        for (const auto& rating : rating_matrix.getRatings()) {
            if (rating.user_id == user_id) {
                rated_items.push_back(rating.item_id);
            }
        }
        return rated_items;
    }
    
    void showUserHistory(int user_id, int max_items = 10) const {
        if (!data_loaded) {
            std::cout << "Rating data not loaded. Cannot show user history." << std::endl;
            return;
        }
        
        std::vector<Rating> user_ratings;
        for (const auto& rating : rating_matrix.getRatings()) {
            if (rating.user_id == user_id) {
                user_ratings.push_back(rating);
            }
        }
        
        if (user_ratings.empty()) {
            std::cout << "No rating history found for User " << user_id << std::endl;
            return;
        }
        
        // Sort by rating (highest first)
        std::sort(user_ratings.begin(), user_ratings.end(), 
                  [](const Rating& a, const Rating& b) { return a.rating > b.rating; });
        
        std::cout << "\n=== User " << user_id << " Rating History (Top " << max_items << ") ===" << std::endl;
        int count = 0;
        for (const auto& rating : user_ratings) {
            if (count >= max_items) break;
            
            std::string title = movie_db.hasMovie(rating.item_id) ? 
                               movie_db.getMovieTitle(rating.item_id) : "Unknown Movie";
            std::vector<std::string> genres = movie_db.hasMovie(rating.item_id) ? 
                                             movie_db.getMovieGenres(rating.item_id) : 
                                             std::vector<std::string>{"Unknown"};
            
            std::cout << (count + 1) << ". " << title << " (ID: " << rating.item_id << ")" << std::endl;
            std::cout << "   User Rating: " << std::fixed << std::setprecision(1) << rating.rating << "/5.0" << std::endl;
            std::cout << "   Genres: ";
            for (size_t i = 0; i < genres.size(); i++) {
                std::cout << genres[i];
                if (i < genres.size() - 1) std::cout << ", ";
            }
            std::cout << std::endl << std::endl;
            count++;
        }
    }
    
    void generateRecommendations(int user_id, int num_recommendations = 10) {
        if (!model_loaded) {
            std::cout << "Error: Model not loaded. Please load model first." << std::endl;
            return;
        }
        
        std::cout << "\n=== Generating Recommendations for User " << user_id << " ===" << std::endl;
        
        // Get user's rated items to exclude from recommendations
        std::vector<int> rated_items = getUserRatedItems(user_id);
        std::cout << "User has rated " << rated_items.size() << " movies." << std::endl;
        
        // Generate recommendations
        auto recommendations = factorizer.generateTopNRecommendations(user_id, num_recommendations, rated_items);
        
        if (recommendations.empty()) {
            std::cout << "No recommendations available for this user." << std::endl;
            return;
        }
        
        std::cout << "\n=== Top " << num_recommendations << " Movie Recommendations ===" << std::endl;
        for (size_t i = 0; i < recommendations.size(); i++) {
            int movie_id = recommendations[i].first;
            double predicted_rating = recommendations[i].second;
            
            std::string title = movie_db.hasMovie(movie_id) ? 
                               movie_db.getMovieTitle(movie_id) : "Unknown Movie";
            std::vector<std::string> genres = movie_db.hasMovie(movie_id) ? 
                                             movie_db.getMovieGenres(movie_id) : 
                                             std::vector<std::string>{"Unknown"};
            
            std::cout << (i + 1) << ". " << title << " (ID: " << movie_id << ")" << std::endl;
            std::cout << "   Predicted Rating: " << std::fixed << std::setprecision(2) << predicted_rating << "/5.0" << std::endl;
            
            // Add confidence indicator
            if (predicted_rating >= 4.0) {
                std::cout << "   Confidence: [***] Highly Recommended" << std::endl;
            } else if (predicted_rating >= 3.5) {
                std::cout << "   Confidence: [**-] Recommended" << std::endl;
            } else {
                std::cout << "   Confidence: [*--] Might Like" << std::endl;
            }
            
            std::cout << "   Genres: ";
            for (size_t j = 0; j < genres.size(); j++) {
                std::cout << genres[j];
                if (j < genres.size() - 1) std::cout << ", ";
            }
            std::cout << std::endl << std::endl;
        }
    }
    
    void predictRating(int user_id, int movie_id) {
        if (!model_loaded) {
            std::cout << "Error: Model not loaded. Please load model first." << std::endl;
            return;
        }
        
        double predicted = factorizer.predictRating(user_id, movie_id);
        std::string title = movie_db.hasMovie(movie_id) ? 
                           movie_db.getMovieTitle(movie_id) : "Unknown Movie";
        
        std::cout << "\n=== Rating Prediction ===" << std::endl;
        std::cout << "Movie: " << title << " (ID: " << movie_id << ")" << std::endl;
        std::cout << "Predicted rating for User " << user_id << ": " 
                  << std::fixed << std::setprecision(2) << predicted << "/5.0" << std::endl;
        
        if (predicted >= 4.0) {
            std::cout << "Recommendation: [***] Highly Recommended!" << std::endl;
        } else if (predicted >= 3.5) {
            std::cout << "Recommendation: [**-] Worth Watching" << std::endl;
        } else if (predicted >= 2.5) {
            std::cout << "Recommendation: [*--] Might Not Enjoy" << std::endl;
        } else {
            std::cout << "Recommendation: [---] Probably Skip" << std::endl;
        }
    }
    
    void showSystemInfo() const {
        std::cout << "\n=== Recommendation System Info ===" << std::endl;
        std::cout << "Model Status: " << (model_loaded ? "[LOADED]" : "[NOT LOADED]") << std::endl;
        std::cout << "Movie Database: " << (movie_db.getMovieCount() > 0 ? "[LOADED] (" + std::to_string(movie_db.getMovieCount()) + " movies)" : "[NOT LOADED]") << std::endl;
        std::cout << "Rating Data: " << (data_loaded ? "[LOADED] (" + std::to_string(rating_matrix.getRatings().size()) + " ratings)" : "[NOT LOADED]") << std::endl;
        
        if (data_loaded) {
            std::cout << "Users: " << rating_matrix.getNumUsers() << std::endl;
            std::cout << "Movies: " << rating_matrix.getNumItems() << std::endl;
        }
    }
};

void printMenu() {
    std::cout << "\n=== Movie Recommendation System ===" << std::endl;
    std::cout << "1. Get recommendations for user" << std::endl;
    std::cout << "2. Predict rating for specific movie" << std::endl;
    std::cout << "3. Show user rating history" << std::endl;
    std::cout << "4. System information" << std::endl;
    std::cout << "5. Exit" << std::endl;
    std::cout << "Choose option (1-5): ";
}

int main(int argc, char** argv) {
    try {
        std::string model_file = "trained_model.bin";
        std::string movies_file = "data/movies.dat";
        std::string ratings_file = "data/ratings.dat";
        
        // Parse command line arguments
        if (argc > 1) {
            model_file = argv[1];
        }
        
        std::cout << "=== Movie Recommendation System - Inference Mode ===" << std::endl;
        std::cout << "Model file: " << model_file << std::endl;
        std::cout << "Movies file: " << movies_file << std::endl;
        std::cout << "Ratings file: " << ratings_file << std::endl;
        
        // Initialize recommendation engine
        RecommendationEngine engine;
        
        // Load model
        std::cout << "\nLoading trained model..." << std::endl;
        engine.loadModel(model_file);
        
        // Load movie database
        std::cout << "Loading movie database..." << std::endl;
        try {
            engine.loadMovieDatabase(movies_file);
        } catch (const std::exception& e) {
            std::cout << "Warning: Could not load movie database: " << e.what() << std::endl;
        }
        
        // Load rating data for user history
        std::cout << "Loading rating data..." << std::endl;
        try {
            engine.loadRatingData(ratings_file);
        } catch (const std::exception& e) {
            std::cout << "Warning: Could not load rating data: " << e.what() << std::endl;
        }
        
        // Interactive menu
        int choice;
        while (true) {
            printMenu();
            std::cin >> choice;
            
            if (std::cin.fail()) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "Invalid input. Please enter a number." << std::endl;
                continue;
            }
            
            std::cin.ignore(); // Clear any remaining newline
            
            switch (choice) {
                case 1: {
                    int user_id, num_recs;
                    std::cout << "Enter User ID: ";
                    std::cin >> user_id;
                    std::cout << "Number of recommendations (default 10): ";
                    std::cin >> num_recs;
                    
                    if (num_recs <= 0) num_recs = 10;
                    
                    engine.showUserHistory(user_id, 5);
                    engine.generateRecommendations(user_id, num_recs);
                    break;
                }
                case 2: {
                    int user_id, movie_id;
                    std::cout << "Enter User ID: ";
                    std::cin >> user_id;
                    std::cout << "Enter Movie ID: ";
                    std::cin >> movie_id;
                    
                    engine.predictRating(user_id, movie_id);
                    break;
                }
                case 3: {
                    int user_id, num_items;
                    std::cout << "Enter User ID: ";
                    std::cin >> user_id;
                    std::cout << "Number of items to show (default 10): ";
                    std::cin >> num_items;
                    
                    if (num_items <= 0) num_items = 10;
                    
                    engine.showUserHistory(user_id, num_items);
                    break;
                }
                case 4: {
                    engine.showSystemInfo();
                    break;
                }
                case 5: {
                    std::cout << "Thank you for using the Movie Recommendation System!" << std::endl;
                    return 0;
                }
                default: {
                    std::cout << "Invalid choice. Please select 1-5." << std::endl;
                    break;
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}