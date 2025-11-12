#ifndef RATING_MATRIX_H
#define RATING_MATRIX_H

#include <vector>
#include <string>

struct Rating {
    int user_id;
    int item_id;
    double rating;
    long timestamp;
};

class RatingMatrix {
private:
    std::vector<Rating> ratings;
    int num_users;
    int num_items;
    double sparsity;

public:
    RatingMatrix();
    ~RatingMatrix();
    
    void loadFromDat(const std::string& filename);
    void createSparseRepresentation();
    std::vector<Rating> getUserBlock(int start_user, int end_user);
    void printStatistics();
    
    int getNumUsers() const { return num_users; }
    int getNumItems() const { return num_items; }
    double getSparsity() const { return sparsity; }
    const std::vector<Rating>& getRatings() const { return ratings; }
};

#endif // RATING_MATRIX_H