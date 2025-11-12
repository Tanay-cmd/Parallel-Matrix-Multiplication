#ifndef MOVIE_INFO_H
#define MOVIE_INFO_H

#include <string>
#include <vector>
#include <map>
#include <unordered_map>

struct MovieInfo {
    int id;
    std::string title;
    std::vector<std::string> genres;
};

class MovieDatabase {
private:
    std::unordered_map<int, MovieInfo> movies;
    
public:
    void loadFromDat(const std::string& filename);
    MovieInfo getMovieInfo(int movie_id) const;
    bool hasMovie(int movie_id) const;
    std::string getMovieTitle(int movie_id) const;
    std::vector<std::string> getMovieGenres(int movie_id) const;
    void printMovieInfo(int movie_id) const;
    int getMovieCount() const { return movies.size(); }
};

#endif