#include "movie_info.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

void MovieDatabase::loadFromDat(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open movies file: " + filename);
    }
    
    std::string line;
    int count = 0;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        // Parse MovieLens format: id::title::genres
        std::vector<std::string> parts;
        std::stringstream ss(line);
        std::string part;
        
        while (std::getline(ss, part, ':')) {
            if (!part.empty() && part != ":") {
                parts.push_back(part);
            }
        }
        
        if (parts.size() >= 3) {
            MovieInfo movie;
            movie.id = std::stoi(parts[0]);
            movie.title = parts[1];
            
            // Parse genres (separated by |)
            std::string genres_str = parts[2];
            std::stringstream genre_ss(genres_str);
            std::string genre;
            
            while (std::getline(genre_ss, genre, '|')) {
                if (!genre.empty()) {
                    movie.genres.push_back(genre);
                }
            }
            
            movies[movie.id] = movie;
            count++;
        }
    }
    
    file.close();
    std::cout << "Loaded " << count << " movies from " << filename << std::endl;
}

MovieInfo MovieDatabase::getMovieInfo(int movie_id) const {
    auto it = movies.find(movie_id);
    if (it != movies.end()) {
        return it->second;
    }
    
    // Return empty movie info if not found
    MovieInfo empty;
    empty.id = movie_id;
    empty.title = "Unknown Movie";
    empty.genres = {"Unknown"};
    return empty;
}

bool MovieDatabase::hasMovie(int movie_id) const {
    return movies.find(movie_id) != movies.end();
}

std::string MovieDatabase::getMovieTitle(int movie_id) const {
    return getMovieInfo(movie_id).title;
}

std::vector<std::string> MovieDatabase::getMovieGenres(int movie_id) const {
    return getMovieInfo(movie_id).genres;
}

void MovieDatabase::printMovieInfo(int movie_id) const {
    MovieInfo movie = getMovieInfo(movie_id);
    std::cout << "Movie ID: " << movie.id << std::endl;
    std::cout << "Title: " << movie.title << std::endl;
    std::cout << "Genres: ";
    for (size_t i = 0; i < movie.genres.size(); i++) {
        std::cout << movie.genres[i];
        if (i < movie.genres.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
}