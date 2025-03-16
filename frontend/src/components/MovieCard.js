import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import { Star, Info, Heart } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';

// Stub function for favoriting a movie
// Replace this with your actual API call to favorite/unfavorite a movie.
const favoriteMovie = async (movieId) => {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      // Simulate success 90% of the time
      if (Math.random() < 0.9) {
        resolve();
      } else {
        reject(new Error('Failed to update favorite status.'));
      }
    }, 500);
  });
};

const MovieCard = ({ movie }) => {
  const [showExplanation, setShowExplanation] = useState(false);
  const [isFavorited, setIsFavorited] = useState(false);
  const [favoriteLoading, setFavoriteLoading] = useState(false);
  const [favoriteError, setFavoriteError] = useState('');

  // On component mount, check localStorage to see if this movie is favorited.
  useEffect(() => {
    const favorites = JSON.parse(localStorage.getItem('favorites')) || [];
    setIsFavorited(favorites.includes(movie.id));
  }, [movie.id]);

  const handleFavoriteMovie = async (movieId) => {
    if (favoriteLoading) return;
    setFavoriteError('');
    setFavoriteLoading(true);
    try {
      await favoriteMovie(movieId);
      // Get the current favorites from localStorage.
      let favorites = JSON.parse(localStorage.getItem('favorites')) || [];
      if (favorites.includes(movieId)) {
        // Remove the movie from favorites.
        favorites = favorites.filter((id) => id !== movieId);
        setIsFavorited(false);
      } else {
        // Add the movie to favorites.
        favorites.push(movieId);
        setIsFavorited(true);
      }
      localStorage.setItem('favorites', JSON.stringify(favorites));
    } catch (error) {
      console.error('Error favoriting movie:', error);
      setFavoriteError('Failed to update favorite status. Please try again.');
    } finally {
      setFavoriteLoading(false);
    }
  };

  return (
    <Card className="mb-4">
      <CardContent className="p-6">
        <div className="flex justify-between items-start">
          <div>
            <h3 className="text-lg font-semibold mb-2">
              {movie.title} ({movie.year})
            </h3>
            <div className="flex items-center space-x-2 mb-2">
              <Star className="text-yellow-500" size={16} />
              <span>{movie.rating}/5.0</span>
            </div>
            <div className="flex flex-wrap gap-2 mb-2">
              {movie.genres.map((genre) => (
                <span
                  key={genre}
                  className="px-2 py-1 bg-gray-100 rounded-full text-sm"
                >
                  {genre}
                </span>
              ))}
            </div>
          </div>
          <div className="flex space-x-2">
            <button
              onClick={() => setShowExplanation((prev) => !prev)}
              className="p-2 hover:bg-gray-100 rounded"
              aria-label="Show explanation"
              aria-expanded={showExplanation}
            >
              <Info size={20} />
            </button>
            <button
              onClick={() => handleFavoriteMovie(movie.id)}
              className="p-2 hover:bg-gray-100 rounded"
              aria-label={isFavorited ? 'Remove from favorites' : 'Add to favorites'}
              disabled={favoriteLoading}
            >
              <Heart size={20} color={isFavorited ? 'red' : 'gray'} />
            </button>
          </div>
        </div>

        {favoriteError && (
          <Alert variant="destructive" className="mt-2">
            <AlertDescription>{favoriteError}</AlertDescription>
          </Alert>
        )}

        {showExplanation && (
          <Alert className="mt-4">
            <AlertDescription>
              {/* Improved explanation (example) */}
              This movie was recommended because you enjoyed other movies with similar genres
              like {movie.genres.slice(0, 2).join(' and ')}. It also has a similar rating pattern
              to your favorites.
              <br />
              Similarity score: {movie.similarity}
            </AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

MovieCard.propTypes = {
  movie: PropTypes.shape({
    id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
    title: PropTypes.string.isRequired,
    year: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
    rating: PropTypes.number.isRequired,
    genres: PropTypes.arrayOf(PropTypes.string).isRequired,
    similarity: PropTypes.number,
  }).isRequired,
};

export default MovieCard;
