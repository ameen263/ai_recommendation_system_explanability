import React, { useState, useEffect, useRef, useCallback } from 'react';
import MovieCard from './MovieCard';
import debounce from 'lodash.debounce';

const PAGE_SIZE = 10;

const MovieList = ({ movies, onFairnessChange, fetchRealTimeUpdates }) => {
  // Pagination, search, and sorting states.
  const [currentPage, setCurrentPage] = useState(1);
  const [displayedMovies, setDisplayedMovies] = useState([]);
  const [filterQuery, setFilterQuery] = useState('');
  const [sortBy, setSortBy] = useState('title');
  const [notification, setNotification] = useState(null);

  // Fairness control state: 0 means "More Personalized", 100 means "More Fair"
  const [fairnessPreference, setFairnessPreference] = useState(50);

  // Real-time updates state
  const [realTimeUpdates, setRealTimeUpdates] = useState([]);
  const [updateError, setUpdateError] = useState('');

  // Continue Watching state
  const [continueList, setContinueList] = useState([]);

  // Ref for detecting changes in movies prop
  const prevMoviesRef = useRef(null);

  // Load Continue Watching list from localStorage on mount
  useEffect(() => {
    const storedList = JSON.parse(localStorage.getItem('continueWatching')) || [];
    setContinueList(storedList);
  }, []);

  // Poll for real-time updates every 10 seconds (if fetchRealTimeUpdates is provided)
  useEffect(() => {
    let intervalId;
    if (fetchRealTimeUpdates) {
      const fetchUpdates = async () => {
        try {
          const updates = await fetchRealTimeUpdates();
          setRealTimeUpdates(updates);
          if (updates && updates.length > 0) {
            setNotification('New real-time updates available!');
            setTimeout(() => setNotification(null), 5000);
          }
        } catch (err) {
          setUpdateError('Failed to fetch real-time updates.');
        }
      };
      fetchUpdates();
      intervalId = setInterval(fetchUpdates, 10000);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [fetchRealTimeUpdates]);

  // Show notification when movies prop changes
  useEffect(() => {
    if (
      prevMoviesRef.current &&
      movies &&
      movies.length !== prevMoviesRef.current.length
    ) {
      setNotification('New recommendations available!');
      const timer = setTimeout(() => setNotification(null), 5000);
      return () => clearTimeout(timer);
    }
    prevMoviesRef.current = movies;
  }, [movies]);

  // Debounce search filtering to improve performance
  const debouncedFilter = useCallback(
    debounce((moviesData, query, sortOption, page) => {
      let filtered = moviesData;
      if (query) {
        filtered = filtered.filter((movie) =>
          movie.title.toLowerCase().includes(query.toLowerCase())
        );
      }
      // Sorting logic
      if (sortOption === 'title') {
        filtered = [...filtered].sort((a, b) => a.title.localeCompare(b.title));
      } else if (sortOption === 'rating') {
        filtered = [...filtered].sort((a, b) => b.rating - a.rating);
      } else if (sortOption === 'year') {
        filtered = [...filtered].sort((a, b) => b.year - a.year);
      }
      // Pagination
      const startIndex = (page - 1) * PAGE_SIZE;
      const endIndex = startIndex + PAGE_SIZE;
      setDisplayedMovies(filtered.slice(startIndex, endIndex));
    }, 300),
    []
  );

  // Update displayed movies when movies, filter, sort, or page changes
  useEffect(() => {
    if (!movies) return;
    debouncedFilter(movies, filterQuery, sortBy, currentPage);
  }, [movies, filterQuery, sortBy, currentPage, debouncedFilter]);

  // Fairness control handler
  const handleFairnessChange = (e) => {
    const value = parseInt(e.target.value, 10);
    setFairnessPreference(value);
    if (onFairnessChange) {
      onFairnessChange({ fairnessPreference: value });
    }
  };

  // Display skeleton loader if movies haven't loaded yet
  if (!movies) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4">
        {Array.from({ length: PAGE_SIZE }).map((_, index) => (
          <div key={index} className="animate-pulse flex flex-col space-y-2">
            <div className="bg-gray-300 h-48 w-full rounded"></div>
            <div className="bg-gray-300 h-4 w-3/4 rounded"></div>
          </div>
        ))}
      </div>
    );
  }

  // Display empty state if no movies are available
  if (movies.length === 0) {
    return (
      <div className="p-4 text-center">
        <p>No movies available.</p>
      </div>
    );
  }

  // Calculate total number of movies after filtering and total pages
  const totalFilteredMovies = movies.filter((movie) =>
    movie.title.toLowerCase().includes(filterQuery.toLowerCase())
  ).length;
  const totalPages = Math.ceil(totalFilteredMovies / PAGE_SIZE);

  return (
    <div className="p-4">
      {notification && (
        <div className="bg-green-500 text-white p-3 rounded mb-4">
          {notification}
        </div>
      )}
      {updateError && <p className="text-red-500 mb-4">{updateError}</p>}

      {/* Continue Watching Section */}
      {continueList.length > 0 && (
        <div className="mb-6">
          <h2 className="text-xl font-bold mb-4">Continue Watching</h2>
          <div className="flex space-x-4 overflow-x-auto">
            {continueList.map((movie) => (
              <div key={movie.id} className="min-w-[150px] cursor-pointer">
                <img
                  src={movie.poster}
                  alt={movie.title}
                  className="w-full rounded"
                />
                <p className="text-sm mt-2 text-center">{movie.title}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Fairness Control */}
      <div className="mb-4">
        <label htmlFor="fairnessSlider" className="block text-sm font-medium mb-1">
          Fairness vs. Personalization
        </label>
        <input
          id="fairnessSlider"
          type="range"
          min="0"
          max="100"
          value={fairnessPreference}
          onChange={handleFairnessChange}
          className="w-full"
        />
        <div className="flex justify-between text-xs mt-1">
          <span>More Personalized</span>
          <span>More Fair</span>
        </div>
        <p className="mt-1 text-sm">
          Current Preference: <strong>{fairnessPreference}</strong>
        </p>
      </div>

      {/* Sorting & Filtering Controls */}
      <div className="flex flex-col md:flex-row justify-between mb-4">
        <input
          type="text"
          placeholder="Search movies..."
          value={filterQuery}
          onChange={(e) => {
            setFilterQuery(e.target.value);
            setCurrentPage(1);
          }}
          className="p-2 border rounded mb-2 md:mb-0"
        />
        <select
          value={sortBy}
          onChange={(e) => {
            setSortBy(e.target.value);
            setCurrentPage(1);
          }}
          className="p-2 border rounded"
        >
          <option value="title">Sort by Title</option>
          <option value="rating">Sort by Rating</option>
          <option value="year">Sort by Year</option>
        </select>
      </div>

      {/* Movie Cards */}
      <div className="space-y-4">
        {displayedMovies.map((movie) => (
          <MovieCard key={movie.id} movie={movie} />
        ))}
      </div>

      {/* Pagination Controls */}
      {totalFilteredMovies > PAGE_SIZE && (
        <div className="flex justify-center mt-4 space-x-4">
          <button
            onClick={() => setCurrentPage(currentPage - 1)}
            disabled={currentPage === 1}
            className="bg-blue-600 text-white py-2 px-4 rounded disabled:opacity-50"
          >
            Previous
          </button>
          <span className="py-2 px-4">
            Page {currentPage} of {totalPages}
          </span>
          <button
            onClick={() => setCurrentPage(currentPage + 1)}
            disabled={currentPage === totalPages}
            className="bg-blue-600 text-white py-2 px-4 rounded disabled:opacity-50"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
};

export default MovieList;
