import React, { useState, useEffect, useCallback } from 'react';
import { Search, X } from 'lucide-react';
import debounce from 'lodash.debounce';

const SearchBar = ({ onSearchSelect }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState('');

  // Function to fetch suggestions from the API
  const fetchSuggestions = async (term) => {
    try {
      setIsSearching(true);
      setError('');
      // Replace `/api/search` with your API endpoint.
      const response = await fetch(`/api/search?q=${encodeURIComponent(term)}`);
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const data = await response.json();
      // Assume the API returns an object with a `results` array
      setSuggestions(data.results || []);
    } catch (err) {
      console.error(err);
      setError('Search failed. Please try again.');
    } finally {
      setIsSearching(false);
    }
  };

  // Debounced version of fetchSuggestions (500ms delay)
  const debouncedFetch = useCallback(
    debounce((term) => {
      if (term.trim() !== '') {
        fetchSuggestions(term);
      } else {
        setSuggestions([]);
      }
    }, 500),
    []
  );

  // Trigger debounced API call when the searchTerm changes
  useEffect(() => {
    debouncedFetch(searchTerm);
    return () => {
      debouncedFetch.cancel();
    };
  }, [searchTerm, debouncedFetch]);

  // Handlers for input, clear, suggestion click and form submit
  const handleInputChange = (event) => {
    setSearchTerm(event.target.value);
  };

  const handleClear = () => {
    setSearchTerm('');
    setSuggestions([]);
  };

  const handleSuggestionClick = (suggestion) => {
    setSearchTerm(suggestion.title);
    setSuggestions([]);
    if (onSearchSelect) {
      onSearchSelect(suggestion);
    }
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    // Optionally, you can force a search here by canceling the debounce and calling fetchSuggestions immediately.
    debouncedFetch.cancel();
    if (searchTerm.trim() !== '') {
      fetchSuggestions(searchTerm);
    }
  };

  return (
    <div className="relative w-64">
      <form onSubmit={handleSubmit} className="relative">
        <input
          type="text"
          placeholder="Search movies..."
          value={searchTerm}
          onChange={handleInputChange}
          className="pl-10 pr-10 py-2 border rounded-lg w-full"
        />
        <Search className="absolute left-3 top-2.5 text-gray-400" size={20} />
        {searchTerm && (
          <button
            type="button"
            onClick={handleClear}
            className="absolute right-3 top-2.5 text-gray-400 hover:text-gray-600"
            aria-label="Clear search"
          >
            <X size={20} />
          </button>
        )}
      </form>
      {isSearching && (
        <div className="mt-1 text-sm text-gray-500">Searching...</div>
      )}
      {error && (
        <div className="mt-1 text-sm text-red-500">{error}</div>
      )}
      {/* Display live search suggestions */}
      {suggestions.length > 0 && (
        <div className="absolute bg-white border rounded mt-1 w-full shadow-md z-10">
          <ul>
            {suggestions.map((suggestion, index) => (
              <li
                key={index}
                className="px-4 py-2 hover:bg-gray-100 cursor-pointer"
                onClick={() => handleSuggestionClick(suggestion)}
              >
                {suggestion.title}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default SearchBar;
