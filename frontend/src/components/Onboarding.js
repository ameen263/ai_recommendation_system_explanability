import React, { useState } from 'react';

const Onboarding = ({ onSubmitPreferences }) => {
  // Define available genres (you can expand this list as needed)
  const availableGenres = [
    "Action",
    "Comedy",
    "Drama",
    "Horror",
    "Sci-Fi",
    "Romance",
    "Documentary",
    "Thriller",
    "Animation"
  ];

  const [selectedGenres, setSelectedGenres] = useState([]);

  const handleCheckboxChange = (genre) => {
    if (selectedGenres.includes(genre)) {
      setSelectedGenres(selectedGenres.filter(g => g !== genre));
    } else {
      setSelectedGenres([...selectedGenres, genre]);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (onSubmitPreferences) {
      onSubmitPreferences(selectedGenres);
    }
  };

  return (
    <div className="onboarding p-4 border rounded">
      <h2 className="text-2xl font-bold mb-4">Select Your Favorite Genres</h2>
      <form onSubmit={handleSubmit}>
        <div className="grid grid-cols-2 gap-4 mb-4">
          {availableGenres.map((genre) => (
            <label key={genre} className="flex items-center">
              <input
                type="checkbox"
                value={genre}
                checked={selectedGenres.includes(genre)}
                onChange={() => handleCheckboxChange(genre)}
                className="mr-2"
              />
              {genre}
            </label>
          ))}
        </div>
        <button 
          type="submit" 
          className="px-4 py-2 bg-blue-600 text-white rounded"
        >
          Save Preferences
        </button>
      </form>
    </div>
  );
};

export default Onboarding;
