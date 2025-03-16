import React, { useState } from 'react';
import PropTypes from 'prop-types';

const UserPreferences = ({ onUpdatePreferences }) => {
  // fairnessPreference: 0 means "More Personalized", 100 means "More Fair"
  const [fairnessPreference, setFairnessPreference] = useState(50);

  const handleSliderChange = (e) => {
    setFairnessPreference(parseInt(e.target.value, 10));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (onUpdatePreferences) {
      // Send the selected preference to the parent component/API
      onUpdatePreferences({ fairnessPreference });
    }
  };

  return (
    <div className="user-preferences p-4 border rounded my-4">
      <h3 className="text-xl font-bold mb-2">Adjust Recommendation Preferences</h3>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="fairnessPreference" className="block text-sm font-medium mb-1">
            Fairness vs. Personalization
          </label>
          <input
            id="fairnessPreference"
            type="range"
            min="0"
            max="100"
            value={fairnessPreference}
            onChange={handleSliderChange}
            className="w-full"
          />
          <div className="flex justify-between text-xs mt-1">
            <span>More Personalized</span>
            <span>More Fair</span>
          </div>
          <p className="mt-1 text-sm">
            Selected: <strong>{fairnessPreference}</strong>
          </p>
        </div>
        <button
          type="submit"
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
        >
          Save Preferences
        </button>
      </form>
    </div>
  );
};

UserPreferences.propTypes = {
  onUpdatePreferences: PropTypes.func.isRequired,
};

export default UserPreferences;
