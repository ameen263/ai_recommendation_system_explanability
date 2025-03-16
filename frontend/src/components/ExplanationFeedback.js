import React, { useState } from 'react';

const ExplanationFeedback = ({ onSubmitFeedback }) => {
  const [rating, setRating] = useState(null);
  const [comment, setComment] = useState('');

  // Handle rating selection (thumbs up or thumbs down)
  const handleRatingChange = (value) => {
    setRating(value);
  };

  // Handle feedback submission
  const handleSubmit = () => {
    if (rating === null) {
      alert("Please select a rating before submitting.");
      return;
    }
    const feedback = { rating, comment };
    if (onSubmitFeedback) {
      onSubmitFeedback(feedback);
    }
    // Optionally clear feedback after submission
    setRating(null);
    setComment('');
  };

  return (
    <div className="explanation-feedback p-4 border rounded my-4">
      <h3 className="text-lg font-bold mb-2">Feedback on Explanation</h3>
      <div className="mb-2">
        <span className="mr-2">How useful was this explanation?</span>
        <button 
          className={`px-3 py-1 border rounded mr-2 ${rating === 'up' ? 'bg-green-500 text-white' : ''}`}
          onClick={() => handleRatingChange('up')}
          aria-label="Rate explanation as useful"
        >
          👍
        </button>
        <button 
          className={`px-3 py-1 border rounded ${rating === 'down' ? 'bg-red-500 text-white' : ''}`}
          onClick={() => handleRatingChange('down')}
          aria-label="Rate explanation as not useful"
        >
          👎
        </button>
      </div>
      <div className="mb-2">
        <textarea 
          className="w-full p-2 border rounded"
          placeholder="Additional comments (optional)"
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          aria-label="Enter additional feedback"
        ></textarea>
      </div>
      <button 
        className="px-4 py-2 bg-blue-600 text-white rounded"
        onClick={handleSubmit}
        aria-label="Submit feedback"
      >
        Submit Feedback
      </button>
    </div>
  );
};

export default ExplanationFeedback;
