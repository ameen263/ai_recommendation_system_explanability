import React, { useState, useEffect } from 'react';
import ExplanationFeedback from './ExplanationFeedback';

const RecommendationExplanation = ({ userId, movieId, token }) => {
  const [explanation, setExplanation] = useState(null);
  const [error, setError] = useState('');
  const [detailLevel, setDetailLevel] = useState("simple");
  const [loading, setLoading] = useState(false);

  const fetchExplanation = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch(
        `${process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000'}/explain`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(token && { 'Authorization': `Bearer ${token}` }),
          },
          body: JSON.stringify({ user_id: userId, movie_id: movieId, detail_level: detailLevel }),
        }
      );
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Failed to fetch explanation');
      }
      const data = await res.json();
      setExplanation(data.explanation);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Refetch explanation when userId, movieId, or detailLevel changes.
  useEffect(() => {
    if (userId && movieId) {
      fetchExplanation();
    }
  }, [userId, movieId, detailLevel]);

  const handleDetailChange = (e) => {
    setDetailLevel(e.target.value);
  };

  return (
    <div className="recommendation-explanation p-4 border rounded">
      <h2 className="text-xl font-bold mb-2">Recommendation Explanation</h2>

      <div className="mb-4">
        <label htmlFor="detailLevel" className="mr-2">Explanation Detail:</label>
        <select
          id="detailLevel"
          value={detailLevel}
          onChange={handleDetailChange}
          className="p-2 border rounded"
        >
          <option value="simple">Simple</option>
          <option value="detailed">Advanced</option>
        </select>
      </div>

      {loading ? (
        <p>Loading explanation...</p>
      ) : error ? (
        <p className="text-red-500">{error}</p>
      ) : (
        <div>
          {explanation ? (
            <pre className="bg-gray-100 p-2 rounded">
              {typeof explanation === "object"
                ? JSON.stringify(explanation, null, 2)
                : explanation}
            </pre>
          ) : (
            <p>No explanation available.</p>
          )}
        </div>
      )}
      
      {/* Interactive feedback component */}
      <ExplanationFeedback onSubmitFeedback={(feedback) => console.log("Feedback submitted:", feedback)} />
    </div>
  );
};

export default RecommendationExplanation;
