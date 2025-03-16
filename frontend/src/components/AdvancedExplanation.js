import React from 'react';
import PropTypes from 'prop-types';

const AdvancedExplanation = ({ explanationData }) => {
  if (!explanationData) {
    return <div className="p-4 border rounded my-4">No advanced explanation available.</div>;
  }

  if (explanationData.error) {
    return (
      <div className="p-4 border rounded my-4 bg-red-100 text-red-700">
        <h3 className="text-xl font-bold mb-2">Advanced Explanation Error</h3>
        <p>Error: {explanationData.error}</p>
      </div>
    );
  }

  return (
    <div className="advanced-explanation p-4 border rounded my-4 bg-gray-50">
      <h3 className="text-2xl font-bold mb-4">Advanced Explanation Details</h3>

      <div className="mb-3">
        <h4 className="text-lg font-semibold">User Contribution</h4>
        <p>
          {explanationData.user_contribution !== undefined
            ? explanationData.user_contribution.toFixed(4)
            : "N/A"}
        </p>
      </div>

      <div className="mb-3">
        <h4 className="text-lg font-semibold">Movie Contribution</h4>
        <p>
          {explanationData.movie_contribution !== undefined
            ? explanationData.movie_contribution.toFixed(4)
            : "N/A"}
        </p>
      </div>

      <div className="mb-3">
        <h4 className="text-lg font-semibold">Convergence Delta</h4>
        <p>
          {explanationData.convergence_delta !== undefined
            ? explanationData.convergence_delta.toFixed(4)
            : "N/A"}
        </p>
      </div>

      <div className="mb-3">
        <h4 className="text-lg font-semibold">Interpretation</h4>
        <p>{explanationData.interpretation || "N/A"}</p>
      </div>
    </div>
  );
};

AdvancedExplanation.propTypes = {
  explanationData: PropTypes.shape({
    user_contribution: PropTypes.number,
    movie_contribution: PropTypes.number,
    convergence_delta: PropTypes.number,
    interpretation: PropTypes.string,
    error: PropTypes.string,
  }),
};
