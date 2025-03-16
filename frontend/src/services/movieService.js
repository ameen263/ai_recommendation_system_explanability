/**
 * movieService.js
 *
 * Provides helper functions for fetching movie recommendations and explanations.
 */

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

// Shared timeout constant to avoid magic numbers
const DEFAULT_TIMEOUT = 8000;

/**
 * Fetch with timeout - wraps fetch call with a timeout.
 * @param {string} resource - The URL to fetch.
 * @param {Object} options - Fetch options.
 * @returns {Promise<Response>} The fetch response promise.
 */
const fetchWithTimeout = (resource, { timeout = DEFAULT_TIMEOUT, ...fetchOptions } = {}) => {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  return fetch(resource, { ...fetchOptions, signal: controller.signal })
      .then(response => {
        clearTimeout(id);
        return response;
      });
};

/**
 * Fetch with retry - attempts to fetch the resource multiple times if needed.
 * @param {string} resource - The URL to fetch.
 * @param {Object} options - Fetch options.
 * @param {number} retries - Number of retries.
 * @param {number} backoff - Backoff time in milliseconds.
 * @returns {Promise<Response>} The fetch response promise.
 */
const fetchWithRetry = async (resource, options, retries = 3, backoff = 300) => {
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      return await fetchWithTimeout(resource, options);
    } catch (error) {
      if (attempt < retries - 1) {
        await new Promise(res => setTimeout(res, backoff));
        backoff *= 2;
      } else {
        throw error;
      }
    }
  }
};

/**
 * Get movie recommendations for a specific user.
 * @param {number} userId - User ID.
 * @param {string} token - (Optional) JWT token for authentication.
 * @returns {Promise<Object>} The recommendations data.
 */
export const getRecommendations = async (userId, token) => {
  try {
    if (!Number.isInteger(userId) || userId <= 0) {
      throw new Error('Invalid userId: must be a positive integer.');
    }

    const response = await fetchWithRetry(`${API_BASE_URL}/recommend/${userId}?top_n=10`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        ...(token && { 'Authorization': `Bearer ${token}` }),
      },
      timeout: DEFAULT_TIMEOUT, // Use shared timeout constant
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to get recommendations');
    }

    const data = await response.json();
    if (typeof data !== 'object' || data === null) {
      throw new Error('Unexpected response structure.');
    }

    return data;

  } catch (error) {
    console.error('Error in getRecommendations:', error);
    throw error;
  }
};

/**
 * Get an explanation for a specific movie recommendation.
 * @param {number} userId - User ID.
 * @param {number} movieId - Movie ID.
 * @param {string} detailLevel - Level of detail ("simple" or "detailed").
 * @param {string} token - (Optional) JWT token for authentication.
 * @returns {Promise<Object>} The explanation data.
 */
export const getExplanation = async (userId, movieId, detailLevel = "simple", token) => {
  try {
    // Validate parameters
    if (!Number.isInteger(userId) || userId <= 0) {
      throw new Error('Invalid userId: must be a positive integer.');
    }
    if (!Number.isInteger(movieId) || movieId <= 0) {
      throw new Error('Invalid movieId: must be a positive integer.');
    }
    if (!['simple', 'detailed'].includes(detailLevel)) {
      throw new Error('Invalid detailLevel: must be "simple" or "detailed".');
    }

    const response = await fetchWithRetry(`${API_BASE_URL}/explain`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token && { 'Authorization': `Bearer ${token}` }),
      },
      body: JSON.stringify({ user_id: userId, movie_id: movieId, detail_level: detailLevel }),
      timeout: DEFAULT_TIMEOUT, // Use shared timeout constant
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to get explanation');
    }

    const data = await response.json();
    if (typeof data !== 'object' || data === null) {
      throw new Error('Unexpected response structure.');
    }

    return data;

  } catch (error) {
    console.error('Error in getExplanation:', error);
    throw error;
  }
};

export default {
  getRecommendations,
  getExplanation,
};