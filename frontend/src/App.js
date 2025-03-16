// frontend/src/App.js
import React, { useState, useEffect } from 'react';
import { Star } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { getRecommendations } from './services/movieService';
import MovieList from './components/MovieList';
import LoginForm from './components/LoginForm';
import RegisterForm from './components/RegisterForm';
import DarkModeToggle from './components/DarkModeToggle';


// Helper functions for token handling
const getToken = () => {
  try {
    return localStorage.getItem('userToken');
  } catch (error) {
    console.error("Error accessing localStorage", error);
    return null;
  }
};

const setToken = (token) => {
  try {
    localStorage.setItem('userToken', token);
  } catch (error) {
    console.error("Error setting token in localStorage", error);
  }
};

const removeToken = () => {
  try {
    localStorage.removeItem('userToken');
  } catch (error) {
    console.error("Error removing token from localStorage", error);
  }
};

// Dummy implementation of getUserIdFromToken using base64 decoding.
// In production, consider using a robust library like jwt-decode.
const getUserIdFromToken = (token) => {
  try {
    const payload = token.split('.')[1];
    const decoded = JSON.parse(atob(payload));
    return decoded.userId;
  } catch (error) {
    console.error("Invalid token:", error);
    return null;
  }
};

// ErrorBoundary component to catch runtime errors
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, info) {
    console.error("ErrorBoundary caught an error", error, info);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="p-4">
          <h2>Something went wrong.</h2>
          <p>{this.state.error?.toString()}</p>
        </div>
      );
    }
    return this.props.children;
  }
}

const MovieRecommendationApp = () => {
  const [activeTab, setActiveTab] = useState('recommendations');
  const [recommendations, setRecommendations] = useState(null);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch recommendations when login state changes
  useEffect(() => {
    let isMounted = true;
    const fetchRecommendations = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const token = getToken();
        if (token) {
          const userId = getUserIdFromToken(token);
          if (!userId) throw new Error("Invalid token. Please log in again.");
          const data = await getRecommendations(userId, token);
          if (isMounted) setRecommendations(data);
        }
      } catch (err) {
        console.error('Error fetching recommendations:', err);
        if (isMounted) setError('Failed to load recommendations. Please try again later.');
      } finally {
        if (isMounted) setIsLoading(false);
      }
    };
    const token = getToken();
    if (token) {
      setIsLoggedIn(true);
      fetchRecommendations();
    }
    return () => { isMounted = false; };
  }, [isLoggedIn]);

  const handleLoginSuccess = (token) => {
    setToken(token);
    setIsLoggedIn(true);
    // Optionally trigger fetching of recommendations here
  };

  const handleLogout = () => {
    removeToken();
    setIsLoggedIn(false);
    setRecommendations(null);
  };

  const renderSideNav = () => (
    <div className="w-64 bg-gray-100 h-screen p-4 border-r">
      <div className="space-y-4">
        <h2 className="text-xl font-bold mb-6">MovieLens</h2>
        <button
          onClick={() => setActiveTab('recommendations')}
          className={`flex items-center space-x-2 w-full p-2 rounded ${
            activeTab === 'recommendations' ? 'bg-blue-100' : 'hover:bg-gray-200'
          }`}
        >
          <Star size={20} />
          <span>Recommendations</span>
        </button>
        {/* Additional navigation buttons can be added here */}
      </div>
    </div>
  );

  const renderHeader = () => (
    <div className="bg-white border-b p-4">
      <div className="max-w-4xl mx-auto flex justify-between items-center">
        <span>Movie Recommendation App</span>
        {isLoggedIn && (
          <div className="flex items-center space-x-4">
            <DarkModeToggle />
            <button
              onClick={handleLogout}
              className="text-blue-600 hover:underline"
            >
              Logout
            </button>
          </div>
        )}
      </div>
    </div>
  );

  const renderContent = () => {
    if (!isLoggedIn) {
      return (
        <div className="max-w-4xl mx-auto p-6">
          <h2 className="text-xl font-semibold mb-6">Welcome to MovieLens!</h2>
          <p>Please log in or create an account to get started.</p>
          <div className="flex gap-4 mt-6">
            <LoginForm onLoginSuccess={handleLoginSuccess} />
            <RegisterForm />
          </div>
        </div>
      );
    }
    if (isLoading) {
      return (
        <div className="max-w-4xl mx-auto p-6">
          <p>Loading recommendations...</p>
        </div>
      );
    }
    if (error) {
      return (
        <div className="max-w-4xl mx-auto p-6">
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        </div>
      );
    }
    return (
      <div className="max-w-4xl mx-auto p-6">
        {activeTab === 'recommendations' && (
          <div>
            <ContinueWatching onSelect={(movie) => console.log("Resume movie:", movie)} />
            <h2 className="text-xl font-semibold mb-6">Recommended for You</h2>
            <MovieList movies={recommendations} />
          </div>
        )}
        {/* Additional tab content can be rendered here */}
      </div>
    );
  };

  return (
    <ErrorBoundary>
      <div className="flex min-h-screen bg-gray-50">
        {/* Side Navigation */}
        <div className="hidden md:block">{renderSideNav()}</div>
        {/* Main Content */}
        <div className="flex-1">
          {renderHeader()}
          {renderContent()}
        </div>
      </div>
    </ErrorBoundary>
  );
};

export default MovieRecommendationApp;
