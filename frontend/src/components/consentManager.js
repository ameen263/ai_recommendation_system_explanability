import React, { useState, useEffect } from 'react';

const ConsentManager = ({ onConsentUpdate }) => {
  const [consent, setConsent] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // Load consent setting from localStorage on mount
  useEffect(() => {
    try {
      const storedConsent = localStorage.getItem('userConsent');
      if (storedConsent !== null) {
        setConsent(storedConsent === 'true');
      } else {
        // Default to false if no consent info is found
        setConsent(false);
      }
      setLoading(false);
    } catch (err) {
      console.error("Error loading consent:", err);
      setError("Failed to load consent settings.");
      setLoading(false);
    }
  }, []);

  const handleConsentChange = (e) => {
    const newConsent = e.target.checked;
    setConsent(newConsent);
    try {
      localStorage.setItem('userConsent', newConsent);
      // Optionally, send the updated consent to your backend via an API call.
      if (onConsentUpdate) {
        onConsentUpdate(newConsent);
      }
    } catch (err) {
      console.error("Error saving consent:", err);
      setError("Failed to save consent settings.");
    }
  };

  if (loading) {
    return <div>Loading consent settings...</div>;
  }

  return (
    <div className="consent-manager p-4 border rounded my-4">
      <h3 className="text-xl font-bold mb-2">Privacy & Consent Settings</h3>
      {error && <p className="text-red-500 mb-2">{error}</p>}
      <p className="mb-4">
        We value your privacy. Please indicate whether you consent to the processing of your data for personalized recommendations.
      </p>
      <label className="flex items-center">
        <input
          type="checkbox"
          checked={consent}
          onChange={handleConsentChange}
          className="mr-2"
        />
        <span>I consent to data processing for personalized recommendations.</span>
      </label>
    </div>
  );
};

export default ConsentManager;
