import axios from 'axios';

const API_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add token to every request
api.interceptors.request.use(
  (config) => {
    // Get token from localStorage
    const token = localStorage.getItem('auth_token');
    const authStore = localStorage.getItem('auth-storage');
    
    let finalToken = token;
    
    if (!finalToken && authStore) {
      try {
        const parsed = JSON.parse(authStore);
        finalToken = parsed.state?.token;
      } catch (e) {
        // Ignore parse errors
      }
    }
    
    if (finalToken) {
      config.headers.Authorization = 'Bearer ' + finalToken;
    }
    
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.code === 'ECONNREFUSED') {
      throw new Error('Cannot connect to backend server');
    }
    if (error.response?.status === 401) {
      throw new Error('Please login again');
    }
    throw new Error(error.response?.data?.detail || error.message);
  }
);

export const predictionService = {
  predict: async (data) => {
    try {
      const response = await api.post('/predict/', data);
      return response.data;
    } catch (error) {
      console.error('Prediction error:', error);
      throw error;
    }
  },
  getHistory: async () => {
    const response = await api.get('/predict/history');
    return response.data;
  }
};

export default api;
