import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import api from '../services/api';

const useAuthStore = create(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      
      login: async (username, password) => {
        try {
          const response = await api.post('/auth/login', { username, password });
          const data = response.data;
          set({ 
            user: data, 
            token: data.token, 
            isAuthenticated: true 
          });
          // Store token in localStorage for api interceptor
          localStorage.setItem('auth_token', data.token);
          return data;
        } catch (error) {
          throw error;
        }
      },
      
      register: async (userData) => {
        try {
          const response = await api.post('/auth/register', userData);
          const data = response.data;
          set({ 
            user: data, 
            token: data.token, 
            isAuthenticated: true 
          });
          // Store token in localStorage for api interceptor
          localStorage.setItem('auth_token', data.token);
          return data;
        } catch (error) {
          throw error;
        }
      },
      
      logout: () => {
        localStorage.removeItem('auth_token');
        set({ 
          user: null, 
          token: null, 
          isAuthenticated: false 
        });
      },
      
      getToken: () => get().token
    }),
    {
      name: 'auth-storage',
    }
  )
);

export default useAuthStore;
