import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export const useAuthStore = create(
  persist(
    (set) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      login: (userData) => set({ user: userData.user, token: userData.token, isAuthenticated: true }),
      logout: () => set({ user: null, token: null, isAuthenticated: false }),
      updateProfile: (profileData) => set((state) => ({ user: { ...state.user, ...profileData } }))
    }),
    { name: 'auth-storage' }
  )
);

export const usePredictionStore = create((set, get) => ({
  predictions: [],
  addPrediction: (prediction) => set((state) => ({ predictions: [prediction, ...state.predictions].slice(0, 50) })),
  clearPredictions: () => set({ predictions: [] }),
  getStats: () => {
    const predictions = get().predictions;
    return {
      total: predictions.length,
      avgRisk: predictions.reduce((acc, p) => acc + p.insomnia_probability, 0) / (predictions.length || 1),
      highRisk: predictions.filter(p => p.insomnia_probability > 0.6).length,
      mediumRisk: predictions.filter(p => p.insomnia_probability > 0.3 && p.insomnia_probability <= 0.6).length,
      lowRisk: predictions.filter(p => p.insomnia_probability <= 0.3).length
    };
  }
}));

export const useChatStore = create((set) => ({
  messages: [],
  unreadCount: 0,
  addMessage: (message) => set((state) => ({ 
    messages: [...state.messages, message],
    unreadCount: state.unreadCount + 1 
  })),
  markAsRead: () => set({ unreadCount: 0 }),
  clearChat: () => set({ messages: [], unreadCount: 0 })
}));
