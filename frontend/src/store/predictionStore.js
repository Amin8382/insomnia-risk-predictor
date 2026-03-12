import { create } from 'zustand';
import { persist } from 'zustand/middleware';

const usePredictionStore = create(
  persist(
    (set, get) => ({
      predictions: [],
      
      addPrediction: (prediction) => {
        const newPrediction = {
          ...prediction,
          id: Date.now(),
          timestamp: new Date().toISOString(),
          patientData: prediction.patientData || {}
        };
        
        set((state) => ({
          predictions: [newPrediction, ...state.predictions].slice(0, 100)
        }));
      },
      
      clearPredictions: () => set({ predictions: [] }),
      
      getStats: () => {
        const predictions = get().predictions;
        const total = predictions.length;
        
        if (total === 0) {
          return {
            total: 0,
            avgRisk: 0,
            highRisk: 0,
            mediumRisk: 0,
            lowRisk: 0,
            weeklyData: [0, 0, 0, 0, 0, 0, 0]
          };
        }
        
        const avgRisk = predictions.reduce((sum, p) => sum + p.insomnia_probability, 0) / total;
        const highRisk = predictions.filter(p => p.insomnia_probability > 0.6).length;
        const mediumRisk = predictions.filter(p => p.insomnia_probability > 0.3 && p.insomnia_probability <= 0.6).length;
        const lowRisk = predictions.filter(p => p.insomnia_probability <= 0.3).length;
        
        // Get last 7 days of data for weekly trend
        const weeklyData = [0, 0, 0, 0, 0, 0, 0];
        const now = new Date();
        predictions.slice(0, 50).forEach(p => {
          if (p.timestamp) {
            const daysAgo = Math.floor((now - new Date(p.timestamp)) / (1000 * 60 * 60 * 24));
            if (daysAgo >= 0 && daysAgo < 7) {
              weeklyData[6 - daysAgo] = (weeklyData[6 - daysAgo] + p.insomnia_probability * 100) / 2;
            }
          }
        });
        
        return {
          total,
          avgRisk: avgRisk * 100,
          highRisk,
          mediumRisk,
          lowRisk,
          weeklyData: weeklyData.map(v => Math.round(v * 10) / 10)
        };
      }
    }),
    {
      name: 'predictions-storage',
    }
  )
);

export default usePredictionStore;
