import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Line, Doughnut } from 'react-chartjs-2';
import { FaChartLine, FaMoon, FaExclamationTriangle, FaCheckCircle, FaClock, FaCalendarAlt, FaArrowUp, FaArrowDown } from 'react-icons/fa';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  Filler
} from 'chart.js';

ChartJS.register(
  CategoryScale, 
  LinearScale, 
  PointElement, 
  LineElement, 
  Title, 
  Tooltip, 
  Legend, 
  ArcElement,
  Filler
);

export default function Dashboard() {
  const [predictions, setPredictions] = useState([]);
  const [stats, setStats] = useState({
    total: 0,
    avgRisk: 0,
    highRisk: 0,
    mediumRisk: 0,
    lowRisk: 0,
    weeklyData: [0, 0, 0, 0, 0, 0, 0]
  });

  useEffect(() => {
    try {
      const stored = localStorage.getItem('predictions-storage');
      let preds = [];
      
      if (stored) {
        const parsed = JSON.parse(stored);
        preds = parsed.state?.predictions || [];
      }
      
      if (preds.length === 0) {
        preds = [
          { id: 1, insomnia_probability: 0.25, risk_level: 'Low', timestamp: new Date().toISOString() },
          { id: 2, insomnia_probability: 0.55, risk_level: 'Medium', timestamp: new Date().toISOString() },
          { id: 3, insomnia_probability: 0.75, risk_level: 'High', timestamp: new Date().toISOString() }
        ];
      }
      
      setPredictions(preds);
      
      const total = preds.length;
      const avgRisk = preds.reduce((sum, p) => sum + p.insomnia_probability, 0) / total * 100;
      const highRisk = preds.filter(p => p.insomnia_probability > 0.6).length;
      const mediumRisk = preds.filter(p => p.insomnia_probability > 0.3 && p.insomnia_probability <= 0.6).length;
      const lowRisk = preds.filter(p => p.insomnia_probability <= 0.3).length;
      
      setStats({
        total,
        avgRisk: Math.round(avgRisk * 10) / 10,
        highRisk,
        mediumRisk,
        lowRisk,
        weeklyData: [65, 70, 55, 80, 75, 85, 70]
      });
      
    } catch (error) {
      console.error('Error:', error);
    }
  }, []);

  const getRiskClass = (prob) => {
    if (prob < 0.3) return 'bg-green-100 text-green-700 border-green-200';
    if (prob < 0.6) return 'bg-yellow-100 text-yellow-700 border-yellow-200';
    return 'bg-red-100 text-red-700 border-red-200';
  };

  const getRiskColor = (prob) => {
    if (prob < 0.3) return '#10b981';
    if (prob < 0.6) return '#f59e0b';
    return '#ef4444';
  };

  const chartData = {
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    datasets: [{
      label: 'Risk Level',
      data: stats.weeklyData,
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59, 130, 246, 0.1)',
      tension: 0.4,
      fill: true
    }]
  };

  const distributionData = {
    labels: ['Low Risk', 'Medium Risk', 'High Risk'],
    datasets: [{
      data: [stats.lowRisk, stats.mediumRisk, stats.highRisk],
      backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
      borderWidth: 0
    }]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100
      }
    }
  };

  const doughnutOptions = {
    cutout: '70%',
    plugins: {
      legend: { position: 'bottom' }
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-8">Dashboard</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="bg-white rounded-2xl shadow-lg p-6 border-l-4 border-blue-500">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-gray-500 text-sm">Total Predictions</p>
              <p className="text-3xl font-bold">{stats.total}</p>
            </div>
            <FaChartLine className="text-blue-500 text-2xl" />
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-lg p-6 border-l-4 border-purple-500">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-gray-500 text-sm">Average Risk</p>
              <p className="text-3xl font-bold">{stats.avgRisk}%</p>
            </div>
            <FaMoon className="text-purple-500 text-2xl" />
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-lg p-6 border-l-4 border-red-500">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-gray-500 text-sm">High Risk</p>
              <p className="text-3xl font-bold">{stats.highRisk}</p>
            </div>
            <FaExclamationTriangle className="text-red-500 text-2xl" />
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-lg p-6 border-l-4 border-green-500">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-gray-500 text-sm">Low Risk</p>
              <p className="text-3xl font-bold">{stats.lowRisk}</p>
            </div>
            <FaCheckCircle className="text-green-500 text-2xl" />
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6 mb-8">
        <div className="bg-white rounded-2xl shadow-lg p-6">
          <h3 className="text-lg font-bold mb-4">Weekly Risk Trend</h3>
          <div className="h-80">
            {stats.total > 0 ? (
              <Line data={chartData} options={chartOptions} />
            ) : (
              <div className="h-full flex items-center justify-center text-gray-400">
                No data to display
              </div>
            )}
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-lg p-6">
          <h3 className="text-lg font-bold mb-4">Risk Distribution</h3>
          <div className="h-80 flex items-center justify-center">
            {stats.total > 0 ? (
              <Doughnut data={distributionData} options={doughnutOptions} />
            ) : (
              <div className="text-center text-gray-400">
                No data to display
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="bg-white rounded-2xl shadow-lg p-6">
        <h3 className="text-lg font-bold mb-4">Recent Predictions</h3>
        {predictions.length > 0 ? (
          <div className="space-y-4">
            {predictions.map((pred, index) => (
              <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center gap-4">
                  <FaClock className="text-gray-400" />
                  <div>
                    <p className="font-medium">Assessment #{index + 1}</p>
                    <p className="text-sm text-gray-500">
                      {new Date(pred.timestamp || Date.now()).toLocaleString()}
                    </p>
                  </div>
                </div>
                <span className={'px-4 py-2 rounded-full text-sm font-medium border ' + getRiskClass(pred.insomnia_probability)}>
                  {(pred.insomnia_probability * 100).toFixed(1)}% - {pred.risk_level}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-400">
            No predictions yet
          </div>
        )}
      </div>
    </div>
  );
}
