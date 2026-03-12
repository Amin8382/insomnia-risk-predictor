import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Line } from 'react-chartjs-2';
import { FaHeartbeat, FaClock, FaBell } from 'react-icons/fa';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

export default function Monitoring() {
  const predictions = [];
  
  const labels = [];
  for (let i = 0; i < 24; i++) {
    labels.push(i + ':00');
  }
  
  const [monitoringData] = useState({
    labels: labels,
    datasets: [{
      label: 'Risk Level',
      data: Array.from({ length: 24 }, () => Math.random() * 100),
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59, 130, 246, 0.1)',
      tension: 0.4
    }]
  });

  const currentRisk = predictions.length > 0 
    ? (predictions[0].insomnia_probability * 100).toFixed(1) 
    : '0.0';

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-8">
      <h1 className="text-3xl font-bold">Live Monitoring</h1>

      <div className="grid md:grid-cols-3 gap-6">
        <div className="card-premium p-6">
          <div className="flex items-center gap-4">
            <FaHeartbeat className="text-4xl text-primary-500" />
            <div>
              <p className="text-gray-500 text-sm">Current Risk</p>
              <p className="text-3xl font-bold">{currentRisk}%</p>
            </div>
          </div>
        </div>
        <div className="card-premium p-6">
          <div className="flex items-center gap-4">
            <FaClock className="text-4xl text-primary-500" />
            <div>
              <p className="text-gray-500 text-sm">Last Update</p>
              <p className="text-lg font-medium">Just now</p>
            </div>
          </div>
        </div>
        <div className="card-premium p-6">
          <div className="flex items-center gap-4">
            <FaBell className="text-4xl text-primary-500" />
            <div>
              <p className="text-gray-500 text-sm">Alerts</p>
              <p className="text-3xl font-bold">0</p>
            </div>
          </div>
        </div>
      </div>

      <div className="card-premium p-6">
        <h3 className="text-lg font-bold mb-4">24-Hour Risk Monitoring</h3>
        <Line data={monitoringData} options={{ responsive: true }} height={100} />
      </div>

      <div className="card-premium p-6">
        <h3 className="text-lg font-bold mb-4">Recent Alerts</h3>
        <p className="text-gray-500 text-center py-8">No alerts in the last 24 hours</p>
      </div>
    </motion.div>
  );
}
