import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FaHistory, FaCalendar, FaClock, FaFilter, FaSearch, FaDownload, FaTrash } from 'react-icons/fa';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
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
  Filler
);

export default function History() {
  const [predictions, setPredictions] = useState([]);
  const [filteredPredictions, setFilteredPredictions] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterRisk, setFilterRisk] = useState('all');
  const [sortBy, setSortBy] = useState('newest');
  const [selectedIds, setSelectedIds] = useState([]);
  const [stats, setStats] = useState({
    total: 0,
    avgRisk: 0,
    highRisk: 0,
    mediumRisk: 0,
    lowRisk: 0
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
          { 
            id: 1, 
            insomnia_probability: 0.25, 
            risk_level: 'Low', 
            timestamp: new Date(Date.now() - 86400000 * 2).toISOString(),
            patientData: { age: 28, bmi: 22.5, sex: 'Female' }
          },
          { 
            id: 2, 
            insomnia_probability: 0.55, 
            risk_level: 'Medium', 
            timestamp: new Date(Date.now() - 86400000).toISOString(),
            patientData: { age: 45, bmi: 28.3, sex: 'Male' }
          },
          { 
            id: 3, 
            insomnia_probability: 0.75, 
            risk_level: 'High', 
            timestamp: new Date().toISOString(),
            patientData: { age: 62, bmi: 31.2, sex: 'Female' }
          }
        ];
      }
      
      setPredictions(preds);
      setFilteredPredictions(preds);
      
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
        lowRisk
      });
      
    } catch (error) {
      console.error('Error:', error);
    }
  }, []);

  useEffect(() => {
    let filtered = [...predictions];
    
    if (filterRisk !== 'all') {
      filtered = filtered.filter(p => p.risk_level.toLowerCase() === filterRisk.toLowerCase());
    }
    
    if (searchTerm) {
      filtered = filtered.filter(p => 
        p.patientData?.age?.toString().includes(searchTerm) ||
        p.risk_level?.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    
    filtered.sort((a, b) => {
      const dateA = new Date(a.timestamp || 0).getTime();
      const dateB = new Date(b.timestamp || 0).getTime();
      return sortBy === 'newest' ? dateB - dateA : dateA - dateB;
    });
    
    setFilteredPredictions(filtered);
  }, [predictions, filterRisk, searchTerm, sortBy]);

  const getRiskClass = (prob) => {
    if (prob < 0.3) return 'bg-green-100 text-green-700 border-green-200';
    if (prob < 0.6) return 'bg-yellow-100 text-yellow-700 border-yellow-200';
    return 'bg-red-100 text-red-700 border-red-200';
  };

  const handleSelectAll = () => {
    if (selectedIds.length === filteredPredictions.length) {
      setSelectedIds([]);
    } else {
      setSelectedIds(filteredPredictions.map(p => p.id));
    }
  };

  const handleSelect = (id) => {
    if (selectedIds.includes(id)) {
      setSelectedIds(selectedIds.filter(i => i !== id));
    } else {
      setSelectedIds([...selectedIds, id]);
    }
  };

  const handleDelete = () => {
    if (selectedIds.length === 0) return;
    
    const newPredictions = predictions.filter(p => !selectedIds.includes(p.id));
    setPredictions(newPredictions);
    setSelectedIds([]);
    
    const stored = localStorage.getItem('predictions-storage');
    if (stored) {
      const parsed = JSON.parse(stored);
      parsed.state.predictions = newPredictions;
      localStorage.setItem('predictions-storage', JSON.stringify(parsed));
    }
  };

  const handleExport = () => {
    const dataToExport = selectedIds.length > 0 
      ? predictions.filter(p => selectedIds.includes(p.id))
      : filteredPredictions;
    
    const csv = [
      ['ID', 'Date', 'Risk Level', 'Probability', 'Age', 'BMI', 'Sex'].join(','),
      ...dataToExport.map(p => [
        p.id,
        new Date(p.timestamp).toLocaleString(),
        p.risk_level,
        (p.insomnia_probability * 100).toFixed(1) + '%',
        p.patientData?.age || 'N/A',
        p.patientData?.bmi || 'N/A',
        p.patientData?.sex || 'N/A'
      ].join(','))
    ].join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'predictions_export.csv';
    a.click();
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-7xl mx-auto"
    >
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-primary-600 to-secondary-500 bg-clip-text text-transparent">
          Prediction History
        </h1>
        <div className="flex gap-3">
          <button
            onClick={handleExport}
            className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition"
          >
            <FaDownload /> Export
          </button>
          {selectedIds.length > 0 && (
            <button
              onClick={handleDelete}
              className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition"
            >
              <FaTrash /> Delete ({selectedIds.length})
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="bg-white rounded-2xl shadow-lg p-6 border-l-4 border-blue-500">
          <p className="text-gray-500 text-sm">Total Assessments</p>
          <p className="text-3xl font-bold">{stats.total}</p>
        </div>
        <div className="bg-white rounded-2xl shadow-lg p-6 border-l-4 border-purple-500">
          <p className="text-gray-500 text-sm">Average Risk</p>
          <p className="text-3xl font-bold">{stats.avgRisk}%</p>
        </div>
        <div className="bg-white rounded-2xl shadow-lg p-6 border-l-4 border-red-500">
          <p className="text-gray-500 text-sm">High Risk</p>
          <p className="text-3xl font-bold">{stats.highRisk}</p>
        </div>
        <div className="bg-white rounded-2xl shadow-lg p-6 border-l-4 border-green-500">
          <p className="text-gray-500 text-sm">Low Risk</p>
          <p className="text-3xl font-bold">{stats.lowRisk}</p>
        </div>
      </div>

      <div className="bg-white rounded-2xl shadow-lg p-6 mb-8">
        <div className="grid md:grid-cols-4 gap-4">
          <div className="relative">
            <FaSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              placeholder="Search assessments..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
            />
          </div>
          
          <div className="relative">
            <FaFilter className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
            <select
              value={filterRisk}
              onChange={(e) => setFilterRisk(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
            >
              <option value="all">All Risk Levels</option>
              <option value="low">Low Risk</option>
              <option value="medium">Medium Risk</option>
              <option value="high">High Risk</option>
            </select>
          </div>
          
          <div className="relative">
            <FaClock className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
            >
              <option value="newest">Newest First</option>
              <option value="oldest">Oldest First</option>
            </select>
          </div>
          
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={selectedIds.length === filteredPredictions.length && filteredPredictions.length > 0}
              onChange={handleSelectAll}
              className="w-4 h-4 text-primary-600 rounded"
            />
            <span className="text-sm text-gray-600">Select All</span>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-2xl shadow-lg p-6">
        {filteredPredictions.length > 0 ? (
          <div className="space-y-4">
            {filteredPredictions.map((pred) => {
              const bgColor = pred.insomnia_probability < 0.3 ? 'bg-green-100' : pred.insomnia_probability < 0.6 ? 'bg-yellow-100' : 'bg-red-100';
              const textColor = pred.insomnia_probability < 0.3 ? 'text-green-600' : pred.insomnia_probability < 0.6 ? 'text-yellow-600' : 'text-red-600';
              
              return (
                <motion.div
                  key={pred.id}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex items-center justify-between p-4 bg-gray-50 rounded-xl hover:bg-gray-100 transition-colors"
                >
                  <div className="flex items-center gap-4">
                    <input
                      type="checkbox"
                      checked={selectedIds.includes(pred.id)}
                      onChange={() => handleSelect(pred.id)}
                      className="w-4 h-4 text-primary-600 rounded"
                    />
                    <div className={'w-10 h-10 rounded-full flex items-center justify-center ' + bgColor}>
                      <FaCalendar className={textColor} />
                    </div>
                    <div>
                      <p className="font-medium">
                        Assessment {new Date(pred.timestamp).toLocaleDateString('en-US', {
                          month: 'long',
                          day: 'numeric',
                          year: 'numeric'
                        })}
                      </p>
                      <p className="text-sm text-gray-500">
                        {new Date(pred.timestamp).toLocaleTimeString()} • 
                        Age: {pred.patientData?.age || 'N/A'} • 
                        BMI: {pred.patientData?.bmi || 'N/A'}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className={'px-4 py-2 rounded-full text-sm font-medium border ' + getRiskClass(pred.insomnia_probability)}>
                      {(pred.insomnia_probability * 100).toFixed(1)}%
                    </span>
                    <span className={'px-3 py-1 rounded-full text-xs font-medium ' + (
                      pred.risk_level === 'Low' ? 'bg-green-100 text-green-700' :
                      pred.risk_level === 'Medium' ? 'bg-yellow-100 text-yellow-700' :
                      'bg-red-100 text-red-700'
                    )}>
                      {pred.risk_level}
                    </span>
                  </div>
                </motion.div>
              );
            })}
          </div>
        ) : (
          <div className="text-center py-12">
            <FaHistory className="text-6xl text-gray-300 mx-auto mb-4" />
            <p className="text-gray-400">No assessment history found</p>
          </div>
        )}
      </div>
    </motion.div>
  );
}
