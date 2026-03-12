import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FaUser, FaVenusMars, FaSmoking, FaWeight, FaHeartbeat } from 'react-icons/fa';
import { Doughnut } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { predictionService } from '../services/api';
import useAuthStore from '../store/authStore';
import toast from 'react-hot-toast';
import { useNavigate } from 'react-router-dom';

ChartJS.register(ArcElement, Tooltip, Legend);

export default function Predict() {
  const navigate = useNavigate();
  const { user, token, isAuthenticated } = useAuthStore();
  const [formData, setFormData] = useState({
    age: '',
    sex: '',
    ethnicity: '',
    smoking_status: '',
    bmi: '',
    hypertension: 0,
    diabetes: 0,
    asthma: 0,
    copd: 0,
    cancer: 0,
    obesity: 0,
    anxiety_or_depression: 0,
    psychiatric_disorder: 0,
    sleep_disorder_note_count: 0,
    insomnia_billing_code_count: 0,
    anx_depr_billing_code_count: 0,
    psych_note_count: 0,
    insomnia_rx_count: 0,
    joint_disorder_billing_code_count: 0,
    emr_fact_count: 0
  });
  
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showMedical, setShowMedical] = useState(false);

  // Check if user is authenticated
  useEffect(() => {
    if (!isAuthenticated) {
      toast.error('Please login first');
      navigate('/login');
    }
  }, [isAuthenticated, navigate]);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]: type === 'checkbox' ? (checked ? 1 : 0) : value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const patientData = {
        ...formData,
        age: parseFloat(formData.age) || 0,
        bmi: formData.bmi ? parseFloat(formData.bmi) : null
      };
      
      const res = await predictionService.predict(patientData);
      setResult(res);
      toast.success('Prediction saved to database!');
    } catch (error) {
      toast.error(error.message);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (prob) => {
    if (prob < 0.3) return '#10b981';
    if (prob < 0.6) return '#f59e0b';
    return '#ef4444';
  };

  const chartData = result ? {
    labels: ['Risk', 'Remaining'],
    datasets: [{
      data: [result.insomnia_probability * 100, 100 - (result.insomnia_probability * 100)],
      backgroundColor: [getRiskColor(result.insomnia_probability), '#e5e7eb'],
      borderWidth: 0
    }]
  } : null;

  if (!isAuthenticated) {
    return null; // Will redirect via useEffect
  }

  return (
    <div className="max-w-7xl mx-auto">
      <h1 className="text-3xl font-bold mb-8">Risk Assessment</h1>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Form Section */}
        <div className="bg-white rounded-2xl shadow-lg p-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <label className="flex items-center gap-2 font-medium mb-2">
                  <FaUser className="text-primary-500" /> Age
                </label>
                <input
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                  placeholder="Enter age"
                  required
                />
              </div>

              <div>
                <label className="flex items-center gap-2 font-medium mb-2">
                  <FaWeight className="text-primary-500" /> BMI
                </label>
                <input
                  type="number"
                  name="bmi"
                  step="0.1"
                  value={formData.bmi}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                  placeholder="Enter BMI"
                />
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <label className="flex items-center gap-2 font-medium mb-2">
                  <FaVenusMars className="text-primary-500" /> Sex
                </label>
                <select
                  name="sex"
                  value={formData.sex}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                  required
                >
                  <option value="">Select</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>

              <div>
                <label className="block font-medium mb-2">Ethnicity</label>
                <select
                  name="ethnicity"
                  value={formData.ethnicity}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                  required
                >
                  <option value="">Select</option>
                  <option value="Caucasian">Caucasian</option>
                  <option value="African American">African American</option>
                  <option value="Hispanic">Hispanic</option>
                  <option value="Asian">Asian</option>
                </select>
              </div>
            </div>

            <div>
              <label className="flex items-center gap-2 font-medium mb-2">
                <FaSmoking className="text-primary-500" /> Smoking Status
              </label>
              <select
                name="smoking_status"
                value={formData.smoking_status}
                onChange={handleChange}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                required
              >
                <option value="">Select</option>
                <option value="Never">Never</option>
                <option value="Past">Past</option>
                <option value="Current">Current</option>
              </select>
            </div>

            {/* Medical Conditions Toggle */}
            <div>
              <button
                type="button"
                onClick={() => setShowMedical(!showMedical)}
                className="flex items-center gap-2 text-primary-600 font-medium"
              >
                <FaHeartbeat /> {showMedical ? 'Hide' : 'Show'} Medical History
              </button>
            </div>

            {/* Medical Conditions */}
            {showMedical && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="space-y-4 p-4 bg-gray-50 rounded-lg"
              >
                <h3 className="font-semibold">Medical Conditions</h3>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    'hypertension', 'diabetes', 'asthma', 'copd',
                    'cancer', 'obesity', 'anxiety_or_depression', 'psychiatric_disorder'
                  ].map((condition) => (
                    <label key={condition} className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        name={condition}
                        checked={formData[condition] === 1}
                        onChange={handleChange}
                        className="rounded text-primary-600"
                      />
                      <span className="text-sm capitalize">{condition.replace('_', ' ')}</span>
                    </label>
                  ))}
                </div>

                <h3 className="font-semibold mt-4">Healthcare Utilization</h3>
                <div className="space-y-3">
                  <div>
                    <label className="text-sm">Sleep Disorder Notes</label>
                    <input
                      type="number"
                      name="sleep_disorder_note_count"
                      value={formData.sleep_disorder_note_count}
                      onChange={handleChange}
                      className="w-full px-3 py-2 border rounded-lg text-sm"
                      min="0"
                    />
                  </div>
                  <div>
                    <label className="text-sm">Insomnia Billing Codes</label>
                    <input
                      type="number"
                      name="insomnia_billing_code_count"
                      value={formData.insomnia_billing_code_count}
                      onChange={handleChange}
                      className="w-full px-3 py-2 border rounded-lg text-sm"
                      min="0"
                    />
                  </div>
                </div>
              </motion.div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-primary-600 text-white py-3 rounded-lg hover:bg-primary-700 transition font-medium disabled:opacity-50"
            >
              {loading ? 'Processing...' : 'Get Prediction'}
            </button>
          </form>
        </div>

        {/* Results Section */}
        <div className="space-y-6">
          {result ? (
            <>
              <div className="bg-white rounded-2xl shadow-lg p-8 text-center">
                <h3 className="text-xl font-bold mb-6">Your Risk Analysis</h3>
                <div className="relative w-48 h-48 mx-auto mb-4">
                  <Doughnut 
                    data={chartData} 
                    options={{ 
                      cutout: '70%',
                      plugins: { legend: { display: false } }
                    }} 
                  />
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span className="text-3xl font-bold" style={{ color: getRiskColor(result.insomnia_probability) }}>
                      {(result.insomnia_probability * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
                <p className="text-lg">
                  Risk Level: <span className="font-bold" style={{ color: getRiskColor(result.insomnia_probability) }}>
                    {result.risk_level}
                  </span>
                </p>
              </div>
            </>
          ) : (
            <div className="bg-white rounded-2xl shadow-lg p-12 text-center">
              <div className="text-6xl mb-4">🔬</div>
              <h3 className="text-xl font-bold text-gray-700 mb-2">Complete the form</h3>
              <p className="text-gray-500">Fill out the form and click Get Prediction to see your risk assessment</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
