import React from "react";
import { Link } from "react-router-dom";
import { FaMoon, FaBrain, FaChartLine } from "react-icons/fa";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-20">
        <div className="text-center mb-16">
          <h1 className="text-6xl font-bold text-gray-800 mb-6">😴 Insomnia Risk Predictor</h1>
          <p className="text-2xl text-gray-600 mb-8">AI-powered insomnia risk assessment using clinical data</p>
          <Link to="/predict" className="bg-blue-600 text-white px-8 py-4 rounded-xl text-xl hover:bg-blue-700 transition shadow-lg inline-block">
            Start Assessment →
          </Link>
        </div>
        <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          {[
            { icon: <FaMoon className="text-5xl text-blue-600 mb-4" />, title: "31 Clinical Factors", desc: "Comprehensive analysis" },
            { icon: <FaBrain className="text-5xl text-blue-600 mb-4" />, title: "ML-Powered", desc: "Random Forest model" },
            { icon: <FaChartLine className="text-5xl text-blue-600 mb-4" />, title: "Instant Results", desc: "Real-time assessment" }
          ].map((item, i) => (
            <div key={i} className="bg-white/80 backdrop-blur rounded-2xl p-8 text-center shadow-xl">
              {item.icon}
              <h3 className="text-xl font-bold mb-2">{item.title}</h3>
              <p className="text-gray-600">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
