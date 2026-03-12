import React from "react";
import { getRiskLevel, formatProbability } from "../../utils/formatters";

export default function RiskGauge({ probability }) {
  const risk = getRiskLevel(probability);
  const percentage = probability * 100;

  return (
    <div className="card-premium p-8 text-center">
      <h3 className="text-xl font-bold mb-6">Risk Analysis</h3>
      <div className="relative w-64 h-64 mx-auto mb-6">
        <svg className="w-full h-full transform -rotate-90">
          <defs>
            <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#22c55e" />
              <stop offset="50%" stopColor="#f59e0b" />
              <stop offset="100%" stopColor="#ef4444" />
            </linearGradient>
          </defs>
          <circle cx="128" cy="128" r="112" fill="none" stroke="#e2e8f0" strokeWidth="16" />
          <circle
            cx="128" cy="128" r="112" fill="none"
            stroke="url(#gradient)" strokeWidth="16" strokeLinecap="round"
            strokeDasharray={2 * Math.PI * 112}
            strokeDashoffset={2 * Math.PI * 112 * (1 - probability)}
            style={{ transition: "stroke-dashoffset 1.5s ease-out" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-5xl font-bold" style={{ color: risk.color }}>{percentage.toFixed(1)}%</span>
          <span className="text-2xl mt-2">{risk.emoji} {risk.level}</span>
        </div>
      </div>
      <div className="flex justify-between text-sm text-gray-500">
        <span>Low Risk</span>
        <span>Medium</span>
        <span>High Risk</span>
      </div>
    </div>
  );
}
