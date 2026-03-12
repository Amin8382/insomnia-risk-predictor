import React from "react";
import { getRiskLevel, getRiskDescription, formatDate } from "../../utils/formatters";

export default function PredictionCard({ prediction }) {
  const risk = getRiskLevel(prediction.insomnia_probability);
  return (
    <div className="bg-white rounded-2xl shadow-premium p-8">
      <h3 className="text-xl font-semibold mb-4">Results</h3>
      <div className="bg-gray-50 p-6 rounded-xl mb-4">
        <p className="text-gray-700">{getRiskDescription(prediction.insomnia_probability)}</p>
      </div>
      <div className="grid grid-cols-2 gap-4 text-center">
        <div className="p-4 bg-primary-50 rounded-lg">
          <p className="text-sm text-primary-600">Probability</p>
          <p className="text-2xl font-bold text-primary-700">{(prediction.insomnia_probability * 100).toFixed(1)}%</p>
        </div>
        <div className="p-4" style={{ backgroundColor: `${risk.color}20` }} style-border-radius="0.5rem">
          <p className="text-sm" style={{ color: risk.color }}>Risk Level</p>
          <p className="text-2xl font-bold" style={{ color: risk.color }}>{risk.level}</p>
        </div>
      </div>
    </div>
  );
}
