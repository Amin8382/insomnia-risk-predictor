export const formatProbability = (p) => `${(p * 100).toFixed(1)}%`;
export const getRiskLevel = (p) => {
  if (p < 0.3) return { level: "Low", color: "#22c55e", emoji: "??" };
  if (p < 0.6) return { level: "Medium", color: "#f59e0b", emoji: "??" };
  return { level: "High", color: "#ef4444", emoji: "??" };
};
export const getRiskDescription = (p) => {
  if (p < 0.3) return "Your risk is LOW. Maintain healthy sleep habits!";
  if (p < 0.6) return "Your risk is MODERATE. Consider improving sleep hygiene.";
  return "Your risk is HIGH. Please consult a healthcare provider.";
};
