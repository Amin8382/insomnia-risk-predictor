export const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";
export const RISK_LEVELS = {
  LOW: { min: 0, max: 0.3, color: "#22c55e", label: "Low Risk", emoji: "??" },
  MEDIUM: { min: 0.3, max: 0.6, color: "#f59e0b", label: "Medium Risk", emoji: "??" },
  HIGH: { min: 0.6, max: 1, color: "#ef4444", label: "High Risk", emoji: "??" }
};
export const ETHNICITY_OPTIONS = [
  { value: "Caucasian", label: "Caucasian" },
  { value: "African American", label: "African American" },
  { value: "Hispanic", label: "Hispanic" },
  { value: "Asian", label: "Asian" },
  { value: "Other", label: "Other" }
];
export const SMOKING_STATUS_OPTIONS = [
  { value: "Never", label: "Never" },
  { value: "Past", label: "Past" },
  { value: "Current", label: "Current" }
];
export const MEDICAL_CONDITIONS = [
  { id: "hypertension", label: "Hypertension", icon: "??" },
  { id: "diabetes", label: "Diabetes", icon: "??" },
  { id: "asthma", label: "Asthma", icon: "??" },
  { id: "copd", label: "COPD", icon: "??" },
  { id: "cancer", label: "Cancer", icon: "???" },
  { id: "obesity", label: "Obesity", icon: "??" }
];
