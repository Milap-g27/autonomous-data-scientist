const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000/api";

// ── Types ──

export interface DatasetInfo {
  rows: number;
  columns: number;
  numeric_columns: string[];
  categorical_columns: string[];
  column_names: string[];
  missing_values: number;
  preview: Record<string, unknown>[];
}

export interface UploadResponse {
  session_id: string;
  dataset_info: DatasetInfo;
  message: string;
}

export interface ConfigureRequest {
  session_id: string;
  target: string | null;
  problem_hint: string;
  random_seed: number;
  test_size: number;
  scaling: boolean;
  feature_selection: boolean;
}

export interface FigureData {
  heading: string;
  description: string;
  image_base64: string;
}

export interface AnalyzeResponse {
  session_id: string;
  problem_type: string;
  best_model: string;
  metrics: Record<string, Record<string, number>>;
  eda_results: Record<string, unknown>;
  eda_figures: FigureData[];
  explanation: string;
  cleaning_report: string;
  feature_report: string;
  training_time: number;
  message: string;
}

export interface PredictResponse {
  predicted_value: unknown;
  model_used: string;
}

export interface ChatResponse {
  reply: string;
  image_base64?: string;
}

// ── API Client ──

export async function uploadDataset(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  return res.json();
}

export async function configureSession(cfg: ConfigureRequest): Promise<unknown> {
  const res = await fetch(`${API_BASE}/configure`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cfg),
  });
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  return res.json();
}

export async function analyzeSession(sessionId: string): Promise<AnalyzeResponse> {
  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  return res.json();
}

export async function predictWithModel(
  sessionId: string,
  inputData: Record<string, unknown>
): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, input_data: inputData }),
  });
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  return res.json();
}

export async function chatWithAssistant(
  sessionId: string,
  message: string
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  return res.json();
}
