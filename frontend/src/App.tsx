import { useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './pages/Layout';
import HomePage from './pages/HomePage';
import ConfigurePage from './pages/ConfigurePage';
import ResultsPage from './pages/ResultsPage';
import type { DatasetInfo, AnalyzeResponse } from './services/api';

function App() {
  const [sessionId, setSessionId] = useState('');
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [target, setTarget] = useState<string | null>(null);

  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout sessionId={sessionId} hasResults={!!result} />}>
          {/* Home — always shows 3D robot + upload */}
          <Route
            path="/"
            element={
              <HomePage
                onSessionCreated={(sid, info) => {
                  setSessionId(sid);
                  setDatasetInfo(info);
                  setResult(null);
                }}
              />
            }
          />

          {/* Configure — requires uploaded data */}
          <Route
            path="/configure"
            element={
              sessionId && datasetInfo ? (
                <ConfigurePage
                  sessionId={sessionId}
                  datasetInfo={datasetInfo}
                  onAnalysisComplete={(res, tgt) => {
                    setResult(res);
                    setTarget(tgt);
                  }}
                />
              ) : (
                <Navigate to="/" replace />
              )
            }
          />

          {/* Results — requires completed analysis */}
          <Route
            path="/results"
            element={
              result && datasetInfo ? (
                <ResultsPage
                  result={result}
                  datasetInfo={datasetInfo}
                  sessionId={sessionId}
                  target={target}
                />
              ) : (
                <Navigate to="/" replace />
              )
            }
          />

          {/* Catch-all */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
