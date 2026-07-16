import { Route, Routes } from 'react-router-dom';
import './App.css';
import Nav from './components/Nav';
import ComparisonPage from './pages/ComparisonPage';
import ResultsPage from './pages/ResultsPage';
import RunConfigPage from './pages/RunConfigPage';
import SystemHealthPage from './pages/SystemHealthPage';

function App() {
  return (
    <div className="app-shell">
      <Nav />
      <main className="app-main">
        <Routes>
          <Route path="/" element={<RunConfigPage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/compare" element={<ComparisonPage />} />
          <Route path="/health" element={<SystemHealthPage />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
