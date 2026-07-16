export default function SystemHealthPage() {
  return (
    <div>
      <h1>System health</h1>
      <p>Trends and drift detection across historical runs.</p>

      <div className="card">
        <div className="empty-state">
          Not wired up yet — this page will chart latency percentiles, hallucination-rate
          trends, and a drift flag once persistent metrics storage lands (Checkpoint 3) and
          the observability dashboard is built on top of it (Checkpoint 4).
        </div>
      </div>
    </div>
  );
}
