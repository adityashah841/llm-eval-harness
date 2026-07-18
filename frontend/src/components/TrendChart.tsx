import { useRef, useState } from 'react';

export interface TrendSeries {
  id: string;
  label: string;
  color: string;
  points: { x: number; y: number }[];
}

interface ReferenceLine {
  y: number;
  label: string;
  color: string;
}

interface TrendChartProps {
  series: TrendSeries[];
  height?: number;
  yFormat?: (v: number) => string;
  xFormat?: (v: number) => string;
  referenceLine?: ReferenceLine;
  yMin?: number;
  yMax?: number;
}

const WIDTH = 640;
const BASE_PAD = { top: 16, right: 16, bottom: 28, left: 44 };

const defaultYFormat = (v: number) => v.toFixed(2);
const defaultXFormat = (v: number) => new Date(v).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });

export default function TrendChart({
  series,
  height = 220,
  yFormat = defaultYFormat,
  xFormat = defaultXFormat,
  referenceLine,
  yMin,
  yMax,
}: TrendChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);

  const allPoints = series.flatMap((s) => s.points);
  const xs = allPoints.map((p) => p.x);
  const ys = allPoints.map((p) => p.y).concat(referenceLine ? [referenceLine.y] : []);

  const xDomain: [number, number] = xs.length ? [Math.min(...xs), Math.max(...xs)] : [0, 1];
  const yDomainRaw: [number, number] = ys.length ? [Math.min(...ys), Math.max(...ys)] : [0, 1];
  const yPad = (yDomainRaw[1] - yDomainRaw[0]) * 0.15 || Math.abs(yDomainRaw[1] || 1) * 0.1;
  const yDomain: [number, number] = [
    yMin ?? Math.max(0, yDomainRaw[0] - yPad),
    yMax ?? (yDomainRaw[1] + yPad || 1),
  ];

  const tickCount = 4;
  const yTicks = Array.from(
    { length: tickCount + 1 },
    (_, i) => yDomain[0] + ((yDomain[1] - yDomain[0]) * i) / tickCount
  );
  const longestTickLabel = Math.max(...yTicks.map((t) => yFormat(t).length));

  const innerH = height - BASE_PAD.top - BASE_PAD.bottom;
  const yScale = (y: number) =>
    yDomain[1] === yDomain[0]
      ? BASE_PAD.top + innerH / 2
      : BASE_PAD.top + innerH - ((y - yDomain[0]) / (yDomain[1] - yDomain[0])) * innerH;

  const lastPoints = series.map((s) => s.points.slice().sort((a, b) => a.x - b.x).at(-1));
  const lastYPixels = lastPoints.filter((p) => p).map((p) => yScale(p!.y)).sort((a, b) => a - b);
  const labelsCollide = lastYPixels.some((y, i) => i > 0 && y - lastYPixels[i - 1] < 14);
  // Converging end-labels read as noise (see marks-and-anatomy.md) — fall back to
  // the legend + tooltip instead of stacking overlapping text.
  const showEndLabels = series.length <= 4 && !labelsCollide;
  const longestEndLabel = showEndLabels ? Math.max(...series.map((s) => s.label.length)) : 0;

  const PAD = {
    ...BASE_PAD,
    left: Math.max(BASE_PAD.left, longestTickLabel * 6 + 16),
    right: Math.max(BASE_PAD.right, longestEndLabel * 6 + 16),
  };

  const innerW = WIDTH - PAD.left - PAD.right;

  const xScale = (x: number) =>
    xDomain[1] === xDomain[0] ? PAD.left + innerW / 2 : PAD.left + ((x - xDomain[0]) / (xDomain[1] - xDomain[0])) * innerW;

  let nearestX: number | null = null;
  if (hoverX !== null && xs.length > 0) {
    const dataX = xDomain[0] + ((hoverX - PAD.left) / innerW) * (xDomain[1] - xDomain[0]);
    nearestX = xs.reduce((best, x) => (Math.abs(x - dataX) < Math.abs(best - dataX) ? x : best), xs[0]);
  }

  function handleMove(e: React.PointerEvent<SVGSVGElement>) {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const scaleX = WIDTH / rect.width;
    const localX = (e.clientX - rect.left) * scaleX;
    setHoverX(Math.min(Math.max(localX, PAD.left), PAD.left + innerW));
  }

  const hoverRows = nearestX === null
    ? []
    : series
        .map((s) => ({
          ...s,
          point: s.points.find((p) => p.x === nearestX) ?? s.points.reduce((closest, p) =>
            Math.abs(p.x - nearestX) < Math.abs(closest.x - nearestX) ? p : closest, s.points[0]),
        }))
        .filter((s) => s.point);

  const hasData = allPoints.length > 0;
  const showLegend = series.length >= 2;

  return (
    <div className="trend-chart">
      {!hasData ? (
        <div className="empty-state">No data points in range.</div>
      ) : (
        <>
          <svg
            ref={svgRef}
            viewBox={`0 0 ${WIDTH} ${height}`}
            className="trend-chart-svg"
            onPointerMove={handleMove}
            onPointerLeave={() => setHoverX(null)}
            role="img"
          >
            {yTicks.map((t, i) => (
              <g key={i}>
                <line x1={PAD.left} x2={WIDTH - PAD.right} y1={yScale(t)} y2={yScale(t)} className="chart-gridline" />
                <text x={PAD.left - 8} y={yScale(t)} className="chart-axis-label" textAnchor="end" dominantBaseline="middle">
                  {yFormat(t)}
                </text>
              </g>
            ))}

            {referenceLine && (
              <>
                <line
                  x1={PAD.left}
                  x2={WIDTH - PAD.right}
                  y1={yScale(referenceLine.y)}
                  y2={yScale(referenceLine.y)}
                  className="chart-threshold-line"
                  stroke={referenceLine.color}
                />
                <text x={WIDTH - PAD.right} y={yScale(referenceLine.y) - 4} textAnchor="end" className="chart-axis-label">
                  {referenceLine.label}
                </text>
              </>
            )}

            {xDomain[0] !== xDomain[1] && (
              <>
                <text x={PAD.left} y={height - 8} className="chart-axis-label" textAnchor="start">
                  {xFormat(xDomain[0])}
                </text>
                <text x={WIDTH - PAD.right} y={height - 8} className="chart-axis-label" textAnchor="end">
                  {xFormat(xDomain[1])}
                </text>
              </>
            )}

            {series.map((s) => {
              const path = s.points
                .slice()
                .sort((a, b) => a.x - b.x)
                .map((p, i) => `${i === 0 ? 'M' : 'L'}${xScale(p.x)},${yScale(p.y)}`)
                .join(' ');
              const last = s.points.slice().sort((a, b) => a.x - b.x).at(-1);
              return (
                <g key={s.id}>
                  <path d={path} fill="none" stroke={s.color} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
                  {last && (
                    <circle
                      cx={xScale(last.x)}
                      cy={yScale(last.y)}
                      r={4}
                      fill={s.color}
                      stroke="var(--surface-1)"
                      strokeWidth={2}
                    />
                  )}
                  {last && showEndLabels && (
                    <text
                      x={xScale(last.x) + 8}
                      y={yScale(last.y)}
                      className="chart-end-label"
                      dominantBaseline="middle"
                    >
                      {s.label}
                    </text>
                  )}
                </g>
              );
            })}

            {nearestX !== null && (
              <line
                x1={xScale(nearestX)}
                x2={xScale(nearestX)}
                y1={PAD.top}
                y2={PAD.top + innerH}
                className="chart-crosshair"
              />
            )}
          </svg>

          {showLegend && (
            <div className="chart-legend">
              {series.map((s) => (
                <span key={s.id} className="chart-legend-item">
                  <span className="chart-legend-key" style={{ background: s.color }} />
                  {s.label}
                </span>
              ))}
            </div>
          )}

          {nearestX !== null && hoverRows.length > 0 && (
            <div
              className="chart-tooltip"
              style={{ left: `${((xScale(nearestX) / WIDTH) * 100).toFixed(2)}%` }}
            >
              <div className="chart-tooltip-x">{xFormat(nearestX)}</div>
              {hoverRows.map((row) => (
                <div className="chart-tooltip-row" key={row.id}>
                  <span className="chart-legend-key" style={{ background: row.color }} />
                  <span className="chart-tooltip-value">{yFormat(row.point.y)}</span>
                  <span className="chart-tooltip-label">{row.label}</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
