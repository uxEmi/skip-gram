import { useState, useEffect, useRef, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";

const API = "http://localhost:8000";

export default function App() {
  const [word, setWord] = useState("king");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loss, setLoss] = useState(null);
  const [health, setHealth] = useState(null);
  const [comparisons, setComparisons] = useState(null);
  const [embedding3d, setEmbedding3d] = useState(null);

  useEffect(() => {
    fetch(`${API}/loss`).then((r) => r.json()).then(setLoss)
      .catch(() => setLoss({ error: true }));
    fetch(`${API}/health`).then((r) => r.json()).then(setHealth)
      .catch(() => setHealth({ error: true }));
    fetch(`${API}/comparisons`).then((r) => r.json()).then(setComparisons)
      .catch(() => setComparisons({ error: true }));
    fetch(`${API}/embedding_3d`).then((r) => r.json()).then(setEmbedding3d)
      .catch(() => setEmbedding3d({ error: true }));
  }, []);


  useEffect(() => { search();  }, []);

  async function search(e) {
    e?.preventDefault();
    if (!word.trim()) return;
    setLoading(true);
    try {
      const r = await fetch(`${API}/neighbours?word=${encodeURIComponent(word)}`);
      setResult(await r.json());
    } catch { setResult({ error: true }); }
    setLoading(false);
  }

  const epochs = loss?.loss_history?.length ?? null;
  const maxScore = result?.neighbours?.[0]?.score ?? 1;

  return (
    <div style={S.app}>
      <GlobalStyles />
      <div style={S.bgOrb1} aria-hidden />
      <div style={S.bgOrb2} aria-hidden />

      <main style={S.page}>
        <header style={S.hero}>
          <h1 style={S.h1}>
            Word<span style={S.h1Accent}>2</span>Vec
          </h1>

          <div style={S.statsRow}>
            <Stat label="vocabulary" value={health?.vocab_size ?? "—"} healthy={!!health && !health.error} />
            <Stat label="dimensions" value={health?.dim ?? "—"} healthy={!!health && !health.error} />
            <Stat label="epochs"     value={epochs ?? "—"}        healthy={!!loss && !loss.error} />
          </div>
        </header>

        <section style={S.card}>
          <div style={S.cardHead}>
            <div style={S.badge}>01</div>
            <div>
              <h2 style={S.h2}>Nearest neighbours</h2>
            </div>
          </div>

          <form onSubmit={search} style={S.searchRow}>
            <div style={S.searchWrap}>
              <SearchIcon />
              <input
                style={S.input}
                className="w2v-input"
                value={word}
                onChange={(e) => setWord(e.target.value)}
                placeholder="try a word — king, river, france…"
                spellCheck={false}
                autoComplete="off"
              />
            </div>
            <button style={S.btn} disabled={loading} className="w2v-btn">
              {loading ? <Spinner /> : <>find <Arrow /></>}
            </button>
          </form>

          <div style={S.resultBox}>
            {result?.error && <Message kind="err">Server not reachable — is uvicorn running on :8000?</Message>}
            {result && !result.error && !result.found &&
              <Message kind="muted">"{result.word}" is not in the vocabulary.</Message>}
            {result?.found && (
              <ol style={S.list}>
                {result.neighbours.map((n, i) => (
                  <li key={n.token} style={{ ...S.li, animationDelay: `${i * 30}ms` }} className="w2v-row">
                    <span style={S.rank}>{String(i + 1).padStart(2, "0")}</span>
                    <span style={S.tok}>{n.token}</span>
                    <div style={S.barTrack}>
                      <div style={{
                        ...S.barFill,
                        width: `${Math.max(2, (n.score / maxScore) * 100)}%`,
                      }} />
                    </div>
                    <span style={S.scoreNum}>{n.score.toFixed(3)}</span>
                  </li>
                ))}
              </ol>
            )}
          </div>
        </section>

        <section style={S.card}>
          {!loss && <Message kind="muted">loading training trace…</Message>}
          {loss?.error && <Message kind="err">Server not reachable — is uvicorn running on :8000?</Message>}
          {loss && !loss.error && (
            <>
              <div style={S.lossRow}>
                <LossStat label="start" value={loss.start} tone="dim" />
                <div style={S.lossArrow}>→</div>
                <LossStat label="end" value={loss.end} tone="bright" />
                <div style={S.lossDelta}>
                  {loss.start != null && loss.end != null
                    ? `−${((1 - loss.end / loss.start) * 100).toFixed(1)}%`
                    : ""}
                </div>
              </div>
              <LossCurve data={loss.loss_history} />
            </>
          )}
        </section>

        <section style={S.card}>
          <h2 style={S.h2Plain}>Learning rate sweep</h2>
          {!comparisons && <Message kind="muted">loading comparisons…</Message>}
          {comparisons?.error && (
            <Message kind="err">
              Run <code style={S.code}>python generate_comparisons.py</code> in <code style={S.code}>backend/</code>, then restart uvicorn.
            </Message>
          )}
          {comparisons && !comparisons.error && (
            <MultiLossCurve
              curves={comparisons.lr.curves}
              labels={comparisons.lr.values.map((v) => `LR = ${v}`)}
              keys={comparisons.lr.values.map(String)}
              logY
            />
          )}
        </section>

        <section style={S.card}>
          <h2 style={S.h2Plain}>Embedding space</h2>
          {!embedding3d && <Message kind="muted">loading embedding scatter…</Message>}
          {embedding3d?.error && <Message kind="err">Server not reachable — is uvicorn running on :8000?</Message>}
          {embedding3d?.points && <EmbeddingScatter points={embedding3d.points} />}
        </section>
      </main>
    </div>
  );
}


function Stat({ label, value, healthy }) {
  return (
    <div style={S.stat}>
      <div style={{ ...S.statValue, color: healthy ? "#1a1a2e" : "#bbb" }}>{value}</div>
      <div style={S.statLabel}>{label}</div>
    </div>
  );
}

function LossStat({ label, value, tone }) {
  return (
    <div style={S.lossStat}>
      <div style={S.lossStatLabel}>{label}</div>
      <div style={{
        ...S.lossStatValue,
        color: tone === "bright" ? "#534AB7" : "#7a7a90",
      }}>
        {value != null ? value.toFixed(3) : "—"}
      </div>
    </div>
  );
}

function Message({ kind, children }) {
  const tone = kind === "err"
    ? { color: "#A32D2D", background: "#FBECEC", borderColor: "#F2C9C9" }
    : { color: "#6b6b80", background: "#F5F4FB", borderColor: "#E5E2F2" };
  return (
    <div style={{ ...S.msg, ...tone }}>
      {kind === "err" ? <WarnIcon /> : <InfoIcon />}
      <span>{children}</span>
    </div>
  );
}

function LossCurve({ data }) {
  const ref = useRef(null);
  if (!data || data.length < 2) return null;
  const W = 640, H = 240, pad = { l: 44, r: 20, t: 20, b: 32 };
  const lo = Math.min(...data), hi = Math.max(...data);
  const range = hi - lo || 1;
  const x = (i) => pad.l + (i / (data.length - 1)) * (W - pad.l - pad.r);
  const y = (v) => pad.t + (1 - (v - lo) / range) * (H - pad.t - pad.b);

  const linePath = data.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(2)},${y(v).toFixed(2)}`).join(" ");
  const areaPath = `${linePath} L${x(data.length - 1).toFixed(2)},${H - pad.b} L${x(0).toFixed(2)},${H - pad.b} Z`;

  const gridY = [0, 0.25, 0.5, 0.75, 1].map((t) => lo + (1 - t) * range);
  const last = data[data.length - 1];

  return (
    <div style={S.chartWrap}>
      <svg ref={ref} viewBox={`0 0 ${W} ${H}`} width="100%" style={S.chart} role="img" aria-label="loss curve">
        <defs>
          <linearGradient id="lossArea" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"  stopColor="#534AB7" stopOpacity="0.32" />
            <stop offset="100%" stopColor="#534AB7" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="lossLine" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%"  stopColor="#8A82E8" />
            <stop offset="100%" stopColor="#534AB7" />
          </linearGradient>
        </defs>

        {gridY.map((v, i) => (
          <g key={i}>
            <line
              x1={pad.l} x2={W - pad.r}
              y1={y(v)}  y2={y(v)}
              stroke="#ECEAF6" strokeWidth="1"
              strokeDasharray={i === gridY.length - 1 ? "0" : "3 4"}
            />
            <text x={pad.l - 8} y={y(v) + 3} fontSize="10" fill="#9a9ab0" textAnchor="end" fontVariantNumeric="tabular-nums">
              {v.toFixed(2)}
            </text>
          </g>
        ))}

        <path d={areaPath} fill="url(#lossArea)" />
        <path
          d={linePath}
          fill="none"
          stroke="url(#lossLine)"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="w2v-loss-path"
        />

        <circle cx={x(data.length - 1)} cy={y(last)} r="5" fill="#fff" stroke="#534AB7" strokeWidth="2.5" />

        <text x={pad.l} y={H - 10} fontSize="11" fill="#9a9ab0">epoch 1</text>
        <text x={W - pad.r} y={H - 10} fontSize="11" fill="#9a9ab0" textAnchor="end">
          epoch {data.length}
        </text>
      </svg>
    </div>
  );
}

function MultiLossCurve({ curves, labels, keys, logY = false }) {
  const series = keys.map((k) => curves[k]);
  if (!series.length || !series[0]?.length) return null;

  const W = 640, H = 260, pad = { l: 50, r: 16, t: 16, b: 36 };
  const epochs = series[0].length;
  const transform = logY ? Math.log10 : (v) => v;

  let lo = Infinity, hi = -Infinity;
  for (const s of series) for (const v of s) {
    const t = transform(Math.max(v, 1e-6));
    if (t < lo) lo = t;
    if (t > hi) hi = t;
  }
  const range = hi - lo || 1;
  const x = (i) => pad.l + (i / (epochs - 1)) * (W - pad.l - pad.r);
  const y = (v) => pad.t + (1 - (transform(Math.max(v, 1e-6)) - lo) / range) * (H - pad.t - pad.b);

  const COLORS = ["#534AB7", "#E08A2A", "#2C8F5A", "#C44A6E"];
  const gridY = [0, 0.25, 0.5, 0.75, 1].map((t) => lo + (1 - t) * range);
  const formatTick = (lv) => logY ? Math.pow(10, lv).toPrecision(2) : lv.toFixed(2);

  return (
    <div style={S.chartWrap}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={S.chart}>
        {gridY.map((lv, i) => (
          <g key={i}>
            <line x1={pad.l} x2={W - pad.r} y1={y(logY ? Math.pow(10, lv) : lv)} y2={y(logY ? Math.pow(10, lv) : lv)}
                  stroke="#ECEAF6" strokeWidth="1"
                  strokeDasharray={i === gridY.length - 1 ? "0" : "3 4"} />
            <text x={pad.l - 8} y={y(logY ? Math.pow(10, lv) : lv) + 3} fontSize="10"
                  fill="#9a9ab0" textAnchor="end" fontVariantNumeric="tabular-nums">
              {formatTick(lv)}
            </text>
          </g>
        ))}

        {series.map((data, idx) => {
          const path = data.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(2)},${y(v).toFixed(2)}`).join(" ");
          return (
            <path key={idx} d={path} fill="none" stroke={COLORS[idx % COLORS.length]}
                  strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" opacity="0.95" />
          );
        })}

        <text x={pad.l} y={H - 12} fontSize="11" fill="#9a9ab0">epoch 1</text>
        <text x={W - pad.r} y={H - 12} fontSize="11" fill="#9a9ab0" textAnchor="end">
          epoch {epochs}
        </text>
        {logY && (
          <text x={pad.l - 38} y={pad.t + 8} fontSize="10" fill="#9a9ab0" transform={`rotate(-90, ${pad.l - 38}, ${pad.t + 8})`}>
            loss (log)
          </text>
        )}
      </svg>

      <div style={S.legend}>
        {labels.map((lab, idx) => (
          <div key={lab} style={S.legendItem}>
            <span style={{ ...S.legendSwatch, background: COLORS[idx % COLORS.length] }} />
            <span style={S.legendLabel}>{lab}</span>
            <span style={S.legendEnd}>
              {series[idx][series[idx].length - 1].toFixed(2)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function EmbeddingScatter({ points }) {
  const ENGLISH = useMemo(
    () => points.filter((p) => /^[a-z]{2,}$/i.test(p.token)),
    [points]
  );

  return (
    <div style={S.canvasWrap}>
      <Canvas camera={{ position: [2.2, 1.8, 2.2], fov: 55 }} dpr={[1, 2]}>
        <color attach="background" args={["#FBFAFE"]} />
        <ambientLight intensity={0.7} />
        <pointLight position={[5, 5, 5]} intensity={0.8} />

        <gridHelper args={[3, 12, "#E5E2F2", "#F1EFF9"]} position={[0, -1.05, 0]} />
        <axesHelper args={[0.6]} />

        <OrbitControls enableDamping makeDefault />

        {ENGLISH.map((p) => (
          <mesh key={`pt-${p.token}`} position={[p.x, p.y, p.z]}>
            <sphereGeometry args={[0.018, 12, 12]} />
            <meshStandardMaterial color="#534AB7" />
          </mesh>
        ))}

        {ENGLISH.map((p) => (
          <Html
            key={`lab-${p.token}`}
            position={[p.x, p.y, p.z]}
            distanceFactor={4}
            style={{ pointerEvents: "none" }}
            zIndexRange={[10, 0]}
          >
            <span style={S.tokenLabel}>{p.token}</span>
          </Html>
        ))}
      </Canvas>
    </div>
  );
}


const SearchIcon = () => (
  <svg viewBox="0 0 24 24" width="18" height="18" style={S.searchIcon} aria-hidden>
    <circle cx="11" cy="11" r="6.5" fill="none" stroke="#9a9ab0" strokeWidth="1.8" />
    <line x1="16" y1="16" x2="20.5" y2="20.5" stroke="#9a9ab0" strokeWidth="1.8" strokeLinecap="round" />
  </svg>
);
const Arrow = () => (
  <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden style={{ marginLeft: 4 }}>
    <path d="M5 12h13M13 6l6 6-6 6" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);
const Spinner = () => (
  <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden className="w2v-spin">
    <circle cx="12" cy="12" r="9" fill="none" stroke="#fff" strokeOpacity="0.3" strokeWidth="2.5" />
    <path d="M21 12a9 9 0 0 0-9-9" fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" />
  </svg>
);
const WarnIcon = () => (
  <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden>
    <path d="M12 3 22 21H2L12 3z" fill="none" stroke="#A32D2D" strokeWidth="1.8" strokeLinejoin="round" />
    <line x1="12" y1="10" x2="12" y2="14" stroke="#A32D2D" strokeWidth="1.8" strokeLinecap="round" />
    <circle cx="12" cy="17.5" r="1" fill="#A32D2D" />
  </svg>
);
const InfoIcon = () => (
  <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden>
    <circle cx="12" cy="12" r="9" fill="none" stroke="#7a7a90" strokeWidth="1.8" />
    <line x1="12" y1="11" x2="12" y2="16" stroke="#7a7a90" strokeWidth="1.8" strokeLinecap="round" />
    <circle cx="12" cy="8" r="1" fill="#7a7a90" />
  </svg>
);

const GlobalStyles = () => (
  <style>{`
    html, body, #root { margin: 0; padding: 0; min-height: 100%; }
    body {
      background: #FAF9FD;
      font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", system-ui, sans-serif;
      -webkit-font-smoothing: antialiased;
      color: #1a1a2e;
    }
    *, *::before, *::after { box-sizing: border-box; }

    .w2v-btn { transition: transform .12s ease, box-shadow .18s ease, filter .18s ease; }
    .w2v-btn:hover:not(:disabled) { transform: translateY(-1px); filter: brightness(1.06);
      box-shadow: 0 10px 24px -10px rgba(83,74,183,0.55); }
    .w2v-btn:active:not(:disabled) { transform: translateY(0); }
    .w2v-btn:disabled { opacity: 0.7; cursor: progress; }

    .w2v-input:focus { outline: none; border-color: #534AB7;
      box-shadow: 0 0 0 4px rgba(83,74,183,0.12); }

    .w2v-spin { animation: w2v-rotate 0.8s linear infinite; }
    @keyframes w2v-rotate { to { transform: rotate(360deg); } }

    .w2v-loss-path {
      stroke-dasharray: 2000; stroke-dashoffset: 2000;
      animation: w2v-draw 1.4s ease-out 0.1s forwards;
    }
    @keyframes w2v-draw { to { stroke-dashoffset: 0; } }

    .w2v-row { animation: w2v-rise .35s ease-out both; }
    @keyframes w2v-rise { from { opacity: 0; transform: translateY(4px);} to { opacity: 1; transform: none; } }
  `}</style>
);


const S = {
  app: {
    position: "relative", minHeight: "100vh", overflow: "hidden",
    background:
      "radial-gradient(1200px 600px at 10% -10%, #EFEBFD 0%, transparent 60%)," +
      "radial-gradient(800px 500px at 100% 10%, #E6F4FB 0%, transparent 55%)," +
      "linear-gradient(180deg, #FAF9FD 0%, #FAF9FD 100%)",
  },
  bgOrb1: {
    position: "absolute", top: -120, left: -120, width: 360, height: 360,
    background: "radial-gradient(circle, rgba(175,169,236,0.35), transparent 70%)",
    filter: "blur(20px)", pointerEvents: "none", zIndex: 0,
  },
  bgOrb2: {
    position: "absolute", top: 120, right: -160, width: 420, height: 420,
    background: "radial-gradient(circle, rgba(83,74,183,0.18), transparent 70%)",
    filter: "blur(30px)", pointerEvents: "none", zIndex: 0,
  },

  page: {
    position: "relative", zIndex: 1,
    maxWidth: 760, margin: "0 auto", padding: "56px 24px 64px",
  },

  hero: { marginBottom: 36 },
  h1: {
    fontSize: 56, fontWeight: 700, letterSpacing: "-0.03em",
    margin: "0 0 24px", lineHeight: 1.05, color: "#15152b",
  },
  h1Accent: {
    background: "linear-gradient(135deg, #8A82E8 0%, #534AB7 60%, #3a318e 100%)",
    WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
    backgroundClip: "text", color: "transparent",
  },

  statsRow: {
    display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12,
    padding: 4,
  },
  stat: {
    background: "rgba(255,255,255,0.7)", backdropFilter: "blur(6px)",
    border: "1px solid #ECEAF6", borderRadius: 14,
    padding: "14px 16px",
  },
  statValue: { fontSize: 24, fontWeight: 700, letterSpacing: "-0.02em",
    fontVariantNumeric: "tabular-nums" },
  statLabel: { fontSize: 11, color: "#8a8aa0", textTransform: "uppercase",
    letterSpacing: "0.08em", marginTop: 2, fontWeight: 600 },

  card: {
    background: "#fff", borderRadius: 20,
    padding: "26px 26px 24px", marginBottom: 20,
    border: "1px solid #ECEAF6",
    boxShadow: "0 1px 0 rgba(20,20,40,0.02), 0 24px 48px -28px rgba(60,50,140,0.18)",
  },
  cardHead: { display: "flex", gap: 14, alignItems: "flex-start", marginBottom: 18 },
  badge: {
    flexShrink: 0, width: 36, height: 36, borderRadius: 10,
    display: "grid", placeItems: "center",
    fontSize: 12, fontWeight: 700, color: "#fff",
    background: "linear-gradient(135deg, #8A82E8 0%, #534AB7 100%)",
    boxShadow: "0 8px 20px -8px rgba(83,74,183,0.55)",
    letterSpacing: "0.04em",
  },
  h2: { fontSize: 18, fontWeight: 700, margin: 0, color: "#15152b",
    letterSpacing: "-0.01em" },

  searchRow: { display: "flex", gap: 10, marginBottom: 8 },
  searchWrap: { position: "relative", flex: 1 },
  searchIcon: { position: "absolute", left: 14, top: "50%",
    transform: "translateY(-50%)", pointerEvents: "none" },
  input: {
    width: "100%", padding: "13px 14px 13px 42px",
    border: "1.5px solid #E5E2F2", borderRadius: 12,
    fontSize: 15, fontFamily: "inherit", color: "#15152b",
    background: "#FBFAFE",
    transition: "border-color .15s ease, box-shadow .15s ease",
  },
  btn: {
    padding: "0 22px", height: 48, border: "none", borderRadius: 12,
    background: "linear-gradient(135deg, #6258d8 0%, #534AB7 100%)",
    color: "#fff", fontSize: 15, fontWeight: 600, cursor: "pointer",
    display: "inline-flex", alignItems: "center", gap: 2,
    boxShadow: "0 6px 16px -8px rgba(83,74,183,0.7)",
    fontFamily: "inherit",
  },

  resultBox: { marginTop: 14 },
  list: { listStyle: "none", padding: 0, margin: 0 },
  li: {
    display: "grid",
    gridTemplateColumns: "32px 120px 1fr 56px",
    alignItems: "center", gap: 14,
    padding: "10px 0",
    borderBottom: "1px solid #F4F2FA",
  },
  rank: { fontSize: 11, color: "#bcbcd0", fontWeight: 700,
    fontVariantNumeric: "tabular-nums", letterSpacing: "0.04em" },
  tok: { fontSize: 15, color: "#15152b", fontWeight: 500,
    fontFamily: '"SF Mono", "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace' },
  barTrack: { height: 8, background: "#F1EFF9", borderRadius: 999, overflow: "hidden" },
  barFill: {
    height: "100%", borderRadius: 999,
    background: "linear-gradient(90deg, #AFA9EC 0%, #6258d8 50%, #534AB7 100%)",
    transition: "width .5s cubic-bezier(.2,.7,.2,1)",
  },
  scoreNum: { fontSize: 13, color: "#5a5a72", textAlign: "right",
    fontVariantNumeric: "tabular-nums", fontWeight: 500 },

  lossRow: { display: "flex", alignItems: "center", gap: 14,
    margin: "0 0 18px", flexWrap: "wrap" },
  lossStat: { display: "flex", flexDirection: "column", gap: 2 },
  lossStatLabel: { fontSize: 11, color: "#9a9ab0", textTransform: "uppercase",
    letterSpacing: "0.08em", fontWeight: 600 },
  lossStatValue: { fontSize: 22, fontWeight: 700,
    fontVariantNumeric: "tabular-nums", letterSpacing: "-0.02em" },
  lossArrow: { color: "#cdc9e3", fontSize: 22, fontWeight: 300 },
  lossDelta: {
    marginLeft: "auto",
    fontSize: 13, fontWeight: 600,
    color: "#2C8F5A", background: "#E8F6EE",
    padding: "5px 10px", borderRadius: 999,
    letterSpacing: "0.01em",
  },

  chartWrap: {
    background: "linear-gradient(180deg, #FBFAFE 0%, #F6F4FC 100%)",
    border: "1px solid #ECEAF6", borderRadius: 14,
    padding: 10, marginBottom: 12,
  },
  chart: { display: "block", maxWidth: "100%" },

  msg: {
    display: "flex", alignItems: "center", gap: 10,
    padding: "12px 14px", borderRadius: 10,
    border: "1px solid", fontSize: 13.5,
  },

  h2Plain: { fontSize: 18, fontWeight: 700, margin: "0 0 14px",
    color: "#15152b", letterSpacing: "-0.01em" },

  legend: {
    display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "6px 18px",
    marginTop: 14, padding: "0 4px",
  },
  legendItem: { display: "flex", alignItems: "center", gap: 8, fontSize: 13 },
  legendSwatch: { width: 12, height: 12, borderRadius: 3, flexShrink: 0 },
  legendLabel: { color: "#15152b", fontWeight: 500 },
  legendEnd: { marginLeft: "auto", color: "#7a7a90",
    fontVariantNumeric: "tabular-nums", fontSize: 12.5 },

  canvasWrap: {
    background: "linear-gradient(180deg, #FBFAFE 0%, #F6F4FC 100%)",
    border: "1px solid #ECEAF6", borderRadius: 14,
    height: 480, position: "relative", overflow: "hidden",
  },
  tokenLabel: {
    fontFamily: '"SF Mono", "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace',
    fontSize: 11, color: "#3a318e",
    background: "rgba(255,255,255,0.85)",
    padding: "1px 5px", borderRadius: 4,
    marginLeft: 6, whiteSpace: "nowrap",
    border: "1px solid #E5E2F2",
  },
};

