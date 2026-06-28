'use client';

import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { GIFEncoder, quantize, applyPalette } from 'gifenc';

// ── Types ──────────────────────────────────────────────────────────────────

interface SliceData {
  nx: number;
  ny: number;
  nz: number;
  z: number;
  pixels: string; // base64-encoded Uint8Array
  windowMin: number;
  windowMax: number;
}

interface MiData {
  sparkline: number[];
  allJoints: Uint16Array; // decoded client-side; flat [nz * bins * bins]
  bins: number;
  nz: number;
}

type Row = Record<string, string>;
type DataIndex = Record<string, Record<string, Row[]>>;
type Dataset = 'dataset1' | 'dataset2' | 'dataset3';
type SplitKey = 'train_pairs' | 'val_queries' | 'val_gallery' | 'test_queries' | 'test_gallery';

// ── SlicePanel ─────────────────────────────────────────────────────────────

interface SlicePanelProps {
  imagePath: string;
  label: string;
  /** Controlled z; if provided, panel delegates slider to parent */
  z?: number;
  onNzReady?: (nz: number) => void;
  onZChange?: (z: number) => void;
}

function SlicePanel({ imagePath, label, z: zProp, onNzReady, onZChange }: SlicePanelProps) {
  const controlled = zProp !== undefined;
  const [internalZ, setInternalZ] = useState(0);
  const z = controlled ? zProp : internalZ;

  const [sliceData, setSliceData] = useState<SliceData | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const fetchSlice = useCallback(async (p: string, zIdx: number) => {
    const res = await fetch(`/api/slice?path=${encodeURIComponent(p)}&z=${zIdx}`);
    const data = await res.json();
    if (!res.ok || !('pixels' in data)) return null;
    return data as SliceData;
  }, []);

  // On image path change: fetch middle slice to learn nz
  useEffect(() => {
    if (!imagePath) return;
    setSliceData(null);
    fetchSlice(imagePath, -1).then((data) => {
      if (!data) return;
      if (!controlled) setInternalZ(data.z);
      onNzReady?.(data.nz);
      setSliceData(data);
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imagePath]);

  // Fetch when z changes
  useEffect(() => {
    if (!imagePath || !sliceData) return;
    fetchSlice(imagePath, z).then((data) => data && setSliceData(data));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [z]);

  // Render pixels onto canvas
  useEffect(() => {
    if (!sliceData || !canvasRef.current) return;
    const { nx, ny, pixels: pixelsB64 } = sliceData;
    const canvas = canvasRef.current;
    canvas.width = nx;
    canvas.height = ny;
    const ctx = canvas.getContext('2d')!;
    const imageData = ctx.createImageData(nx, ny);
    const pixels = Uint8Array.from(atob(pixelsB64), (c) => c.charCodeAt(0));
    for (let row = 0; row < ny; row++) {
      const srcRow = ny - 1 - row;
      for (let col = 0; col < nx; col++) {
        const src = srcRow * nx + col;
        const dst = (row * nx + col) * 4;
        const v = pixels[src];
        imageData.data[dst]     = v;
        imageData.data[dst + 1] = v;
        imageData.data[dst + 2] = v;
        imageData.data[dst + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);
  }, [sliceData]);

  const handleSlider = (val: number) => {
    if (controlled) onZChange?.(val);
    else setInternalZ(val);
  };

  return (
    <div className="flex flex-col gap-2 items-center">
      <div className="text-xs text-zinc-400 font-mono truncate max-w-[280px]" title={imagePath}>
        {label}
      </div>
      <canvas
        ref={canvasRef}
        className="rounded border border-zinc-700"
        style={{ imageRendering: 'pixelated', maxHeight: '38vh', width: 'auto', maxWidth: '100%' }}
      />
      {sliceData && (
        <div className="flex flex-col gap-1 w-full">
          <div className="flex items-center justify-between text-xs text-zinc-500">
            <span>z = {z + 1} / {sliceData.nz}</span>
            <span className="font-mono">{sliceData.nx}×{sliceData.ny}</span>
          </div>
          <input
            type="range" min={0} max={sliceData.nz - 1} value={z}
            onChange={(e) => handleSlider(Number(e.target.value))}
            className="w-full accent-blue-500"
          />
          <div className="text-xs text-zinc-600 font-mono">
            window [{sliceData.windowMin.toFixed(0)}, {sliceData.windowMax.toFixed(0)}]
          </div>
        </div>
      )}
    </div>
  );
}

// ── MiPanel ────────────────────────────────────────────────────────────────

// MATLAB "hot" colormap: black → red → yellow → white
function hot(t: number): [number, number, number] {
  return [
    Math.round(Math.min(255, t * 3 * 255)),
    Math.round(Math.min(255, Math.max(0, (t * 3 - 1) * 255))),
    Math.round(Math.min(255, Math.max(0, (t * 3 - 2) * 255))),
  ];
}

interface MiPanelProps {
  path1: string;
  path2Match: string;
  path2Neg: string;
  z: number;
  onZChange: (z: number) => void;
}

async function fetchMiData(path1: string, path2: string): Promise<MiData | null> {
  const r = await fetch(`/api/mi?path1=${encodeURIComponent(path1)}&path2=${encodeURIComponent(path2)}`);
  const d = await r.json();
  if (!('allJoints' in d)) return null;
  const raw = Uint8Array.from(atob(d.allJoints), (c) => c.charCodeAt(0));
  return { sparkline: d.sparkline, allJoints: new Uint16Array(raw.buffer), bins: d.bins, nz: d.nz };
}

function renderJoint(canvas: HTMLCanvasElement, allJoints: Uint16Array, bins: number, z: number) {
  canvas.width = bins;
  canvas.height = bins;
  const ctx = canvas.getContext('2d')!;
  const imageData = ctx.createImageData(bins, bins);
  const offset = z * bins * bins;

  let maxCount = 0;
  for (let k = 0; k < bins * bins; k++) if (allJoints[offset + k] > maxCount) maxCount = allJoints[offset + k];
  const logMax = Math.log1p(maxCount);

  for (let a = 0; a < bins; a++) {
    for (let b = 0; b < bins; b++) {
      const count = allJoints[offset + a * bins + b];
      const t = logMax > 0 ? Math.log1p(count) / logMax : 0;
      const [r, g, bl] = hot(t);
      const dst = ((bins - 1 - b) * bins + a) * 4;
      imageData.data[dst]     = r;
      imageData.data[dst + 1] = g;
      imageData.data[dst + 2] = bl;
      imageData.data[dst + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);
}

function MiPanel({ path1, path2Match, path2Neg, z, onZChange }: MiPanelProps) {
  const [matchData, setMatchData] = useState<MiData | null>(null);
  const [negData, setNegData] = useState<MiData | null>(null);
  const [loading, setLoading] = useState(false);
  const jointRef = useRef<HTMLCanvasElement>(null);

  // Fetch both in parallel when paths change
  useEffect(() => {
    if (!path1 || !path2Match || !path2Neg) return;
    setMatchData(null);
    setNegData(null);
    setLoading(true);
    Promise.all([fetchMiData(path1, path2Match), fetchMiData(path1, path2Neg)])
      .then(([m, n]) => { setMatchData(m); setNegData(n); })
      .finally(() => setLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path1, path2Match, path2Neg]);

  // Render joint histogram for true match
  useEffect(() => {
    if (!matchData || !jointRef.current) return;
    renderJoint(jointRef.current, matchData.allJoints, matchData.bins, z);
  }, [matchData, z]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-40 text-zinc-500 text-sm">
        Computing MI…
      </div>
    );
  }
  if (!matchData) return null;

  const { sparkline: matchLine, nz, bins } = matchData;
  const negLine = negData?.sparkline ?? null;
  const allValues = negLine ? [...matchLine, ...negLine] : matchLine;
  const sparkMax = Math.max(...allValues, 0.01);
  const W = 400, H = 64;

  function toPoints(line: number[]) {
    return line.map((v, i) => `${(i / (nz - 1)) * W},${H * (1 - v / sparkMax)}`).join(' ');
  }

  return (
    <div className="flex gap-8 flex-wrap items-start mt-4 pt-4 border-t border-zinc-800">
      {/* Joint histogram (true match) */}
      <div className="flex flex-col gap-1 items-center">
        <div className="text-xs text-zinc-500 uppercase tracking-widest">
          Joint histogram — z={z + 1}
        </div>
        <canvas
          ref={jointRef}
          className="rounded border border-zinc-700"
          style={{ imageRendering: 'pixelated', width: `${bins * 3}px`, height: `${bins * 3}px` }}
        />
        <div className="text-xs text-zinc-600 text-center">T1ce intensity →</div>
        <div className="text-xs font-mono flex gap-3">
          <span className="text-blue-400">match {(matchLine[z] ?? 0).toFixed(4)}</span>
          {negLine && <span className="text-red-400">neg {(negLine[z] ?? 0).toFixed(4)}</span>}
        </div>
      </div>

      {/* Sparkline */}
      <div className="flex flex-col gap-1 flex-1 min-w-[240px]">
        <div className="text-xs text-zinc-500 uppercase tracking-widest flex gap-3">
          <span>NMI vs z</span>
          <span className="text-blue-400">— true match</span>
          {negLine && <span className="text-red-400 opacity-70">- - true negative</span>}
        </div>
        <svg
          width="100%" viewBox={`0 0 ${W} ${H + 16}`}
          className="overflow-visible cursor-crosshair"
          onClick={(e) => {
            const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
            const frac = (e.clientX - rect.left) / rect.width;
            onZChange(Math.max(0, Math.min(nz - 1, Math.round(frac * (nz - 1)))));
          }}
        >
          <line x1={0} y1={H * (1 - 0.5 / sparkMax)} x2={W} y2={H * (1 - 0.5 / sparkMax)}
            stroke="#3f3f46" strokeDasharray="2,4" strokeWidth={1} />

          {negLine && (
            <polyline fill="none" stroke="#ef4444" strokeWidth={1.5} strokeDasharray="4,3"
              points={toPoints(negLine)} />
          )}
          <polyline fill="none" stroke="#3b82f6" strokeWidth={1.5} points={toPoints(matchLine)} />

          <line x1={(z / (nz - 1)) * W} y1={0} x2={(z / (nz - 1)) * W} y2={H}
            stroke="#f97316" strokeWidth={1.5} />
          <circle cx={(z / (nz - 1)) * W} cy={H * (1 - matchLine[z] / sparkMax)} r={3} fill="#3b82f6" />
          {negLine && (
            <circle cx={(z / (nz - 1)) * W} cy={H * (1 - negLine[z] / sparkMax)} r={3} fill="#ef4444" />
          )}

          <text x={2} y={8} fill="#52525b" fontSize={9}>{sparkMax.toFixed(2)}</text>
          <text x={2} y={H} fill="#52525b" fontSize={9}>0</text>
          <text x={W / 2} y={H + 14} fill="#52525b" fontSize={9} textAnchor="middle">
            z slice (click to jump)
          </text>
        </svg>

        <div className="text-xs text-zinc-600 font-mono flex gap-4">
          <span>match mean={(matchLine.reduce((a, b) => a + b, 0) / nz).toFixed(4)}</span>
          {negLine && <span>neg mean={(negLine.reduce((a, b) => a + b, 0) / nz).toFixed(4)}</span>}
        </div>
      </div>
    </div>
  );
}

// ── DinoPanel ──────────────────────────────────────────────────────────────

interface DinoEmbeddings {
  embs: Float32Array;  // flat nz×dim, L2-normalised rows
  nz: number;
  dim: number;
}

async function fetchEmbeddings(relPath: string): Promise<DinoEmbeddings | null> {
  const r = await fetch(`/api/embeddings?path=${encodeURIComponent(relPath)}`);
  if (!r.ok) return null;
  const d = await r.json();
  if (!('data' in d)) return null;
  const raw = Uint8Array.from(atob(d.data), (c) => c.charCodeAt(0));
  return { embs: new Float32Array(raw.buffer), nz: d.nz, dim: d.dim };
}

function rowSlice(embs: Float32Array, dim: number, z: number): Float32Array {
  return embs.subarray(z * dim, (z + 1) * dim);
}

function dot32(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function dotF(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** PCA via power iteration on X^T X. Returns projected [x,y] per input vector, split by volume. */
function pca2D(allEmbs: DinoEmbeddings[]): Array<{ x: number; y: number }[]> | null {
  if (allEmbs.length === 0) return null;
  const dim = allEmbs[0].dim;

  const rows: Float64Array[] = [];
  for (const e of allEmbs) {
    for (let z = 0; z < e.nz; z++) {
      const src = rowSlice(e.embs, dim, z);
      const row = new Float64Array(dim);
      for (let i = 0; i < dim; i++) row[i] = src[i];
      rows.push(row);
    }
  }
  const n = rows.length;

  const mean = new Float64Array(dim);
  for (const row of rows) for (let i = 0; i < dim; i++) mean[i] += row[i] / n;
  const X = rows.map((row) => {
    const c = new Float64Array(dim);
    for (let i = 0; i < dim; i++) c[i] = row[i] - mean[i];
    return c;
  });

  function getEigvec(deflate?: Float64Array): Float64Array {
    const v = new Float64Array(dim);
    v[0] = 1;
    for (let iter = 0; iter < 30; iter++) {
      const Av = new Float64Array(dim);
      for (const x of X) {
        const d = dotF(x, v);
        for (let j = 0; j < dim; j++) Av[j] += d * x[j];
      }
      if (deflate) {
        const p = dotF(Av, deflate);
        for (let j = 0; j < dim; j++) Av[j] -= p * deflate[j];
      }
      let norm = 0;
      for (let j = 0; j < dim; j++) norm += Av[j] * Av[j];
      norm = Math.sqrt(norm);
      if (norm < 1e-12) break;
      for (let j = 0; j < dim; j++) v[j] = Av[j] / norm;
    }
    return v;
  }

  const pc1 = getEigvec();
  const pc2 = getEigvec(pc1);

  const projected = X.map((x) => ({ x: dotF(x, pc1), y: dotF(x, pc2) }));
  const result: Array<{ x: number; y: number }[]> = [];
  let offset = 0;
  for (const e of allEmbs) {
    result.push(projected.slice(offset, offset + e.nz));
    offset += e.nz;
  }
  return result;
}

interface DinoPanelProps {
  path1: string;
  path2Match: string;
  path2Neg: string;
  z: number;
  onZChange: (z: number) => void;
}

function DinoPanel({ path1, path2Match, path2Neg, z, onZChange }: DinoPanelProps) {
  const [queryE, setQueryE] = useState<DinoEmbeddings | null>(null);
  const [matchE, setMatchE] = useState<DinoEmbeddings | null>(null);
  const [negE, setNegE] = useState<DinoEmbeddings | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [loading, setLoading] = useState(false);
  const pcaRef = useRef<HTMLCanvasElement>(null);

  const pcaPoints = useMemo(() => {
    if (!queryE || !matchE || !negE) return null;
    return pca2D([queryE, matchE, negE]);
  }, [queryE, matchE, negE]);

  useEffect(() => {
    if (!path1 || !path2Match || !path2Neg) return;
    setQueryE(null); setMatchE(null); setNegE(null); setNotFound(false);
    setLoading(true);
    Promise.all([fetchEmbeddings(path1), fetchEmbeddings(path2Match), fetchEmbeddings(path2Neg)])
      .then(([q, m, n]) => {
        if (!q && !m && !n) { setNotFound(true); return; }
        setQueryE(q); setMatchE(m); setNegE(n);
      })
      .finally(() => setLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path1, path2Match, path2Neg]);

  // Render PCA scatter
  useEffect(() => {
    if (!pcaPoints || !pcaRef.current) return;
    const [qPts, mPts, nPts] = pcaPoints;
    const canvas = pcaRef.current;
    const S = canvas.width;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#18181b';
    ctx.fillRect(0, 0, S, S);

    const allPts = [...qPts, ...mPts, ...nPts];
    const minX = Math.min(...allPts.map((p) => p.x));
    const maxX = Math.max(...allPts.map((p) => p.x));
    const minY = Math.min(...allPts.map((p) => p.y));
    const maxY = Math.max(...allPts.map((p) => p.y));
    const pad = 12;
    const toX = (x: number) => pad + ((x - minX) / (maxX - minX || 1)) * (S - 2 * pad);
    const toY = (y: number) => S - pad - ((y - minY) / (maxY - minY || 1)) * (S - 2 * pad);

    const sets = [
      { pts: qPts, color: '#3b82f6' },
      { pts: mPts, color: '#22c55e' },
      { pts: nPts, color: '#ef4444' },
    ];

    // Thin trajectory lines
    for (const { pts, color } of sets) {
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.22;
      ctx.lineWidth = 1;
      ctx.beginPath();
      pts.forEach((p, i) =>
        i === 0 ? ctx.moveTo(toX(p.x), toY(p.y)) : ctx.lineTo(toX(p.x), toY(p.y))
      );
      ctx.stroke();
    }

    // Every 5th slice dot
    ctx.globalAlpha = 0.65;
    for (const { pts, color } of sets) {
      ctx.fillStyle = color;
      for (let i = 0; i < pts.length; i += 5) {
        ctx.beginPath();
        ctx.arc(toX(pts[i].x), toY(pts[i].y), 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Current-z dot — large + white outline
    ctx.globalAlpha = 1;
    for (const { pts, color } of sets) {
      const p = pts[Math.min(z, pts.length - 1)];
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(toX(p.x), toY(p.y), 5, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.7;
      ctx.beginPath(); ctx.arc(toX(p.x), toY(p.y), 5, 0, Math.PI * 2); ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }, [pcaPoints, z]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-12 text-zinc-500 text-sm mt-4 pt-4 border-t border-zinc-800">
        Loading DINOv2 embeddings…
      </div>
    );
  }

  if (notFound) {
    return (
      <div className="mt-4 pt-4 border-t border-zinc-800 text-xs text-zinc-600 font-mono">
        DINOv2 embeddings not found. Precompute with:
        <pre className="mt-1 bg-zinc-900 rounded p-2 text-zinc-400 whitespace-pre-wrap">
          modal run laurence/extract_dinov2_embeddings_modal.py
        </pre>
        Then copy <code className="text-zinc-500">*_dinov2.f32</code> files to your local data directory.
      </div>
    );
  }

  if (!queryE || !matchE || !negE) return null;

  const nz = queryE.nz;
  const dim = queryE.dim;
  const matchSim: number[] = [];
  const negSim: number[] = [];
  for (let zi = 0; zi < nz; zi++) {
    const q = rowSlice(queryE.embs, dim, zi);
    matchSim.push(dot32(q, rowSlice(matchE.embs, dim, zi)));
    negSim.push(dot32(q, rowSlice(negE.embs, dim, zi)));
  }

  const sparkMax = Math.max(...matchSim, ...negSim);
  const sparkMin = Math.min(...matchSim, ...negSim);
  const sparkRange = sparkMax - sparkMin || 0.01;
  const W = 400, H = 64;

  function toSimPoints(line: number[]) {
    return line
      .map((v, i) => `${(i / (nz - 1)) * W},${H * (1 - (v - sparkMin) / sparkRange)}`)
      .join(' ');
  }

  const matchMean = matchSim.reduce((a, b) => a + b, 0) / nz;
  const negMean = negSim.reduce((a, b) => a + b, 0) / nz;

  return (
    <div className="flex gap-8 flex-wrap items-start mt-4 pt-4 border-t border-zinc-800">
      {/* PCA scatter */}
      <div className="flex flex-col gap-1 items-center">
        <div className="text-xs text-zinc-500 uppercase tracking-widest">
          DINOv2 PCA — z={z + 1}
        </div>
        <canvas
          ref={pcaRef}
          width={192} height={192}
          className="rounded border border-zinc-700"
          style={{ width: '192px', height: '192px' }}
        />
        <div className="text-xs font-mono flex gap-2">
          <span className="text-blue-400">■ query</span>
          <span className="text-green-400">■ match</span>
          <span className="text-red-400">■ neg</span>
        </div>
      </div>

      {/* Cosine similarity sparkline */}
      <div className="flex flex-col gap-1 flex-1 min-w-[240px]">
        <div className="text-xs text-zinc-500 uppercase tracking-widest flex gap-3">
          <span>DINOv2 cos sim vs z</span>
          <span className="text-green-400">— true match</span>
          <span className="text-red-400 opacity-70">- - true negative</span>
        </div>
        <svg
          width="100%" viewBox={`0 0 ${W} ${H + 16}`}
          className="overflow-visible cursor-crosshair"
          onClick={(e) => {
            const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
            const frac = (e.clientX - rect.left) / rect.width;
            onZChange(Math.max(0, Math.min(nz - 1, Math.round(frac * (nz - 1)))));
          }}
        >
          <polyline fill="none" stroke="#ef4444" strokeWidth={1.5} strokeDasharray="4,3"
            points={toSimPoints(negSim)} />
          <polyline fill="none" stroke="#22c55e" strokeWidth={1.5}
            points={toSimPoints(matchSim)} />

          <line x1={(z / (nz - 1)) * W} y1={0} x2={(z / (nz - 1)) * W} y2={H}
            stroke="#f97316" strokeWidth={1.5} />
          <circle cx={(z / (nz - 1)) * W}
            cy={H * (1 - (matchSim[z] - sparkMin) / sparkRange)} r={3} fill="#22c55e" />
          <circle cx={(z / (nz - 1)) * W}
            cy={H * (1 - (negSim[z] - sparkMin) / sparkRange)} r={3} fill="#ef4444" />

          <text x={2} y={8} fill="#52525b" fontSize={9}>{sparkMax.toFixed(3)}</text>
          <text x={2} y={H} fill="#52525b" fontSize={9}>{sparkMin.toFixed(3)}</text>
          <text x={W / 2} y={H + 14} fill="#52525b" fontSize={9} textAnchor="middle">
            z slice (click to jump)
          </text>
        </svg>

        <div className="text-xs font-mono flex gap-4">
          <span className="text-green-400">match sim={matchSim[z].toFixed(4)} (mean {matchMean.toFixed(4)})</span>
          <span className="text-red-400">neg sim={negSim[z].toFixed(4)} (mean {negMean.toFixed(4)})</span>
          <span className="text-zinc-500">gap={(matchMean - negMean).toFixed(4)}</span>
        </div>
      </div>
    </div>
  );
}

// ── GalleryView ────────────────────────────────────────────────────────────

// ── GalleryMiPlot ──────────────────────────────────────────────────────────

interface GalleryMiPlotProps {
  pathQuery: string;
  pathMatch: string;
  pathNegs: string[];
  z: number;
  onZChange: (z: number) => void;
}

function GalleryMiPlot({ pathQuery, pathMatch, pathNegs, z, onZChange }: GalleryMiPlotProps) {
  const [matchLine, setMatchLine] = useState<number[] | null>(null);
  // parallel array aligned to pathNegs
  const [negLines, setNegLines] = useState<(number[] | null)[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!pathQuery || !pathMatch || pathNegs.length === 0) return;
    setMatchLine(null);
    setNegLines([]);
    setLoading(true);

    async function fetchSparkline(p2: string): Promise<number[] | null> {
      const r = await fetch(
        `/api/mi?path1=${encodeURIComponent(pathQuery)}&path2=${encodeURIComponent(p2)}&sparklineOnly=true`
      );
      const d = await r.json();
      return Array.isArray(d.sparkline) ? d.sparkline : null;
    }

    const allPaths = [pathMatch, ...pathNegs];
    Promise.all(allPaths.map(fetchSparkline))
      .then((results) => {
        setMatchLine(results[0]);
        setNegLines(results.slice(1));
      })
      .finally(() => setLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathQuery, pathMatch, pathNegs.join('|')]);

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-zinc-500 text-xs py-4">
        <span className="animate-pulse">Computing MI for all negatives…</span>
      </div>
    );
  }
  if (!matchLine) return null;

  const nz = matchLine.length;
  const allLines = [matchLine, ...negLines.filter((l): l is number[] => l !== null)];
  const sparkMax = Math.max(...allLines.flat(), 0.01);
  const W = 500, H = 80;

  function toPoints(line: number[]) {
    return line.map((v, i) => `${(i / (nz - 1)) * W},${H * (1 - v / sparkMax)}`).join(' ');
  }

  const meanFn = (line: number[]) => line.reduce((a, b) => a + b, 0) / line.length;
  const matchMean = meanFn(matchLine);

  // Build ordered list for correlation matrix: match first, then negatives
  const corrLines: (number[] | null)[] = [matchLine, ...negLines];
  const labels = ['M', ...negLines.map((_, i) => `N${i + 1}`)];
  const n = corrLines.length;

  // Pearson correlation between two sparklines
  function pearson(a: number[], b: number[]): number {
    const len = a.length;
    const ma = meanFn(a), mb = meanFn(b);
    let num = 0, da = 0, db = 0;
    for (let i = 0; i < len; i++) {
      const ai = a[i] - ma, bi = b[i] - mb;
      num += ai * bi; da += ai * ai; db += bi * bi;
    }
    return da > 0 && db > 0 ? num / Math.sqrt(da * db) : 1;
  }

  // n×n correlation matrix
  const corr: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (__, j) => {
      const a = corrLines[i], b = corrLines[j];
      return a && b ? pearson(a, b) : (i === j ? 1 : 0);
    })
  );

  // Viridis colormap, clamped to [0.92, 1.0]
  function corrColor(r: number): string {
    const t = Math.max(0, Math.min(1, (r - 0.92) / (1.0 - 0.92)));
    // 5-stop viridis approximation
    const stops: [number, number, number, number][] = [
      [0.00,  68,   1,  84],
      [0.25,  58,  82, 139],
      [0.50,  32, 144, 140],
      [0.75,  94, 201,  97],
      [1.00, 253, 231,  37],
    ];
    let i = 0;
    while (i < stops.length - 2 && t > stops[i + 1][0]) i++;
    const [t0, r0, g0, b0] = stops[i];
    const [t1, r1, g1, b1] = stops[i + 1];
    const f = (t - t0) / (t1 - t0);
    return `rgb(${Math.round(r0 + f * (r1 - r0))},${Math.round(g0 + f * (g1 - g0))},${Math.round(b0 + f * (b1 - b0))})`;
  }

  const CELL = 36, LABEL = 26;
  const svgW = LABEL + n * CELL;
  const svgH = LABEL + n * CELL;

  return (
    <div className="flex flex-col gap-2 border-t border-zinc-800 pt-4 mt-2">
      <div className="text-xs text-zinc-500 uppercase tracking-widest flex gap-4">
        <span>NMI vs z — all targets</span>
        <span className="text-blue-400">— true match ({matchMean.toFixed(4)} mean)</span>
        <span className="text-zinc-600">— 10 negatives</span>
      </div>
      <svg
        width="100%" viewBox={`0 0 ${W} ${H + 16}`}
        className="overflow-visible cursor-crosshair"
        onClick={(e) => {
          const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          const frac = (e.clientX - rect.left) / rect.width;
          onZChange(Math.max(0, Math.min(nz - 1, Math.round(frac * (nz - 1)))));
        }}
      >
        <line x1={0} y1={H * (1 - 0.5 / sparkMax)} x2={W} y2={H * (1 - 0.5 / sparkMax)}
          stroke="#3f3f46" strokeDasharray="2,4" strokeWidth={1} />
        {negLines.map((line, i) => line && (
          <polyline key={i} fill="none" stroke="#6b7280" strokeWidth={1} opacity={0.5}
            points={toPoints(line)} />
        ))}
        <polyline fill="none" stroke="#3b82f6" strokeWidth={2} points={toPoints(matchLine)} />
        <line x1={(z / (nz - 1)) * W} y1={0} x2={(z / (nz - 1)) * W} y2={H}
          stroke="#f97316" strokeWidth={1.5} />
        <circle cx={(z / (nz - 1)) * W} cy={H * (1 - matchLine[z] / sparkMax)} r={3} fill="#3b82f6" />
        <text x={2} y={8} fill="#52525b" fontSize={9}>{sparkMax.toFixed(2)}</text>
        <text x={2} y={H} fill="#52525b" fontSize={9}>0</text>
        <text x={W / 2} y={H + 14} fill="#52525b" fontSize={9} textAnchor="middle">
          z slice (click to jump)
        </text>
      </svg>

      {/* n×n Pearson correlation matrix of NMI sparklines */}
      <div className="pt-3 border-t border-zinc-800/60">
        <div className="text-xs text-zinc-500 uppercase tracking-widest mb-3">
          NMI profile correlation matrix — M = true match, N1–N{negLines.length} = negatives
        </div>
        <svg width={svgW} height={svgH} className="overflow-visible font-mono">
          {/* Column labels */}
          {labels.map((lbl, j) => (
            <text key={j} x={LABEL + j * CELL + CELL / 2} y={LABEL - 4}
              textAnchor="middle" fontSize={9}
              fill={j === 0 ? '#93c5fd' : '#52525b'}>
              {lbl}
            </text>
          ))}
          {/* Row labels + cells */}
          {corr.map((row, i) => (
            <g key={i}>
              <text x={LABEL - 4} y={LABEL + i * CELL + CELL / 2 + 3}
                textAnchor="end" fontSize={9}
                fill={i === 0 ? '#93c5fd' : '#52525b'}>
                {labels[i]}
              </text>
              {row.map((val, j) => {
                const cx = LABEL + j * CELL, cy = LABEL + i * CELL;
                const textBright = val > 0.5;
                return (
                  <g key={j}>
                    <rect x={cx} y={cy} width={CELL} height={CELL}
                      fill={corrColor(val)}
                      stroke="#09090b"
                      strokeWidth={0.5}
                    />
                    <text x={cx + CELL / 2} y={cy + CELL / 2 + 3}
                      textAnchor="middle" fontSize={8}
                      fill={textBright ? '#000' : '#aaa'}>
                      {val.toFixed(2)}
                    </text>
                  </g>
                );
              })}
            </g>
          ))}
        </svg>
        {/* Colour legend */}
        <div className="flex items-center gap-2 mt-2">
          <span className="text-xs text-zinc-600 font-mono">0.92</span>
          <svg width={100} height={10}>
            <defs>
              <linearGradient id="corrGrad" x1="0" x2="1" y1="0" y2="0">
                <stop offset="0%"   stopColor={corrColor(0.92)} />
                <stop offset="25%"  stopColor={corrColor(0.94)} />
                <stop offset="50%"  stopColor={corrColor(0.96)} />
                <stop offset="75%"  stopColor={corrColor(0.98)} />
                <stop offset="100%" stopColor={corrColor(1.00)} />
              </linearGradient>
            </defs>
            <rect width={100} height={10} fill="url(#corrGrad)" rx={2} />
          </svg>
          <span className="text-xs text-zinc-600 font-mono">1.00</span>
          <span className="text-xs text-zinc-600 ml-3">Pearson r of per-slice NMI profiles</span>
        </div>
      </div>
    </div>
  );
}

// ── GalleryView ────────────────────────────────────────────────────────────

/** Shows the query + true match + up to 10 random negatives, all z-synced. */
function GalleryView({ row, rows, idx }: { row: Row; rows: Row[]; idx: number }) {
  const [z, setZ] = useState(0);
  const [nz, setNz] = useState(155);

  // Stable sample of 10 negatives — re-seeded only when pair changes
  const negRows = useMemo(() => {
    const pool = rows.filter((_, i) => i !== idx);
    // deterministic shuffle keyed to idx so it doesn't jump on re-renders
    let seed = idx * 2654435761;
    const rand = () => { seed = (seed ^ (seed >>> 16)) * 2246822519 >>> 0; seed = (seed ^ (seed >>> 13)) * 3266489917 >>> 0; return (seed ^ (seed >>> 16)) / 0x100000000; };
    return [...pool].sort(() => rand() - 0.5).slice(0, 10);
  }, [rows, idx]);

  return (
    <div className="flex flex-col gap-4">
      {/* Shared z slider */}
      <div className="flex items-center gap-3 text-xs text-zinc-500">
        <span className="shrink-0">z = {z + 1} / {nz}</span>
        <input
          type="range" min={0} max={nz - 1} value={z}
          onChange={(e) => setZ(Number(e.target.value))}
          className="flex-1 accent-orange-500"
        />
      </div>

      {/* Query + match row */}
      <div className="flex gap-4 flex-wrap">
        <div className="flex flex-col gap-1 items-center" style={{ width: 160 }}>
          <div className="text-xs text-zinc-500 uppercase tracking-widest self-start">Query</div>
          <SlicePanel imagePath={row['query_image']} label={row['query_id']} z={z} onNzReady={setNz} onZChange={setZ} />
        </div>
        <div className="flex flex-col gap-1 items-center" style={{ width: 160 }}>
          <div className="text-xs text-emerald-500 uppercase tracking-widest self-start">✓ true match</div>
          <SlicePanel imagePath={row['target_image']} label={row['target_id']} z={z} onZChange={setZ} />
        </div>
      </div>

      <div className="border-t border-zinc-800 pt-4">
        <div className="text-xs text-zinc-500 uppercase tracking-widest mb-3">10 random negatives</div>
        <div className="flex flex-wrap gap-4">
          {negRows.map((neg) => (
            <div key={neg['pair_id']} className="flex flex-col gap-1 items-center" style={{ width: 160 }}>
              <div className="text-xs text-red-400 uppercase tracking-widest self-start truncate w-full" title={neg['target_id']}>
                ✗ {neg['target_id']}
              </div>
              <SlicePanel imagePath={neg['target_image']} label={neg['target_id']} z={z} onZChange={setZ} />
            </div>
          ))}
        </div>
      </div>

      <GalleryMiPlot
        pathQuery={row['query_image']}
        pathMatch={row['target_image']}
        pathNegs={negRows.map((r) => r['target_image'])}
        z={z}
        onZChange={setZ}
      />
    </div>
  );
}

// ── PairView ───────────────────────────────────────────────────────────────

// Grayscale palette as [[r,g,b], ...] — gifenc's expected format
// index[i] = gray value = palette index directly, so no applyPalette needed
const GRAY_PALETTE: number[][] = Array.from({ length: 256 }, (_, i) => [i, i, i]);

async function fetchFrames(relPath: string): Promise<{ nz: number; nx: number; ny: number; frames: Uint8Array } | null> {
  const r = await fetch(`/api/frames?path=${encodeURIComponent(relPath)}`);
  const d = await r.json();
  if (!('frames' in d)) return null;
  const raw = Uint8Array.from(atob(d.frames), (c) => c.charCodeAt(0));
  return { nz: d.nz, nx: d.nx, ny: d.ny, frames: raw };
}

function PairView({ row, rows, idx }: { row: Row; rows: Row[]; idx: number }) {
  const [tab, setTab] = useState<'pair' | 'gallery'>('pair');
  const [z, setZ] = useState(0);
  const [nz, setNz] = useState(155);
  const [playing, setPlaying] = useState(false);
  const [exportStatus, setExportStatus] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const negRow = rows[(idx + 1) % rows.length];

  async function exportGif() {
    setExportStatus('Fetching frames…');
    const [q, m, n, matchMi, negMi] = await Promise.all([
      fetchFrames(row['query_image']),
      fetchFrames(row['target_image']),
      fetchFrames(negRow['target_image']),
      fetchMiData(row['query_image'], row['target_image']),
      fetchMiData(row['query_image'], negRow['target_image']),
    ]);
    if (!q || !m || !n || !matchMi) { setExportStatus('Failed to fetch data'); return; }

    const { nz: frameCount, nx, ny } = q;
    const sliceSize = nx * ny;
    const GAP = 4;
    const SLICE_W = nx * 3 + GAP * 2;
    const BOTTOM_H = Math.round(ny * 0.55);
    const HIST_SIZE = BOTTOM_H;
    const SPARK_X = HIST_SIZE + GAP;
    const SPARK_W = SLICE_W - SPARK_X;
    const W = SLICE_W;
    const H = ny + GAP + BOTTOM_H;

    const { bins, allJoints, sparkline: matchLine } = matchMi;
    const negLine = negMi?.sparkline ?? null;
    const allSparkVals = negLine ? [...matchLine, ...negLine] : matchLine;
    const sparkMax = Math.max(...allSparkVals, 0.01);

    // Temp canvas for joint histogram (native resolution, then scaled onto main)
    const histCanvas = new OffscreenCanvas(bins, bins);
    const histCtx = histCanvas.getContext('2d') as OffscreenCanvasRenderingContext2D;

    // Main composite canvas
    const canvas = new OffscreenCanvas(W, H);
    const ctx = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D;

    setExportStatus('Encoding GIF…');
    await new Promise(r => setTimeout(r, 0));

    const gif = GIFEncoder();
    const panels = [q.frames, m.frames, n.frames];

    for (let z = 0; z < frameCount; z++) {
      if (z % 10 === 0) {
        setExportStatus(`Encoding GIF… ${Math.round((z / frameCount) * 100)}%`);
        await new Promise(r => setTimeout(r, 0));
      }

      // ── Background ─────────────────────────────────────────────────────────
      ctx.fillStyle = '#18181b';
      ctx.fillRect(0, 0, W, H);

      // ── Three slice panels ─────────────────────────────────────────────────
      for (let p = 0; p < 3; p++) {
        const imgData = ctx.createImageData(nx, ny);
        for (let row2 = 0; row2 < ny; row2++) {
          const srcRow = ny - 1 - row2;
          for (let col = 0; col < nx; col++) {
            const v = panels[p][z * sliceSize + srcRow * nx + col];
            const dst = (row2 * nx + col) * 4;
            imgData.data[dst] = v; imgData.data[dst+1] = v;
            imgData.data[dst+2] = v; imgData.data[dst+3] = 255;
          }
        }
        ctx.putImageData(imgData, p * (nx + GAP), 0);
      }

      // ── Joint histogram (hot colormap, scaled) ─────────────────────────────
      const histImg = histCtx.createImageData(bins, bins);
      const hOffset = z * bins * bins;
      let maxCount = 0;
      for (let k = 0; k < bins * bins; k++) if (allJoints[hOffset + k] > maxCount) maxCount = allJoints[hOffset + k];
      const logMax = Math.log1p(maxCount);
      for (let a = 0; a < bins; a++) {
        for (let b = 0; b < bins; b++) {
          const t = logMax > 0 ? Math.log1p(allJoints[hOffset + a * bins + b]) / logMax : 0;
          const [r2, g2, b2] = hot(t);
          const dst = ((bins - 1 - b) * bins + a) * 4;
          histImg.data[dst] = r2; histImg.data[dst+1] = g2;
          histImg.data[dst+2] = b2; histImg.data[dst+3] = 255;
        }
      }
      histCtx.putImageData(histImg, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(histCanvas, 0, ny + GAP, HIST_SIZE, HIST_SIZE);

      // ── Sparkline ──────────────────────────────────────────────────────────
      const sy = ny + GAP;
      ctx.fillStyle = '#27272a';
      ctx.fillRect(SPARK_X, sy, SPARK_W, BOTTOM_H);

      // Grid at 0.5
      ctx.strokeStyle = '#3f3f46';
      ctx.setLineDash([2, 4]);
      ctx.lineWidth = 1;
      const gridY = sy + BOTTOM_H * (1 - 0.5 / sparkMax);
      ctx.beginPath(); ctx.moveTo(SPARK_X, gridY); ctx.lineTo(SPARK_X + SPARK_W, gridY); ctx.stroke();

      function sparkX(i: number) { return SPARK_X + (i / (frameCount - 1)) * SPARK_W; }
      function sparkY(v: number) { return sy + BOTTOM_H * (1 - v / sparkMax); }

      // Neg line (dashed red)
      if (negLine) {
        ctx.strokeStyle = '#ef4444';
        ctx.setLineDash([4, 3]);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        negLine.forEach((v, i) => i === 0 ? ctx.moveTo(sparkX(i), sparkY(v)) : ctx.lineTo(sparkX(i), sparkY(v)));
        ctx.stroke();
      }

      // Match line (solid blue)
      ctx.strokeStyle = '#3b82f6';
      ctx.setLineDash([]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      matchLine.forEach((v, i) => i === 0 ? ctx.moveTo(sparkX(i), sparkY(v)) : ctx.lineTo(sparkX(i), sparkY(v)));
      ctx.stroke();

      // Cursor
      ctx.strokeStyle = '#f97316';
      ctx.setLineDash([]);
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(sparkX(z), sy); ctx.lineTo(sparkX(z), sy + BOTTOM_H); ctx.stroke();

      // Dots
      ctx.fillStyle = '#3b82f6';
      ctx.beginPath(); ctx.arc(sparkX(z), sparkY(matchLine[z]), 3, 0, Math.PI * 2); ctx.fill();
      if (negLine) {
        ctx.fillStyle = '#ef4444';
        ctx.beginPath(); ctx.arc(sparkX(z), sparkY(negLine[z]), 3, 0, Math.PI * 2); ctx.fill();
      }

      // NMI label
      ctx.fillStyle = '#a1a1aa';
      ctx.font = '10px monospace';
      ctx.fillText(
        `match ${matchLine[z].toFixed(3)}${negLine ? `  neg ${negLine[z].toFixed(3)}` : ''}`,
        SPARK_X + 4, sy + BOTTOM_H - 4,
      );

      // ── Encode frame ───────────────────────────────────────────────────────
      const rgba = new Uint8Array(ctx.getImageData(0, 0, W, H).data.buffer);
      const palette = quantize(rgba, 256, { format: 'rgb565' });
      const index = applyPalette(rgba, palette);
      gif.writeFrame(index, W, H, { palette, delay: 120 });
    }

    gif.finish();
    const blob = new Blob([gif.bytes().buffer as ArrayBuffer], { type: 'image/gif' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${row['pair_id']}.gif`;
    a.click();
    URL.revokeObjectURL(url);
    setExportStatus(null);
  }

  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(() => {
        setZ((prev) => (prev + 1) % nz);
      }, 120);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [playing, nz]);

  return (
    <div className="flex flex-col gap-4">
      {/* Tab bar */}
      <div className="flex gap-1 border-b border-zinc-800 pb-2">
        {(['pair', 'gallery'] as const).map((t) => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
              tab === t ? 'bg-zinc-700 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800'
            }`}>
            {t === 'pair' ? 'Pair + MI' : 'Gallery'}
          </button>
        ))}
      </div>

      {tab === 'gallery' && (
        <GalleryView row={row} rows={rows} idx={idx} />
      )}

      {tab === 'pair' && <>
      {/* Controls bar */}
      <div className="flex items-center gap-3 text-xs text-zinc-500">
        <button
          onClick={() => setPlaying((p) => !p)}
          className="shrink-0 w-7 h-7 flex items-center justify-center rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-300 transition-colors"
          title={playing ? 'Pause' : 'Play'}
        >
          {playing ? '⏸' : '▶'}
        </button>
        <span className="shrink-0">z = {z + 1} / {nz}</span>
        <input
          type="range" min={0} max={nz - 1} value={z}
          onChange={(e) => { setPlaying(false); setZ(Number(e.target.value)); }}
          className="flex-1 accent-orange-500"
        />
        <button
          onClick={exportGif}
          disabled={!!exportStatus}
          className="shrink-0 px-2 py-1 rounded text-xs font-medium bg-zinc-800 text-zinc-400 hover:text-zinc-200 border border-zinc-700 disabled:opacity-50 disabled:cursor-wait transition-colors"
        >
          {exportStatus ?? 'Export GIF'}
        </button>
      </div>

      <div className="flex gap-6 flex-wrap">
        <div className="flex-1 min-w-[200px]">
          <div className="text-xs text-zinc-500 uppercase tracking-widest mb-2">
            Query — {row['query_modality']}
          </div>
          <SlicePanel
            imagePath={row['query_image']}
            label={row['query_id']}
            z={z}
            onNzReady={(n) => setNz(n)}
            onZChange={setZ}
          />
        </div>
        <div className="flex-1 min-w-[200px]">
          <div className="text-xs text-emerald-500 uppercase tracking-widest mb-2">
            ✓ true match — {row['target_modality']}
          </div>
          <SlicePanel
            imagePath={row['target_image']}
            label={row['target_id']}
            z={z}
            onZChange={setZ}
          />
        </div>
        <div className="flex-1 min-w-[200px]">
          <div className="text-xs text-red-400 uppercase tracking-widest mb-2">
            ✗ true negative — {negRow['target_modality']}
          </div>
          <SlicePanel
            imagePath={negRow['target_image']}
            label={negRow['target_id']}
            z={z}
            onZChange={setZ}
          />
        </div>
      </div>

      <MiPanel
        path1={row['query_image']}
        path2Match={row['target_image']}
        path2Neg={negRow['target_image']}
        z={z}
        onZChange={setZ}
      />
      <DinoPanel
        path1={row['query_image']}
        path2Match={row['target_image']}
        path2Neg={negRow['target_image']}
        z={z}
        onZChange={setZ}
      />
      </>}
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────

const DATASETS: Dataset[] = ['dataset1', 'dataset2', 'dataset3'];

const SPLITS_BY_DS: Record<Dataset, SplitKey[]> = {
  dataset1: ['train_pairs', 'val_queries', 'val_gallery', 'test_queries', 'test_gallery'],
  dataset2: ['val_queries', 'val_gallery', 'test_queries', 'test_gallery'],
  dataset3: ['val_queries', 'val_gallery', 'test_queries', 'test_gallery'],
};

const SPLIT_LABELS: Record<SplitKey, string> = {
  train_pairs: 'train pairs',
  val_queries: 'val queries',
  val_gallery: 'val gallery',
  test_queries: 'test queries',
  test_gallery: 'test gallery',
};

export default function NiiViewer() {
  const [dataIndex, setDataIndex] = useState<DataIndex | null>(null);
  const [dataset, setDataset] = useState<Dataset>('dataset1');
  const [split, setSplit] = useState<SplitKey>('train_pairs');
  const [selectedIdx, setSelectedIdx] = useState(0);

  useEffect(() => {
    fetch('/api/files')
      .then((r) => r.json())
      .then((d: DataIndex) => setDataIndex(d));
  }, []);

  useEffect(() => {
    const available = SPLITS_BY_DS[dataset];
    if (!available.includes(split)) setSplit(available[0]);
    setSelectedIdx(0);
  }, [dataset, split]);

  useEffect(() => { setSelectedIdx(0); }, [split]);

  const rows = dataIndex?.[dataset]?.[split] ?? [];
  const selected = rows[selectedIdx] ?? null;
  const isPairs = split === 'train_pairs';
  const isGallery = split.endsWith('_gallery');
  const imagePathKey = isGallery ? 'target_image' : 'query_image';
  const idKey = isPairs ? 'pair_id' : isGallery ? 'target_id' : 'query_id';

  return (
    <div className="flex flex-col min-h-screen bg-zinc-950 text-zinc-100">
      {/* Header */}
      <div className="border-b border-zinc-800 px-6 py-3 flex items-center gap-4 flex-wrap">
        <h1 className="text-lg font-semibold tracking-tight shrink-0">NIfTI Explorer</h1>
        <div className="flex gap-1">
          {DATASETS.map((ds) => (
            <button key={ds} onClick={() => setDataset(ds)}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                dataset === ds ? 'bg-blue-600 text-white' : 'text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800'
              }`}>
              {ds}
            </button>
          ))}
        </div>
        <div className="flex gap-1 ml-2 flex-wrap">
          {SPLITS_BY_DS[dataset].map((s) => {
            const count = dataIndex?.[dataset]?.[s]?.length ?? 0;
            return (
              <button key={s} onClick={() => setSplit(s)}
                className={`px-3 py-1 rounded text-sm transition-colors ${
                  split === s ? 'bg-zinc-700 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800'
                }`}>
                {SPLIT_LABELS[s]}
                {count > 0 && <span className="ml-1 text-xs text-zinc-500">({count})</span>}
              </button>
            );
          })}
        </div>
      </div>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* List */}
        <div className="w-64 shrink-0 border-r border-zinc-800 overflow-y-auto">
          {rows.length === 0
            ? <p className="p-4 text-zinc-500 text-sm">No data</p>
            : rows.map((row, i) => {
              const id = row[idKey] ?? `row-${i}`;
              const mod = row['query_modality'] ?? row['target_modality'] ?? '';
              return (
                <button key={id} onClick={() => setSelectedIdx(i)}
                  className={`w-full text-left px-4 py-2 border-b border-zinc-800/60 text-sm transition-colors ${
                    i === selectedIdx ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200'
                  }`}>
                  <div className="font-mono truncate text-xs">{id}</div>
                  {mod && <div className="text-xs text-zinc-600 mt-0.5">{mod}</div>}
                </button>
              );
            })}
        </div>

        {/* Viewer */}
        <div className="flex-1 overflow-y-auto p-6">
          {!selected
            ? <p className="text-zinc-500 text-sm">Select an item.</p>
            : isPairs
              ? <PairView key={selected['pair_id']} row={selected} rows={rows} idx={selectedIdx} />
              : (
                <div className="max-w-md">
                  <div className="text-xs text-zinc-500 uppercase tracking-widest mb-3">
                    {isGallery ? `Target — ${selected['target_modality']}` : `Query — ${selected['query_modality']}`}
                  </div>
                  <SlicePanel imagePath={selected[imagePathKey]} label={selected[idKey]} />
                </div>
              )
          }
        </div>
      </div>
    </div>
  );
}
