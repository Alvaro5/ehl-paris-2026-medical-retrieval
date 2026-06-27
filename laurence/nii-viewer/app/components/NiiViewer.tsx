'use client';

import { useEffect, useRef, useState, useCallback } from 'react';

interface SliceData {
  nx: number;
  ny: number;
  nz: number;
  z: number;
  pixels: number[];
  windowMin: number;
  windowMax: number;
}

export default function NiiViewer() {
  const [files, setFiles] = useState<string[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>('');
  const [sliceData, setSliceData] = useState<SliceData | null>(null);
  const [z, setZ] = useState(0);
  const [loading, setLoading] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    fetch('/api/files')
      .then((r) => r.json())
      .then(({ files }: { files: string[] }) => {
        setFiles(files);
        if (files.length > 0) setSelectedFile(files[0]);
      });
  }, []);

  const fetchSlice = useCallback(async (file: string, zIdx: number) => {
    if (!file) return;
    setLoading(true);
    try {
      const res = await fetch(`/api/slice?file=${encodeURIComponent(file)}&z=${zIdx}`);
      const data: SliceData = await res.json();
      setSliceData(data);
    } finally {
      setLoading(false);
    }
  }, []);

  // When file changes, fetch middle slice first to learn nz
  useEffect(() => {
    if (!selectedFile) return;
    fetch(`/api/slice?file=${encodeURIComponent(selectedFile)}`)
      .then((r) => r.json())
      .then((data: SliceData) => {
        setZ(data.z);
        setSliceData(data);
      });
  }, [selectedFile]);

  // When z slider changes (after we have nz), fetch that slice
  useEffect(() => {
    if (!selectedFile || !sliceData) return;
    fetchSlice(selectedFile, z);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [z, selectedFile]);

  // Render pixels onto canvas
  useEffect(() => {
    if (!sliceData || !canvasRef.current) return;
    const { nx, ny, pixels } = sliceData;
    const canvas = canvasRef.current;
    canvas.width = nx;
    canvas.height = ny;
    const ctx = canvas.getContext('2d')!;
    const imageData = ctx.createImageData(nx, ny);

    // NIfTI stores rows bottom-up; flip vertically for display
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

  return (
    <div className="flex flex-col items-center gap-6 p-8 min-h-screen bg-zinc-950 text-zinc-100">
      <h1 className="text-2xl font-semibold tracking-tight">NIfTI Viewer — Axial (XY) Plane</h1>

      <div className="flex flex-col gap-3 w-full max-w-xl">
        <label className="text-sm text-zinc-400">Select scan</label>
        <select
          className="rounded-lg bg-zinc-800 px-3 py-2 text-sm text-zinc-100 border border-zinc-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={selectedFile}
          onChange={(e) => { setSelectedFile(e.target.value); setZ(0); }}
        >
          {files.map((f) => (
            <option key={f} value={f}>{f}</option>
          ))}
        </select>
      </div>

      {sliceData && (
        <div className="flex flex-col gap-3 w-full max-w-xl">
          <div className="flex items-center justify-between text-sm text-zinc-400">
            <span>Axial slice</span>
            <span className="font-mono">{z + 1} / {sliceData.nz}</span>
          </div>
          <input
            type="range"
            min={0}
            max={sliceData.nz - 1}
            value={z}
            onChange={(e) => setZ(Number(e.target.value))}
            className="w-full accent-blue-500"
          />
          <div className="flex justify-between text-xs text-zinc-500 font-mono">
            <span>window: [{sliceData.windowMin.toFixed(0)}, {sliceData.windowMax.toFixed(0)}]</span>
            <span>{sliceData.nx} × {sliceData.ny} px</span>
          </div>
        </div>
      )}

      <div className="relative">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-zinc-950/70 z-10 rounded">
            <span className="text-sm text-zinc-400">Loading…</span>
          </div>
        )}
        <canvas
          ref={canvasRef}
          className="rounded border border-zinc-700 max-w-full"
          style={{ imageRendering: 'pixelated', maxHeight: '70vh', width: 'auto' }}
        />
      </div>

      {!sliceData && !loading && (
        <p className="text-zinc-500 text-sm">Select a file to begin.</p>
      )}
    </div>
  );
}
