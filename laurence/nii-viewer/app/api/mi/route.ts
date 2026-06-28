import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import * as nifti from 'nifti-reader-js';

const DATA_ROOT =
  'C:\\Users\\laure\\Projects\\ehl2026\\ehl-paris-2026-medical-retrieval\\data\\ehl-paris-medical-image-retrieval';

const BINS = 64;

function resolvePath(rel: string): string | null {
  const resolved = path.resolve(DATA_ROOT, rel);
  if (!resolved.startsWith(path.resolve(DATA_ROOT))) return null;
  if (fs.existsSync(resolved)) return resolved;
  if (resolved.endsWith('.gz') && fs.existsSync(resolved.slice(0, -3))) return resolved.slice(0, -3);
  return null;
}

interface CachedVolume { data: Float32Array; nx: number; ny: number; nz: number; p2: number; p98: number; }
const MAX_CACHED = 24;
const volumeCache = new Map<string, CachedVolume>();

/** Percentile of a Float32Array (modifies a copy). */
function percentile(arr: Float32Array, p: number): number {
  const sorted = Float32Array.from(arr).sort();
  return sorted[Math.floor((p / 100) * (sorted.length - 1))];
}

function readVolume(filePath: string): CachedVolume | null {
  if (volumeCache.has(filePath)) return volumeCache.get(filePath)!;

  const buf = fs.readFileSync(filePath);
  let ab: ArrayBuffer = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  if (nifti.isCompressed(ab)) ab = nifti.decompress(ab) as ArrayBuffer;
  if (!nifti.isNIFTI(ab)) return null;

  const header = nifti.readHeader(ab);
  const imageBuffer = nifti.readImage(header, ab);
  const nx = header.dims[1], ny = header.dims[2], nz = header.dims[3];
  const total = nx * ny * nz;
  const data = new Float32Array(total);

  switch (header.datatypeCode) {
    case 2:   { const v = new Uint8Array(imageBuffer);   for (let i = 0; i < total; i++) data[i] = v[i]; break; }
    case 256: { const v = new Int8Array(imageBuffer);    for (let i = 0; i < total; i++) data[i] = v[i]; break; }
    case 4:   { const v = new Int16Array(imageBuffer);   for (let i = 0; i < total; i++) data[i] = v[i]; break; }
    case 512: { const v = new Uint16Array(imageBuffer);  for (let i = 0; i < total; i++) data[i] = v[i]; break; }
    case 8:   { const v = new Int32Array(imageBuffer);   for (let i = 0; i < total; i++) data[i] = v[i]; break; }
    case 16:  { const v = new Float32Array(imageBuffer); for (let i = 0; i < total; i++) data[i] = v[i]; break; }
    case 64:  { const v = new Float64Array(imageBuffer); for (let i = 0; i < total; i++) data[i] = v[i]; break; }
    default:  { const v = new Int16Array(imageBuffer);   for (let i = 0; i < total; i++) data[i] = v[i]; break; }
  }

  // Cache percentiles alongside the volume — sorting 8.9M floats is expensive
  const p2 = percentile(data, 2), p98 = percentile(data, 98);

  if (volumeCache.size >= MAX_CACHED) volumeCache.delete(volumeCache.keys().next().value!);
  volumeCache.set(filePath, { data, nx, ny, nz, p2, p98 });
  return { data, nx, ny, nz, p2, p98 };
}

function computeSliceNMI(
  volA: Float32Array, volB: Float32Array,
  nx: number, ny: number, z: number,
  p2A: number, rangeA: number, p2B: number, rangeB: number,
): { joint: number[]; nmi: number } {
  const sliceSize = nx * ny;
  const offset = z * sliceSize;

  const joint = new Array<number>(BINS * BINS).fill(0);

  for (let i = 0; i < sliceSize; i++) {
    const a = Math.min(BINS - 1, Math.max(0, Math.floor(((volA[offset + i] - p2A) / rangeA) * BINS)));
    const b = Math.min(BINS - 1, Math.max(0, Math.floor(((volB[offset + i] - p2B) / rangeB) * BINS)));
    joint[a * BINS + b]++;
  }

  // Marginals
  const margA = new Float64Array(BINS);
  const margB = new Float64Array(BINS);
  for (let a = 0; a < BINS; a++) {
    for (let b = 0; b < BINS; b++) {
      margA[a] += joint[a * BINS + b];
      margB[b] += joint[a * BINS + b];
    }
  }

  const total = sliceSize;
  let hA = 0, hB = 0, hAB = 0;
  for (let a = 0; a < BINS; a++) {
    if (margA[a] > 0) { const p = margA[a] / total; hA -= p * Math.log2(p); }
  }
  for (let b = 0; b < BINS; b++) {
    if (margB[b] > 0) { const p = margB[b] / total; hB -= p * Math.log2(p); }
  }
  for (let k = 0; k < BINS * BINS; k++) {
    if (joint[k] > 0) { const p = joint[k] / total; hAB -= p * Math.log2(p); }
  }

  const denom = hA + hB;
  const nmi = denom > 0 ? (2 * (hA + hB - hAB)) / denom : 0;
  return { joint, nmi };
}

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;
  const rel1 = searchParams.get('path1');
  const rel2 = searchParams.get('path2');
  const zStr = searchParams.get('z');
  const sparklineOnly = searchParams.get('sparklineOnly') === 'true';

  if (!rel1 || !rel2) return NextResponse.json({ error: 'path1 and path2 required' }, { status: 400 });

  const p1 = resolvePath(rel1);
  const p2 = resolvePath(rel2);
  if (!p1) return NextResponse.json({ error: 'path1 not found' }, { status: 404 });
  if (!p2) return NextResponse.json({ error: 'path2 not found' }, { status: 404 });

  const volA = readVolume(p1);
  const volB = readVolume(p2);
  if (!volA || !volB) return NextResponse.json({ error: 'Failed to read NIfTI' }, { status: 422 });

  const { nx, ny, nz } = volA;

  // Use cached percentiles — avoids re-sorting 8.9M floats on every request
  const rangeA = (volA.p98 - volA.p2) || 1;
  const rangeB = (volB.p98 - volB.p2) || 1;

  const sparkline: number[] = [];

  if (sparklineOnly) {
    // Fast path: skip building allJoints (saves ~1.3 MB per response)
    for (let z = 0; z < nz; z++) {
      const { nmi } = computeSliceNMI(volA.data, volB.data, nx, ny, z, volA.p2, rangeA, volB.p2, rangeB);
      sparkline.push(nmi);
    }
    const z = zStr != null ? Math.max(0, Math.min(parseInt(zStr, 10), nz - 1)) : Math.floor(nz / 2);
    return NextResponse.json({ sparkline, nmi: sparkline[z], nz, z });
  }

  // Full path: sparkline + all joint histograms for the joint-histogram canvas
  // Flat Uint16Array: [z0_hist(BINS*BINS), z1_hist(BINS*BINS), ...]
  // Counts fit in uint16 (max = nx*ny = 57600 < 65535)
  const allJoints = new Uint16Array(nz * BINS * BINS);

  for (let z = 0; z < nz; z++) {
    const { joint, nmi } = computeSliceNMI(volA.data, volB.data, nx, ny, z, volA.p2, rangeA, volB.p2, rangeB);
    sparkline.push(nmi);
    allJoints.set(joint, z * BINS * BINS);
  }

  const z = zStr != null ? Math.max(0, Math.min(parseInt(zStr, 10), nz - 1)) : Math.floor(nz / 2);

  return NextResponse.json({
    sparkline,
    bins: BINS,
    nmi: sparkline[z],
    nz,
    z,
    allJoints: Buffer.from(allJoints.buffer).toString('base64'),
  });
}
