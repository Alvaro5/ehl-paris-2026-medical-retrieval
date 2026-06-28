import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import * as nifti from 'nifti-reader-js';

const DATA_ROOT =
  'C:\\Users\\laure\\Projects\\ehl2026\\ehl-paris-2026-medical-retrieval\\data\\ehl-paris-medical-image-retrieval';

interface CachedVolume { data: Float32Array; nx: number; ny: number; nz: number; }
const MAX_CACHED = 10;
const volumeCache = new Map<string, CachedVolume>();

function loadVolume(filePath: string): CachedVolume | null {
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

  if (volumeCache.size >= MAX_CACHED) volumeCache.delete(volumeCache.keys().next().value!);
  volumeCache.set(filePath, { data, nx, ny, nz });
  return { data, nx, ny, nz };
}

function resolveSafe(rel: string): string | null {
  const resolved = path.resolve(DATA_ROOT, rel);
  if (!resolved.startsWith(path.resolve(DATA_ROOT))) return null;
  if (fs.existsSync(resolved)) return resolved;
  if (resolved.endsWith('.gz') && fs.existsSync(resolved.slice(0, -3))) return resolved.slice(0, -3);
  return null;
}

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;
  const relPath = searchParams.get('path');
  if (!relPath) return NextResponse.json({ error: 'No path' }, { status: 400 });

  const filePath = resolveSafe(relPath);
  if (!filePath) return NextResponse.json({ error: 'File not found' }, { status: 404 });

  const vol = loadVolume(filePath);
  if (!vol) return NextResponse.json({ error: 'Not a valid NIfTI file' }, { status: 422 });

  const { data, nx, ny, nz } = vol;
  const sliceSize = nx * ny;
  const allFrames = new Uint8Array(nz * sliceSize);

  for (let z = 0; z < nz; z++) {
    const slice = data.subarray(z * sliceSize, (z + 1) * sliceSize);
    const sorted = Float32Array.from(slice).sort();
    const p2  = sorted[Math.floor(sorted.length * 0.02)];
    const p98 = sorted[Math.floor(sorted.length * 0.98)];
    const range = p98 - p2 || 1;
    for (let i = 0; i < sliceSize; i++) {
      allFrames[z * sliceSize + i] = Math.max(0, Math.min(255, Math.round(((slice[i] - p2) / range) * 255)));
    }
  }

  return NextResponse.json({
    nz, nx, ny,
    frames: Buffer.from(allFrames).toString('base64'),
  });
}
