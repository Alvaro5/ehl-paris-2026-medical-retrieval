import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

const DATA_ROOT =
  'C:\\Users\\laure\\Projects\\ehl2026\\ehl-paris-2026-medical-retrieval\\data\\ehl-paris-medical-image-retrieval';

const MODAL_URL =
  'https://ehl-2026-hack--ehl-dinov2-serve-get-embedding.modal.run';

function sidecarPath(niiRel: string): string | null {
  // Security check only — don't require the NIfTI to exist locally
  const resolved = path.resolve(DATA_ROOT, niiRel);
  if (!resolved.startsWith(path.resolve(DATA_ROOT))) return null;

  // Strip .gz then .nii to get bare stem
  const noGz   = resolved.endsWith('.gz') ? resolved.slice(0, -3) : resolved;
  const stem    = path.basename(noGz, '.nii');
  return path.join(path.dirname(noGz), `${stem}_dinov2.f32`);
}

function parseF32(buf: Buffer): { nz: number; dim: number; data: string } {
  const nz  = buf.readUInt32LE(0);
  const dim = buf.readUInt32LE(4);
  return { nz, dim, data: buf.subarray(8).toString('base64') };
}

export async function GET(request: NextRequest) {
  const relPath = request.nextUrl.searchParams.get('path');
  if (!relPath) return NextResponse.json({ error: 'No path' }, { status: 400 });

  // 1. Try local disk first
  const f32Path = sidecarPath(relPath);
  if (f32Path && fs.existsSync(f32Path)) {
    const buf = fs.readFileSync(f32Path);
    return NextResponse.json(parseF32(buf));
  }

  // 2. Fall back to Modal Volume endpoint
  // Normalise path: strip leading dataset*/ if already absolute-looking
  const modalRes = await fetch(`${MODAL_URL}?path=${encodeURIComponent(relPath)}`, {
    signal: AbortSignal.timeout(15_000),
  });
  if (!modalRes.ok) {
    return NextResponse.json({ error: 'Embeddings not found locally or on Modal Volume' }, { status: 404 });
  }
  const arrayBuf = await modalRes.arrayBuffer();
  const buf = Buffer.from(arrayBuf);

  // Cache to local disk so subsequent requests are instant
  if (f32Path) {
    fs.mkdirSync(path.dirname(f32Path), { recursive: true });
    fs.writeFileSync(f32Path, buf);
  }

  return NextResponse.json(parseF32(buf));
}
