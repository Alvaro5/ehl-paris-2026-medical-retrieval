import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import * as nifti from 'nifti-reader-js';

const GALLERY_DIR =
  'C:\\Users\\laure\\Downloads\\ehl-paris-medical-image-retrieval\\dataset1\\images\\train\\gallery';

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;
  const fileName = searchParams.get('file');
  const zStr = searchParams.get('z');

  if (!fileName) {
    return NextResponse.json({ error: 'No file specified' }, { status: 400 });
  }

  // Only allow bare filenames with .nii or .nii.gz
  if (!/^[\w.-]+\.nii(\.gz)?$/.test(fileName)) {
    return NextResponse.json({ error: 'Invalid filename' }, { status: 400 });
  }

  const filePath = path.join(GALLERY_DIR, fileName);
  if (!fs.existsSync(filePath)) {
    return NextResponse.json({ error: 'File not found' }, { status: 404 });
  }

  const buf = fs.readFileSync(filePath);
  let data: ArrayBuffer = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);

  if (nifti.isCompressed(data)) {
    data = nifti.decompress(data) as ArrayBuffer;
  }

  if (!nifti.isNIFTI(data)) {
    return NextResponse.json({ error: 'Not a valid NIfTI file' }, { status: 422 });
  }

  const header = nifti.readHeader(data);
  const imageBuffer = nifti.readImage(header, data);

  const nx = header.dims[1];
  const ny = header.dims[2];
  const nz = header.dims[3];
  const sliceSize = nx * ny;

  const z = zStr != null ? Math.max(0, Math.min(parseInt(zStr, 10), nz - 1)) : Math.floor(nz / 2);

  const raw = new Float32Array(sliceSize);

  switch (header.datatypeCode) {
    case 2:   { const v = new Uint8Array(imageBuffer);   for (let i = 0; i < sliceSize; i++) raw[i] = v[z * sliceSize + i]; break; }
    case 256: { const v = new Int8Array(imageBuffer);    for (let i = 0; i < sliceSize; i++) raw[i] = v[z * sliceSize + i]; break; }
    case 4:   { const v = new Int16Array(imageBuffer);   for (let i = 0; i < sliceSize; i++) raw[i] = v[z * sliceSize + i]; break; }
    case 512: { const v = new Uint16Array(imageBuffer);  for (let i = 0; i < sliceSize; i++) raw[i] = v[z * sliceSize + i]; break; }
    case 8:   { const v = new Int32Array(imageBuffer);   for (let i = 0; i < sliceSize; i++) raw[i] = v[z * sliceSize + i]; break; }
    case 16:  { const v = new Float32Array(imageBuffer); for (let i = 0; i < sliceSize; i++) raw[i] = v[z * sliceSize + i]; break; }
    case 64:  { const v = new Float64Array(imageBuffer); for (let i = 0; i < sliceSize; i++) raw[i] = v[z * sliceSize + i]; break; }
    default:  { const v = new Int16Array(imageBuffer);   for (let i = 0; i < sliceSize; i++) raw[i] = v[z * sliceSize + i]; break; }
  }

  // Percentile-based windowing (p2–p98) for better contrast
  const sorted = Array.from(raw).sort((a, b) => a - b);
  const p2  = sorted[Math.floor(sorted.length * 0.02)];
  const p98 = sorted[Math.floor(sorted.length * 0.98)];
  const range = p98 - p2 || 1;

  const pixels = new Uint8Array(sliceSize);
  for (let i = 0; i < sliceSize; i++) {
    pixels[i] = Math.max(0, Math.min(255, Math.round(((raw[i] - p2) / range) * 255)));
  }

  return NextResponse.json({
    nx,
    ny,
    nz,
    z,
    pixels: Array.from(pixels),
    windowMin: p2,
    windowMax: p98,
    datatypeCode: header.datatypeCode,
  });
}
