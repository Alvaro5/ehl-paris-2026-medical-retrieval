import { NextResponse } from 'next/server';
import fs from 'fs';

const GALLERY_DIR =
  'C:\\Users\\laure\\Downloads\\ehl-paris-medical-image-retrieval\\dataset1\\images\\train\\gallery';

export async function GET() {
  const files = fs
    .readdirSync(GALLERY_DIR)
    .filter((f) => f.endsWith('.nii') || f.endsWith('.nii.gz'))
    .sort();
  return NextResponse.json({ files });
}
