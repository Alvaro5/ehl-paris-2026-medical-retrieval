import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

const DATA_ROOT =
  'C:\\Users\\laure\\Projects\\ehl2026\\ehl-paris-2026-medical-retrieval\\data\\ehl-paris-medical-image-retrieval';

function parseCSV(filePath: string): Record<string, string>[] {
  if (!fs.existsSync(filePath)) return [];
  const lines = fs.readFileSync(filePath, 'utf8').trim().split('\n');
  const headers = lines[0].split(',');
  return lines.slice(1).map((line) => {
    const vals = line.split(',');
    return Object.fromEntries(headers.map((h, i) => [h.trim(), (vals[i] ?? '').trim()]));
  });
}

export async function GET() {
  const datasets = ['dataset1', 'dataset2', 'dataset3'] as const;

  const result: Record<string, Record<string, Record<string, string>[]>> = {};

  for (const ds of datasets) {
    const dir = path.join(DATA_ROOT, ds);
    result[ds] = {};

    const splits = [
      'train_pairs',
      'val_queries',
      'val_gallery',
      'test_queries',
      'test_gallery',
    ];

    for (const split of splits) {
      const rows = parseCSV(path.join(dir, `${split}.csv`));
      if (rows.length > 0) result[ds][split] = rows;
    }
  }

  return NextResponse.json(result);
}
