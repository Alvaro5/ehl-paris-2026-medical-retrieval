declare module 'gifenc' {
  export interface GIFEncoderInstance {
    writeFrame(index: Uint8Array, width: number, height: number, opts?: {
      palette?: number[][];
      delay?: number;
      repeat?: number;
      transparent?: boolean;
      transparentIndex?: number;
      colorDepth?: number;
      dispose?: number;
    }): void;
    finish(): void;
    bytes(): Uint8Array;
    bytesView(): Uint8Array;
    reset(): void;
  }
  export function GIFEncoder(opts?: { initialCapacity?: number; auto?: boolean }): GIFEncoderInstance;
  export function quantize(rgba: Uint8Array, maxColors: number, opts?: { format?: string; oneBitAlpha?: boolean }): number[][];
  export function applyPalette(rgba: Uint8Array, palette: number[][], format?: string): Uint8Array;
}
