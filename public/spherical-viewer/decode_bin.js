/**
 * Decode USF spherical `.bin` bytes (same layout as `usf.utils.spherical_image._decode_batch_spherical_bin`).
 *
 * Header: 64 bytes — magic `USFBSIMG`, version, `N` (u64), `B` and `C` (u32), four dtype codes (u8).
 * Payloads: `batch_value` (B×N×C), `vector` (N×3), `polar` (N×2), `mask` (N), all little-endian.
 */

const HEADER_SIZE = 64;
const MAGIC_STR = "USFBSIMG";

/** @type {Record<number, number>} dtype code -> bytes per element (matches `BINARY_DTYPE_CODE_TO_NUMPY`) */
const DTYPE_BYTES = {
  1: 4, // float32
  2: 8, // float64
  3: 1, // uint8
  4: 2, // uint16
  5: 4, // int32
  6: 8, // int64
  7: 1, // bool (numpy bool_ single byte)
};

/**
 * @param {number} code
 * @returns {number}
 */
function dtypeBytes(code) {
  const b = DTYPE_BYTES[code];
  if (b === undefined) throw new Error(`Unknown dtype code ${code}`);
  return b;
}

/**
 * @param {ArrayBuffer} buffer
 * @param {number} byteOffset
 * @param {number} count Elements (not bytes)
 * @param {number} code
 * @returns {Float32Array}
 */
function toFloat32Array(buffer, byteOffset, count, code) {
  const out = new Float32Array(count);
  const dv = new DataView(buffer, byteOffset);
  let o = 0;
  let off = 0;
  switch (code) {
    case 1: {
      const u8 = new Uint8Array(buffer, byteOffset, count * 4);
      const copy = new ArrayBuffer(count * 4);
      new Uint8Array(copy).set(u8);
      out.set(new Float32Array(copy));
      break;
    }
    case 2: {
      for (let i = 0; i < count; i++, off += 8) {
        out[o++] = dv.getFloat64(off, true);
      }
      break;
    }
    case 3: {
      for (let i = 0; i < count; i++, off += 1) {
        out[o++] = dv.getUint8(off);
      }
      break;
    }
    case 4: {
      for (let i = 0; i < count; i++, off += 2) {
        out[o++] = dv.getUint16(off, true);
      }
      break;
    }
    case 5: {
      for (let i = 0; i < count; i++, off += 4) {
        out[o++] = dv.getInt32(off, true);
      }
      break;
    }
    case 6: {
      for (let i = 0; i < count; i++, off += 8) {
        out[o++] = Number(dv.getBigInt64(off, true));
      }
      break;
    }
    case 7: {
      for (let i = 0; i < count; i++, off += 1) {
        out[o++] = dv.getUint8(off) !== 0 ? 1 : 0;
      }
      break;
    }
    default:
      throw new Error(`Unknown dtype code ${code}`);
  }
  return out;
}

/**
 * @param {ArrayBuffer} buffer
 * @returns {{
 *   n: number,
 *   b: number,
 *   c: number,
 *   batchValue: Float32Array,
 *   vector: Float32Array,
 *   polar: Float32Array,
 *   mask: Uint8Array,
 * }}
 */
export function decodeSphericalBin(buffer) {
  if (buffer.byteLength < HEADER_SIZE) {
    throw new Error("File too small for spherical .bin header.");
  }
  const u8 = new Uint8Array(buffer, 0, 8);
  let magic = "";
  for (let i = 0; i < 8; i++) magic += String.fromCharCode(u8[i]);
  if (magic !== MAGIC_STR) {
    throw new Error(`Bad magic: expected ${MAGIC_STR}, got ${magic}.`);
  }
  const dv = new DataView(buffer);
  const version = dv.getUint32(8, true);
  if (version !== 1) {
    throw new Error(
      `Unsupported spherical .bin version ${version} (expected 1).`,
    );
  }
  const n = Number(dv.getBigUint64(16, true));
  const b = dv.getUint32(24, true);
  const c = dv.getUint32(28, true);
  const dBv = dv.getUint8(32);
  const dV = dv.getUint8(33);
  const dP = dv.getUint8(34);
  const dM = dv.getUint8(35);

  for (const code of [dBv, dV, dP, dM]) {
    if (DTYPE_BYTES[code] === undefined) {
      throw new Error(`Unknown dtype code ${code} in spherical .bin.`);
    }
  }

  const esBv = dtypeBytes(dBv);
  const esV = dtypeBytes(dV);
  const esP = dtypeBytes(dP);
  const esM = dtypeBytes(dM);
  const pay = b * n * c * esBv + n * 3 * esV + n * 2 * esP + n * esM;
  if (buffer.byteLength !== HEADER_SIZE + pay) {
    throw new Error(
      `File size mismatch: expected ${HEADER_SIZE + pay} bytes, got ${buffer.byteLength}.`,
    );
  }

  let off = HEADER_SIZE;
  const countBv = b * n * c;
  let batchValue = toFloat32Array(buffer, off, countBv, dBv);
  off += countBv * esBv;

  const countV = n * 3;
  const vector = toFloat32Array(buffer, off, countV, dV);
  off += countV * esV;

  const countP = n * 2;
  const polar = toFloat32Array(buffer, off, countP, dP);
  off += countP * esP;

  const maskFloat = toFloat32Array(buffer, off, n, dM);
  off += n * esM;
  const mask = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    mask[i] = maskFloat[i] !== 0 ? 1 : 0;
  }

  let maxVal = 0;
  for (let i = 0; i < batchValue.length; i++) {
    maxVal = Math.max(maxVal, batchValue[i]);
  }
  if (maxVal > 1.0) {
    for (let i = 0; i < batchValue.length; i++) {
      batchValue[i] /= 255.0;
    }
  }

  return {
    n,
    b,
    c,
    batchValue,
    vector,
    polar,
    mask,
  };
}
