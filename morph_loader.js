// Loader for morph.bin (version 2): [u32 header length][header JSON][gzip blobs at header.offsets].
// Triangulations after the first are rebuilt by replaying each remesh round (edge collapses, then edge splits);
// frames at remesh switches are derived exactly as in the packager, so what is shown is what was certified.
async function gunzipBytes(u8) {
  const stream = new Blob([u8]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function collapseApply(F, pairs, nv) {
  const remap = new Int32Array(nv); for (let i = 0; i < nv; i++) remap[i] = i;
  for (let i = 0; i < pairs.length; i += 2) remap[pairs[i + 1]] = pairs[i];
  const kept = [];
  for (let t = 0; t < F.length; t += 3) {
    const a = remap[F[t]], b = remap[F[t + 1]], c = remap[F[t + 2]];
    if (a !== b && b !== c && c !== a) kept.push(a, b, c);
  }
  const used = new Uint8Array(nv); for (const v of kept) used[v] = 1;
  const newid = new Int32Array(nv).fill(-1); let n = 0;
  for (let i = 0; i < nv; i++) if (used[i]) newid[i] = n++;
  return { F: Int32Array.from(kept, v => newid[v]), newid, n };
}

function splitApply(F, picks, nv) {
  const Fl = []; for (let t = 0; t < F.length; t += 3) Fl.push([F[t], F[t + 1], F[t + 2]]);
  const et = new Map();
  Fl.forEach((tri, t) => { for (let i = 0; i < 3; i++) { const a = tri[i], b = tri[(i + 1) % 3], k = Math.min(a, b) + ',' + Math.max(a, b);
    if (!et.has(k)) et.set(k, []); et.get(k).push(t); } });
  let n = nv;
  for (let p = 0; p < picks.length; p += 2) {
    const a = picks[p], b = picks[p + 1], m = n++;
    for (const t of et.get(Math.min(a, b) + ',' + Math.max(a, b)).slice().sort((x, y) => x - y)) {
      const tri = Fl[t]; let i = 0;
      for (let k = 0; k < 3; k++) if ((tri[k] === a || tri[k] === b) && (tri[(k + 1) % 3] === a || tri[(k + 1) % 3] === b)) { i = k; break; }
      const P = tri[i], Q = tri[(i + 1) % 3], R = tri[(i + 2) % 3];
      Fl[t] = [P, m, R]; Fl.push([m, Q, R]);
    }
  }
  return { F: Int32Array.from(Fl.flat()), n };
}

function deriveNextFirst(Xc, F, pairs, picks, nvOld) {
  const { newid, n } = collapseApply(F, pairs, nvOld);
  const X = new Float64Array((n + picks.length / 2) * 3);
  for (let i = 0; i < nvOld; i++) if (newid[i] >= 0) for (let k = 0; k < 3; k++) X[3 * newid[i] + k] = Xc[3 * i + k];
  for (let p = 0; p < picks.length; p += 2) { const m = n + p / 2, a = picks[p], b = picks[p + 1];
    for (let k = 0; k < 3; k++) X[3 * m + k] = 0.5 * (X[3 * a + k] + X[3 * b + k]); }
  return X;
}

async function parseMorph(buf, onProgress) {
  const dv = new DataView(buf), hl = dv.getUint32(0, true);
  const H = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
  const base = 4 + hl, blob = async i => gunzipBytes(new Uint8Array(buf, base + H.offsets[i], H.offsets[i + 1] - H.offsets[i]));
  const S = H.scale, qi = x => Math.floor(x / S + 0.5), epochs = [];
  let F = new Int32Array((await blob(H.epochs[0].faces)).buffer), prev = null;
  for (let k = 0; k < H.epochs.length; k++) {
    if (onProgress && k % 4 === 0) { onProgress(k, H.epochs.length); await new Promise(r => setTimeout(r, 0)); }   // let the bar repaint
    const h = H.epochs[k], nv = h.nv, n3 = nv * 3;
    const L = await blob(h.labels);
    const pairs = new Uint32Array((await blob(h.pairs)).slice().buffer), picks = new Uint32Array((await blob(h.picks)).slice().buffer);
    const raw = await blob(h.frames), half = raw.length / 2;
    const disp = new Float64Array(h.nframes * n3);
    let pos = 0;
    const next = () => { const z = raw[pos] | (raw[half + pos] << 8); pos++; return (z & 1) ? -((z + 1) >> 1) : (z >> 1); };
    const stored = new Set(h.stored);
    for (let i = 0; i < h.nframes; i++) {
      const o = i * n3;
      if (i === 0 && prev) { disp.set(prev, 0); continue; }
      if (stored.has(i)) {
        for (let q = 0; q < n3; q++) disp[o + q] = (i > 0 ? qi(disp[o - n3 + q]) : 0) * S + next() * S;
        // exact integer arithmetic on the quantized grid
        for (let q = 0; q < n3; q++) disp[o + q] = Math.round(disp[o + q] / S) * S;
      } else {   // collapse end: previous frame with each collapsed pair at its midpoint
        disp.set(disp.subarray(o - n3, o), o);
        for (let p = 0; p < pairs.length; p += 2) for (let c = 0; c < 3; c++) {
          const a = pairs[p], b = pairs[p + 1], mid = 0.5 * (disp[o - n3 + 3 * a + c] + disp[o - n3 + 3 * b + c]);
          disp[o + 3 * a + c] = mid; disp[o + 3 * b + c] = mid;
        }
      }
    }
    epochs.push({ F: Uint32Array.from(F), L, frames: Float32Array.from(disp), disp, nv, nf: h.nframes, start: h.start });
    if (k + 1 < H.epochs.length) {
      const last = disp.subarray((h.nframes - 1) * n3);
      prev = deriveNextFirst(last, F, pairs, picks, nv);
      const c = collapseApply(F, pairs, nv); F = splitApply(c.F, picks, c.n).F;
    }
  }
  return { epochs, nf: H.total, stages: H.stages, times: H.times };
}
if (typeof module !== 'undefined') module.exports = { parseMorph, collapseApply, splitApply };
