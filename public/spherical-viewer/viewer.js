/**
 * Three.js point-cloud viewer for USF spherical `.bin` files.
 *
 * Mirrors the behavior of `visualize_spherical_image` in
 * `usf/visualization/spherical_projection.py`: unit sphere + colored points,
 * batch animation at `fps`, orbit camera outside the sphere looking at center.
 *
 * Designed for iframe embedding inside carousels (e.g. the Astro academic
 * project template's <Carousel> component). Pauses rendering when off-screen,
 * supports dark/light themes, and accepts postMessage control from the parent.
 */

import * as THREE from "three";
import { ArcballControls } from "three/addons/controls/ArcballControls.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TrackballControls } from "three/addons/controls/TrackballControls.js";
import { decodeSphericalBin } from "./decode_bin.js";

const _autoRotateQuat = new THREE.Quaternion();
const _autoRotateOffset = new THREE.Vector3();

// ---------------------------------------------------------------------------
// USF → Three.js coordinate mapping
// USF:    x forward, y left, z up   (FLU, right-handed)
// Three:  x right,   y up,   z back (RUB, right-handed)
// ---------------------------------------------------------------------------

/**
 * @param {number} xf  USF x (forward)
 * @param {number} yl  USF y (left)
 * @param {number} zu  USF z (up)
 * @param {THREE.Vector3} [out]
 * @returns {THREE.Vector3}
 */
export function usfToThree(xf, yl, zu, out = new THREE.Vector3()) {
  return out.set(-yl, zu, -xf);
}

// ---------------------------------------------------------------------------
// Theme presets (light / dark)
// ---------------------------------------------------------------------------

const THEMES = {
  light: { bg: [0.88, 0.88, 0.89], mesh: [0.9, 0.9, 0.9] },
  dark: { bg: [0.12, 0.12, 0.14], mesh: [0.2, 0.2, 0.22] },
};

function resolveTheme(raw) {
  if (raw === "light" || raw === "dark") return raw;
  if (typeof window.matchMedia === "function") {
    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }
  return "light";
}

// ---------------------------------------------------------------------------
// Query-string parameters
// ---------------------------------------------------------------------------

function parseVec3Param(raw, dx, dy, dz) {
  if (!raw) return [dx, dy, dz];
  const p = raw.split(",").map((s) => parseFloat(s.trim()));
  return p.length === 3 && p.every(Number.isFinite) ? p : [dx, dy, dz];
}

function parseColorParam(raw, dr, dg, db) {
  if (!raw) return new THREE.Color(dr, dg, db);
  const p = raw.split(",").map((s) => parseFloat(s.trim()));
  return p.length === 3 && p.every(Number.isFinite)
    ? new THREE.Color(p[0], p[1], p[2])
    : new THREE.Color(dr, dg, db);
}

function parseParams(qs) {
  const url = qs.get("url") || "";

  let fps = 1;
  if (qs.has("fps") && qs.get("fps") !== "") {
    const v = parseFloat(qs.get("fps"));
    if (Number.isFinite(v)) fps = Math.max(0, v);
  }

  const pointSize = Math.max(0.5, parseFloat(qs.get("pointSize") || "2") || 2);

  const loop = !["0", "false", "no"].includes(
    (qs.get("loop") || "true").toLowerCase(),
  );

  const fov = Math.max(
    10,
    Math.min(120, parseFloat(qs.get("fov") || "50") || 50),
  );

  const theme = resolveTheme(qs.get("theme") || "auto");
  const defaults = THEMES[theme];

  const meshColor = parseColorParam(qs.get("meshColor"), ...defaults.mesh);
  const bgColor = parseColorParam(qs.get("bgColor"), ...defaults.bg);

  let autoRotateSpeed = 0.4;
  if (qs.has("autoRotateSpeed") && qs.get("autoRotateSpeed") !== "") {
    const v = parseFloat(qs.get("autoRotateSpeed"));
    if (Number.isFinite(v)) autoRotateSpeed = v;
  }

  const enablePan = !["0", "false", "no"].includes(
    (qs.get("enablePan") || "false").toLowerCase(),
  );

  let controlsMode = (qs.get("controls") || "trackball").toLowerCase();
  if (!["orbit", "trackball", "arcball"].includes(controlsMode)) {
    controlsMode = "trackball";
  }

  const lookFrom = parseVec3Param(qs.get("lookFrom"), 2.4, -0.6, 1.6);
  const spinAxis = parseVec3Param(qs.get("spinAxis"), 0, 0, 1);

  let zoom = 1.0;
  if (qs.has("zoom") && qs.get("zoom") !== "") {
    const v = parseFloat(qs.get("zoom"));
    if (Number.isFinite(v) && v > 0) zoom = v;
  }

  let projection = (qs.get("projection") || "perspective").toLowerCase();
  if (!["perspective", "ortho"].includes(projection)) {
    projection = "perspective";
  }

  return {
    url,
    fps,
    pointSize,
    loop,
    fov,
    meshColor,
    bgColor,
    autoRotateSpeed,
    enablePan,
    theme,
    controlsMode,
    lookFrom,
    spinAxis,
    zoom,
    projection,
  };
}

// ---------------------------------------------------------------------------
// Per-frame color fill (batch_value → vertex colors)
// ---------------------------------------------------------------------------

function fillFrameColors(decoded, frameIdx, validIndices, outRgb) {
  const { n, c, batchValue, b: B } = decoded;
  const f = Math.floor(frameIdx) % B;
  for (let j = 0; j < validIndices.length; j++) {
    const i = validIndices[j];
    const src = f * n * c + i * c;
    const dst = j * 3;
    if (c === 1) {
      outRgb[dst] = outRgb[dst + 1] = outRgb[dst + 2] = batchValue[src];
    } else {
      outRgb[dst] = batchValue[src];
      outRgb[dst + 1] = batchValue[src + 1];
      outRgb[dst + 2] = batchValue[src + 2];
    }
  }
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

function showError(container, msg) {
  container.innerHTML = "";
  const el = document.createElement("div");
  el.className = "usf-viewer-error";
  el.textContent = msg;
  container.appendChild(el);
}

function handSVG(dark) {
  const fg = dark ? "#fff" : "#000";
  const bg = dark ? "#000" : "#fff";
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="25" height="36" viewBox="0 0 25 36">` +
    `<defs><path id="A" d="M.001.232h24.997V36H.001z"/></defs>` +
    `<g fill="none" fill-rule="evenodd">` +
    `<path d="M8.733 11.165c.04-1.108.766-2.027 1.743-2.307a2.54 2.54 0 0 1 .628-.089c.16 0 .314.017.463.044 1.088.2 1.9 1.092 1.9 2.16v8.88h1.26c2.943-1.39 5-4.45 5-8.025a9.01 9.01 0 0 0-1.9-5.56l-.43-.5c-.765-.838-1.683-1.522-2.712-2-1.057-.49-2.226-.77-3.46-.77s-2.4.278-3.46.77c-1.03.478-1.947 1.162-2.71 2l-.43.5a9.01 9.01 0 0 0-1.9 5.56 9.04 9.04 0 0 0 .094 1.305c.03.21.088.41.13.617l.136.624c.083.286.196.56.305.832l.124.333a8.78 8.78 0 0 0 .509.953l.065.122a8.69 8.69 0 0 0 3.521 3.191l1.11.537v-9.178z" fill-opacity=".5" fill="#e4e4e4"/>` +
    `<path d="M22.94 26.218l-2.76 7.74c-.172.485-.676.8-1.253.8H12.24c-1.606 0-3.092-.68-3.98-1.82-1.592-2.048-3.647-3.822-6.11-5.27-.095-.055-.15-.137-.152-.23-.004-.1.046-.196.193-.297.56-.393 1.234-.6 1.926-.6a3.43 3.43 0 0 1 .691.069l4.922.994V10.972c0-.663.615-1.203 1.37-1.203s1.373.54 1.373 1.203v9.882h2.953c.273 0 .533.073.757.21l6.257 3.874c.027.017.045.042.07.06.41.296.586.77.426 1.22M4.1 16.614a8.69 8.69 0 0 1-.574-1.075c-.048-.107-.08-.223-.124-.333l-.305-.832-.136-.624-.13-.617a9.03 9.03 0 0 1-.094-1.305c0-2.107.714-4.04 1.9-5.56l.43-.5c.764-.84 1.682-1.523 2.71-2C8.835 2.278 10.003 2 11.237 2s2.402.28 3.46.77c1.03.477 1.947 1.16 2.712 2l.428.5a9 9 0 0 1 1.901 5.559c0 3.577-2.056 6.636-5 8.026h-1.26v-8.882c0-1.067-.822-1.96-1.9-2.16-.15-.028-.304-.044-.463-.044-.22 0-.427.037-.628.09-.977.28-1.703 1.198-1.743 2.306v9.178l-1.11-.537C6.18 19.098 4.96 18 4.1 16.614M22.97 24.09l-6.256-3.874c-.102-.063-.218-.098-.33-.144 2.683-1.8 4.354-4.855 4.354-8.243 0-.486-.037-.964-.104-1.43a9.97 9.97 0 0 0-1.57-4.128l-.295-.408-.066-.092a10.05 10.05 0 0 0-.949-1.078c-.342-.334-.708-.643-1.094-.922-1.155-.834-2.492-1.412-3.94-1.65l-.732-.088-.748-.03a9.29 9.29 0 0 0-1.482.119c-1.447.238-2.786.816-3.94 1.65a9.33 9.33 0 0 0-.813.686 9.59 9.59 0 0 0-.845.877l-.385.437-.36.5-.288.468-.418.778-.04.09c-.593 1.28-.93 2.71-.93 4.222 0 3.832 2.182 7.342 5.56 8.938l1.437.68v4.946L5 25.64a4.44 4.44 0 0 0-.888-.086c-.017 0-.034.003-.05.003-.252.004-.503.033-.75.08a5.08 5.08 0 0 0-.237.056c-.193.046-.382.107-.568.18-.075.03-.15.057-.225.1-.25.114-.494.244-.723.405a1.31 1.31 0 0 0-.566 1.122 1.28 1.28 0 0 0 .645 1.051C4 29.925 5.96 31.614 7.473 33.563a5.06 5.06 0 0 0 .434.491c1.086 1.082 2.656 1.713 4.326 1.715h6.697c.748-.001 1.43-.333 1.858-.872.142-.18.256-.38.336-.602l2.757-7.74c.094-.26.13-.53.112-.794s-.088-.52-.203-.76a2.19 2.19 0 0 0-.821-.91" fill-opacity=".6" fill="${fg}"/>` +
    `<path d="M22.444 24.94l-6.257-3.874a1.45 1.45 0 0 0-.757-.211h-2.953v-9.88c0-.663-.616-1.203-1.373-1.203s-1.37.54-1.37 1.203v16.643l-4.922-.994a3.44 3.44 0 0 0-.692-.069 3.35 3.35 0 0 0-1.925.598c-.147.102-.198.198-.194.298.004.094.058.176.153.23 2.462 1.448 4.517 3.22 6.11 5.27.887 1.14 2.373 1.82 3.98 1.82h6.686c.577 0 1.08-.326 1.253-.8l2.76-7.74c.16-.448-.017-.923-.426-1.22-.025-.02-.043-.043-.07-.06z" fill="${bg}"/>` +
    `<g transform="translate(0 .769)"><mask id="B" fill="#fff"><use xlink:href="#A"/></mask>` +
    `<path d="M23.993 24.992a1.96 1.96 0 0 1-.111.794l-2.758 7.74c-.08.22-.194.423-.336.602-.427.54-1.11.87-1.857.872h-6.698c-1.67-.002-3.24-.633-4.326-1.715-.154-.154-.3-.318-.434-.49C5.96 30.846 4 29.157 1.646 27.773c-.385-.225-.626-.618-.645-1.05a1.31 1.31 0 0 1 .566-1.122 4.56 4.56 0 0 1 .723-.405l.225-.1a4.3 4.3 0 0 1 .568-.18l.237-.056c.248-.046.5-.075.75-.08.018 0 .034-.003.05-.003.303-.001.597.027.89.086l3.722.752V20.68l-1.436-.68c-3.377-1.596-5.56-5.106-5.56-8.938 0-1.51.336-2.94.93-4.222.015-.03.025-.06.04-.09.127-.267.268-.525.418-.778.093-.16.186-.316.288-.468.063-.095.133-.186.2-.277L3.773 5c.118-.155.26-.29.385-.437.266-.3.544-.604.845-.877a9.33 9.33 0 0 1 .813-.686C6.97 2.167 8.31 1.59 9.757 1.35a9.27 9.27 0 0 1 1.481-.119 8.82 8.82 0 0 1 .748.031c.247.02.49.05.733.088 1.448.238 2.786.816 3.94 1.65.387.28.752.588 1.094.922a9.94 9.94 0 0 1 .949 1.078l.066.092c.102.133.203.268.295.408a9.97 9.97 0 0 1 1.571 4.128c.066.467.103.945.103 1.43 0 3.388-1.67 6.453-4.353 8.243.11.046.227.08.33.144l6.256 3.874c.37.23.645.55.82.9.115.24.185.498.203.76m.697-1.195c-.265-.55-.677-1.007-1.194-1.326l-5.323-3.297c2.255-2.037 3.564-4.97 3.564-8.114 0-2.19-.637-4.304-1.84-6.114-.126-.188-.26-.37-.4-.552-.645-.848-1.402-1.6-2.252-2.204C15.472.91 13.393.232 11.238.232A10.21 10.21 0 0 0 5.23 2.19c-.848.614-1.606 1.356-2.253 2.205-.136.18-.272.363-.398.55C1.374 6.756.737 8.87.737 11.06c0 4.218 2.407 8.08 6.133 9.842l.863.41v3.092l-2.525-.51c-.356-.07-.717-.106-1.076-.106a5.45 5.45 0 0 0-3.14.996c-.653.46-1.022 1.202-.99 1.983a2.28 2.28 0 0 0 1.138 1.872c2.24 1.318 4.106 2.923 5.543 4.772 1.26 1.62 3.333 2.59 5.55 2.592h6.698c1.42-.001 2.68-.86 3.134-2.138l2.76-7.74c.272-.757.224-1.584-.134-2.325" fill-opacity=".05" fill="${fg}" mask="url(#B)"/></g>` +
    `</g></svg>`
  );
}

function showInteractionHint(container, theme) {
  const dark = theme === "dark";
  const wrap = document.createElement("div");
  wrap.className = "usf-interaction-hint";
  wrap.innerHTML =
    `<div class="usf-hint-hand-wrap">` +
    `<div class="usf-hint-hand usf-hint-hand--pan">${handSVG(dark)}</div>` +
    `</div>`;
  container.appendChild(wrap);
  return wrap;
}

function showSpinner(container, theme) {
  const wrap = document.createElement("div");
  wrap.className = "usf-viewer-spinner";
  const dark = theme === "dark";
  wrap.innerHTML =
    `<svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="${dark ? "#aaa" : "#666"}" stroke-width="2">` +
    `<circle cx="12" cy="12" r="10" stroke-opacity="0.25"/>` +
    `<path d="M12 2a10 10 0 0 1 10 10" stroke-linecap="round"><animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="0.8s" repeatCount="indefinite"/></path>` +
    `</svg>`;
  container.appendChild(wrap);
  return wrap;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const container = document.getElementById("usf-viewer-root");
  if (!container) return;

  const params = parseParams(new URLSearchParams(window.location.search));

  if (!params.url) {
    showError(
      container,
      'Missing "url" query parameter. Example: ?url=/data/spherical_bin/demo.bin',
    );
    return;
  }

  // -- Loading indicator ----------------------------------------------------

  showSpinner(container, params.theme);

  // -- Fetch & decode -------------------------------------------------------

  let buffer;
  try {
    const res = await fetch(params.url);
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${params.url}`);
    buffer = await res.arrayBuffer();
  } catch (e) {
    showError(container, `Load failed: ${e.message}`);
    return;
  }

  let decoded;
  try {
    decoded = decodeSphericalBin(buffer);
  } catch (e) {
    showError(container, `Decode error: ${e.message}`);
    return;
  }

  const { n, b, c, vector, mask } = decoded;
  if (c !== 1 && c !== 3) {
    showError(container, `Unsupported channel count C=${c} (need 1 or 3).`);
    return;
  }

  // -- Masked subset --------------------------------------------------------

  const validIndices = [];
  for (let i = 0; i < n; i++) {
    if (mask[i]) validIndices.push(i);
  }
  if (validIndices.length === 0) {
    showError(container, "All points masked out — nothing to display.");
    return;
  }

  const vCount = validIndices.length;

  // Unit sphere for the translucent shell; points sit just outside so grid/triangles don’t
  // z-fight with GL_POINTS and read as an equirectangular-like pattern over the image.
  const referenceSphereRadius = 1;
  const pointRadius = 1.004;

  // -- Positions (USF vectors → Three.js) -----------------------------------

  const positions = new Float32Array(vCount * 3);
  const tmp = new THREE.Vector3();
  for (let j = 0; j < vCount; j++) {
    const i = validIndices[j];
    usfToThree(vector[i * 3], vector[i * 3 + 1], vector[i * 3 + 2], tmp);
    if (tmp.lengthSq() > 1e-20) tmp.multiplyScalar(pointRadius);
    const o = j * 3;
    positions[o] = tmp.x;
    positions[o + 1] = tmp.y;
    positions[o + 2] = tmp.z;
  }

  // -- Colors (first frame) -------------------------------------------------

  const colors = new Float32Array(vCount * 3);
  fillFrameColors(decoded, 0, validIndices, colors);

  // -- Point cloud ----------------------------------------------------------

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const pointMaterial = new THREE.PointsMaterial({
    size: params.pointSize,
    sizeAttenuation: false,
    vertexColors: true,
  });
  const points = new THREE.Points(geometry, pointMaterial);

  // -- Reference sphere -----------------------------------------------------

  const sphereGeom = new THREE.SphereGeometry(referenceSphereRadius, 128, 64);
  const sphereMat = new THREE.MeshBasicMaterial({
    color: params.meshColor,
    transparent: true,
    opacity: 0.9,
    depthWrite: false,
  });
  const sphere = new THREE.Mesh(sphereGeom, sphereMat);

  // -- Scene ----------------------------------------------------------------

  const scene = new THREE.Scene();
  scene.background = params.bgColor;
  scene.add(sphere);
  scene.add(points);

  // -- Camera ---------------------------------------------------------------
  // Place camera along the default USF viewing direction, at a distance that
  // frames the unit sphere to fill the viewport (with a small margin).

  const padding = 1.05;
  const eyeDir = usfToThree(...params.lookFrom).normalize();
  const spinAxisThree = usfToThree(...params.spinAxis).normalize();
  const upThree = usfToThree(0, 0, 1).normalize();

  const orthoHalf = (referenceSphereRadius * padding) / params.zoom;

  /** @type {THREE.PerspectiveCamera | THREE.OrthographicCamera} */
  let camera;
  if (params.projection === "ortho") {
    camera = new THREE.OrthographicCamera(
      -orthoHalf,
      orthoHalf,
      orthoHalf,
      -orthoHalf,
      0.01,
      100,
    );
  } else {
    camera = new THREE.PerspectiveCamera(params.fov, 1, 0.01, 100);
  }

  const vfovRad = (params.fov * Math.PI) / 180;
  const dist =
    params.projection === "ortho"
      ? 5
      : ((1.0 / Math.sin(vfovRad / 2)) * padding) / params.zoom;

  camera.position.copy(eyeDir.clone().multiplyScalar(dist));
  camera.up.copy(upThree);
  camera.lookAt(0, 0, 0);

  // -- Renderer -------------------------------------------------------------

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;

  // -- Controls -------------------------------------------------------------
  // `controls=trackball` (default): quaternion rotation — no pole stop.
  // `controls=orbit`: spherical orbit; polar angle clamps at poles.
  // `controls=arcball`: alternate arcball implementation.

  /** @type {OrbitControls | TrackballControls | ArcballControls} */
  let controls;
  const controlsKind = params.controlsMode;

  if (controlsKind === "trackball") {
    controls = new TrackballControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.rotateSpeed = 1.2;
    controls.zoomSpeed = 1.2;
    controls.panSpeed = 0.3;
    controls.noPan = !params.enablePan;
    // Trackball: higher dynamicDampingFactor = snappier zoom/pan/rotate (Three default is 0.2; 0.08 felt sluggish).
    controls.staticMoving = false;
    controls.dynamicDampingFactor = 0.45;
    controls.minDistance = 0.05;
    controls.maxDistance = 20;
  } else if (controlsKind === "arcball") {
    controls = new ArcballControls(camera, renderer.domElement, scene);
    controls.target.set(0, 0, 0);
    controls.enablePan = params.enablePan;
    controls.minDistance = 0.05;
    controls.maxDistance = 20;
    controls.setGizmosVisible(false);
  } else {
    controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.05;
    controls.maxDistance = 20;
    controls.enablePan = params.enablePan;
    controls.autoRotate = params.autoRotateSpeed !== 0;
    controls.autoRotateSpeed = params.autoRotateSpeed;
  }

  let trackballAutoRotate = params.autoRotateSpeed !== 0;

  renderer.domElement.addEventListener("pointerdown", () => {
    if (controlsKind === "orbit") {
      controls.autoRotate = false;
    } else {
      trackballAutoRotate = false;
    }
  });

  controls.update();

  const autoRotateTimer = new THREE.Timer();

  function applyWorldUpAutoRotate(dt) {
    if (params.autoRotateSpeed === 0 || dt <= 0) return;
    const angle = ((2 * Math.PI) / 60) * params.autoRotateSpeed * dt;
    _autoRotateQuat.setFromAxisAngle(spinAxisThree, angle);
    _autoRotateOffset.copy(camera.position).sub(controls.target);
    _autoRotateOffset.applyQuaternion(_autoRotateQuat);
    camera.position.copy(controls.target).add(_autoRotateOffset);
    camera.up.applyQuaternion(_autoRotateQuat);
    if (controlsKind === "arcball") {
      camera.updateMatrix();
      controls.updateMatrixState();
    }
  }

  // -- Mount (replace spinner) ----------------------------------------------

  container.innerHTML = "";
  container.appendChild(renderer.domElement);

  // -- Maximize button ------------------------------------------------------

  const expandSVG =
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">` +
    `<polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/>` +
    `<line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>`;
  const compressSVG =
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">` +
    `<polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/>` +
    `<line x1="14" y1="10" x2="21" y2="3"/><line x1="3" y1="21" x2="10" y2="14"/></svg>`;

  const MAXIMIZE_MARGIN = 12;
  const MAXIMIZE_RADIUS = 16;
  const MAXIMIZE_DURATION = 350;
  const MAXIMIZE_EASING = "cubic-bezier(0.4, 0, 0.15, 1)";
  const MAXIMIZE_TRANSITION = `transform ${MAXIMIZE_DURATION}ms ${MAXIMIZE_EASING}, border-radius ${MAXIMIZE_DURATION}ms ${MAXIMIZE_EASING}`;

  let maximized = false;
  const iframeEl = window.frameElement;
  let savedIframeStyle = "";
  let backdrop = null;

  const maxBtn = document.createElement("button");
  maxBtn.className = "usf-maximize-btn";
  maxBtn.title = "Maximize";
  maxBtn.innerHTML = expandSVG;
  container.appendChild(maxBtn);

  function createBackdrop() {
    const parentDoc = window.parent.document;
    backdrop = parentDoc.createElement("div");
    Object.assign(backdrop.style, {
      position: "fixed",
      inset: "0",
      zIndex: "999998",
      background: "rgba(0, 0, 0, 0)",
      backdropFilter: "blur(0px)",
      WebkitBackdropFilter: "blur(0px)",
      transition: MAXIMIZE_TRANSITION,
      cursor: "pointer",
    });
    parentDoc.body.appendChild(backdrop);
    backdrop.addEventListener("click", () => setMaximized(false));
    requestAnimationFrame(() => {
      Object.assign(backdrop.style, {
        background: "rgba(0, 0, 0, 0.4)",
        backdropFilter: "blur(6px)",
        WebkitBackdropFilter: "blur(6px)",
      });
    });
  }

  function removeBackdrop() {
    if (!backdrop) return;
    Object.assign(backdrop.style, {
      background: "rgba(0, 0, 0, 0)",
      backdropFilter: "blur(0px)",
      WebkitBackdropFilter: "blur(0px)",
    });
    const el = backdrop;
    el.addEventListener("transitionend", () => el.remove(), { once: true });
    setTimeout(() => el.remove(), 400);
    backdrop = null;
  }

  function setMaximized(on) {
    maximized = on;
    maxBtn.innerHTML = maximized ? compressSVG : expandSVG;
    maxBtn.title = maximized ? "Minimize" : "Maximize";

    if (iframeEl) {
      const m = MAXIMIZE_MARGIN;
      const parentHtml = window.parent.document.documentElement;

      const onDone = (fn) => {
        const cleanup = () => fn();
        iframeEl.addEventListener("transitionend", cleanup, { once: true });
        setTimeout(cleanup, MAXIMIZE_DURATION + 50);
      };

      if (maximized) {
        savedIframeStyle = iframeEl.style.cssText;
        const from = iframeEl.getBoundingClientRect();
        const fromRadius = getComputedStyle(iframeEl).borderRadius;
        parentHtml.style.overflow = "hidden";

        Object.assign(iframeEl.style, {
          position: "fixed",
          top: m + "px",
          left: m + "px",
          width: `calc(100% - ${m * 2}px)`,
          height: `calc(100% - ${m * 2}px)`,
          zIndex: "999999",
          margin: "0",
          aspectRatio: "auto",
          borderRadius: MAXIMIZE_RADIUS + "px",
          transition: "none",
          transformOrigin: "0 0",
        });
        void iframeEl.offsetHeight;

        const to = iframeEl.getBoundingClientRect();
        const sx = from.width / to.width;
        const sy = from.height / to.height;
        const tx = from.left - to.left;
        const ty = from.top - to.top;

        iframeEl.style.transform = `translate(${tx}px, ${ty}px) scale(${sx}, ${sy})`;
        iframeEl.style.borderRadius = fromRadius;
        void iframeEl.offsetHeight;

        createBackdrop();
        onDone(() => {
          iframeEl.style.transition = "none";
          iframeEl.style.transform = "";
          iframeEl.style.transformOrigin = "";
        });
        iframeEl.style.transition = MAXIMIZE_TRANSITION;
        iframeEl.style.transform = "none";
        iframeEl.style.borderRadius = MAXIMIZE_RADIUS + "px";
      } else {
        removeBackdrop();
        const from = iframeEl.getBoundingClientRect();

        iframeEl.style.cssText = savedIframeStyle;
        const to = iframeEl.getBoundingClientRect();
        const toRadius = getComputedStyle(iframeEl).borderRadius || "12px";

        Object.assign(iframeEl.style, {
          position: "fixed",
          top: from.top + "px",
          left: from.left + "px",
          width: from.width + "px",
          height: from.height + "px",
          zIndex: "999999",
          margin: "0",
          aspectRatio: "auto",
          borderRadius: MAXIMIZE_RADIUS + "px",
          transition: "none",
          transformOrigin: "0 0",
          transform: "none",
        });
        void iframeEl.offsetHeight;

        const sx = to.width / from.width;
        const sy = to.height / from.height;
        const tx = to.left - from.left;
        const ty = to.top - from.top;

        onDone(() => {
          iframeEl.style.cssText = savedIframeStyle;
          parentHtml.style.overflow = "";
          resize();
        });
        iframeEl.style.transition = MAXIMIZE_TRANSITION;
        iframeEl.style.transform = `translate(${tx}px, ${ty}px) scale(${sx}, ${sy})`;
        iframeEl.style.borderRadius = toRadius;
      }
    } else {
      container.classList.toggle("usf-maximized", maximized);
    }
  }

  maxBtn.addEventListener("click", () => {
    if (window.parent !== window) {
      window.parent.postMessage(
        { type: "usf-maximize", maximized: !maximized },
        "*",
      );
    }
    setMaximized(!maximized);
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && maximized) {
      setMaximized(false);
    }
  });

  // -- Interaction hint (dismiss globally across all viewer iframes) --------

  const HINT_STORAGE_KEY = "usf-viewer-hint-dismissed";
  const hintAlreadyDismissed = (() => {
    try {
      return sessionStorage.getItem(HINT_STORAGE_KEY) === "1";
    } catch {
      return false;
    }
  })();

  const hint = showInteractionHint(container, params.theme);
  const hintChannel = new BroadcastChannel("usf-viewer-hint");
  let hintDismissed = hintAlreadyDismissed;
  let hintTimers = [];

  const hintPanInner = hint.querySelector(".usf-hint-hand--pan");

  const restartPanAnim = () => {
    hintPanInner.style.animation = "none";
    void hintPanInner.offsetWidth;
    hintPanInner.style.animation = "";
  };

  if (hintAlreadyDismissed) {
    hint.remove();
  } else {
    const PAN_MS = 2800;
    const FADE_MS = 450;
    const GAP_MS = 2800;
    const INITIAL_DELAY_MS = 800;

    const scheduleHintCycle = (isFirstCycle) => {
      if (hintDismissed) return;
      let acc = isFirstCycle ? INITIAL_DELAY_MS : GAP_MS;

      hintTimers.push(
        setTimeout(() => {
          if (hintDismissed) return;
          restartPanAnim();
          hint.classList.add("visible");
        }, acc),
      );

      acc += PAN_MS;
      hintTimers.push(
        setTimeout(() => {
          if (hintDismissed) return;
          hint.classList.remove("visible");
        }, acc),
      );

      acc += FADE_MS;
      hintTimers.push(
        setTimeout(() => {
          if (hintDismissed) return;
          scheduleHintCycle(false);
        }, acc),
      );
    };

    scheduleHintCycle(true);

    const dismissHint = (broadcast = true) => {
      hintDismissed = true;
      hintTimers.forEach(clearTimeout);
      hint.classList.remove("visible");
      hint.addEventListener("transitionend", () => hint.remove(), {
        once: true,
      });
      setTimeout(() => hint.remove(), 500);
      renderer.domElement.removeEventListener(
        "pointerdown",
        onLocalInteraction,
      );
      renderer.domElement.removeEventListener("wheel", onLocalInteraction);
      if (broadcast) hintChannel.postMessage("dismiss");
      try {
        sessionStorage.setItem(HINT_STORAGE_KEY, "1");
      } catch {
        /* sessionStorage unavailable (e.g. private browsing) */
      }
    };

    const onLocalInteraction = () => dismissHint(true);
    renderer.domElement.addEventListener("pointerdown", onLocalInteraction);
    renderer.domElement.addEventListener("wheel", onLocalInteraction);
    hintChannel.addEventListener("message", () => dismissHint(false));
  }

  // -- Sizing ---------------------------------------------------------------

  function resize() {
    const w = Math.max(1, container.clientWidth);
    const h = Math.max(1, container.clientHeight);
    if (camera instanceof THREE.OrthographicCamera) {
      const aspect = w / h;
      if (aspect >= 1) {
        camera.left = -orthoHalf * aspect;
        camera.right = orthoHalf * aspect;
        camera.top = orthoHalf;
        camera.bottom = -orthoHalf;
      } else {
        camera.left = -orthoHalf;
        camera.right = orthoHalf;
        camera.top = orthoHalf / aspect;
        camera.bottom = -orthoHalf / aspect;
      }
    } else {
      camera.aspect = w / h;
    }
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    if (typeof controls.handleResize === "function") {
      controls.handleResize();
    }
  }

  resize();
  window.addEventListener("resize", resize);
  new ResizeObserver(resize).observe(container);

  // -- Visibility (pause when off-screen — critical for carousels) ----------

  let visible = true;
  let paused = false;

  if (typeof IntersectionObserver !== "undefined") {
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          visible = e.isIntersecting;
        }
      },
      { threshold: 0.05 },
    );
    io.observe(container);
  }

  document.addEventListener("visibilitychange", () => {
    visible = !document.hidden;
  });

  // -- postMessage API (parent can pause/resume/setFps) ---------------------

  window.addEventListener("message", (evt) => {
    if (!evt.data || typeof evt.data !== "object") return;
    const { type } = evt.data;
    if (type === "pause") paused = true;
    if (type === "resume") paused = false;
    if (type === "setFps" && typeof evt.data.fps === "number") {
      params.fps = Math.max(0, evt.data.fps);
    }
  });

  // -- Animation loop -------------------------------------------------------

  const colorAttr = geometry.getAttribute("color");
  const timer = new THREE.Timer();
  let prevFrame = -1;

  function animate(timestamp) {
    requestAnimationFrame(animate);
    timer.update(timestamp);
    autoRotateTimer.update(timestamp);

    const trackballDt =
      controlsKind === "trackball" || controlsKind === "arcball"
        ? Math.min(autoRotateTimer.getDelta(), 0.05)
        : 0;

    if (!visible || paused) {
      return;
    }

    if (b > 1 && params.fps > 0) {
      let fi = Math.floor(timer.getElapsed() * params.fps);
      fi = params.loop ? fi % b : Math.min(fi, b - 1);
      if (fi !== prevFrame) {
        prevFrame = fi;
        fillFrameColors(decoded, fi, validIndices, colorAttr.array);
        colorAttr.needsUpdate = true;
      }
    }

    if (trackballAutoRotate && trackballDt > 0) {
      applyWorldUpAutoRotate(trackballDt);
    }

    controls.update();
    renderer.render(scene, camera);
  }

  requestAnimationFrame(animate);
}

main();
