/**
 * After `astro build`, serves `dist/` with `astro preview`, opens the homepage
 * in a 16:9 desktop viewport (1920×1080), seeks hero background video(s) to `OG_VIDEO_TIME_SEC` (default 27),
 * then saves a **viewport** PNG (not just the `<header>` box) to `dist/og-thumbnail.png`.
 *
 * Env:
 *   ASTRO_BASE   — same as `astro build --base` / `astro preview --base` (default `/`). Required for project Pages (e.g. `/USF`).
 *   OG_PREVIEW_PORT — local preview port (default `8791`).
 *   OG_VIDEO_TIME_SEC — seek hero background video(s) to this timestamp before capture (default `27`, same as headline posters).
 */
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const distDir = path.join(root, "dist");
const astroCli = path.join(root, "node_modules", "astro", "astro.js");
const port = process.env.OG_PREVIEW_PORT || "8791";
const outFile = path.join(distDir, "og-thumbnail.png");
/** 16:9 link-preview frame */
const VIEWPORT = { width: 1920, height: 1080 };
const OG_VIDEO_TIME_SEC = Number(process.env.OG_VIDEO_TIME_SEC || "27");

/**
 * Same semantics as `astro build --base`: `/` for root sites, `/USF` for project Pages.
 * Must match CI `ASTRO_BASE` / `configure-pages` `base_path`.
 */
function normalizedBasePathname() {
  let p = (process.env.ASTRO_BASE || "").trim();
  if (!p || p === "/") return "/";
  if (!p.startsWith("/")) p = `/${p}`;
  const noTrail = p.replace(/\/+$/, "");
  return noTrail || "/";
}

function homeUrl() {
  const u = new URL(`http://127.0.0.1:${port}`);
  const base = normalizedBasePathname();
  u.pathname = base === "/" ? "/" : `${base}/`;
  return u.href;
}

/** `astro preview` does not read `--base` from the last build; pass the same as CI `astro build --base`. */
function previewBaseCliArgs() {
  const base = normalizedBasePathname();
  if (base === "/") return [];
  return ["--base", base];
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/** Wait for child exit, then drop stdio handles so Node can exit. */
async function stopPreviewServer(child) {
  if (!child?.pid) return;

  const exitPromise = new Promise((resolve) => {
    child.once("exit", resolve);
  });

  child.kill("SIGTERM");
  await Promise.race([exitPromise, sleep(4000)]);

  if (child.exitCode === null && child.signalCode === null) {
    child.kill("SIGKILL");
    await Promise.race([exitPromise, sleep(2000)]);
  }

  child.stdout?.destroy();
  child.stderr?.destroy();
}

async function waitForServer(url, { timeoutMs = 90000 } = {}) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url, { redirect: "follow" });
      if (res.ok) return;
    } catch {
      /* server not ready */
    }
    await sleep(300);
  }
  throw new Error(`Timed out waiting for ${url}`);
}

async function main() {
  if (!fs.existsSync(distDir)) {
    console.error("dist/ not found. Run astro build first.");
    process.exit(1);
  }

  if (!fs.existsSync(astroCli)) {
    console.error(
      "astro CLI not found. Run npm install from the project root.",
    );
    process.exit(1);
  }

  const url = homeUrl();
  /** Run Astro directly (not via npx) so SIGTERM/SIGKILL stops the preview server. */
  const preview = spawn(
    process.execPath,
    [
      astroCli,
      "preview",
      "--host",
      "127.0.0.1",
      "--port",
      port,
      ...previewBaseCliArgs(),
    ],
    {
      cwd: root,
      // stdout ignored so a chatty server cannot fill the pipe and block shutdown
      stdio: ["ignore", "ignore", "pipe"],
      env: { ...process.env },
    },
  );

  let stderrBuf = "";
  preview.stderr?.on("data", (c) => {
    stderrBuf += c.toString();
  });

  const onSigInt = () => {
    void stopPreviewServer(preview);
    process.exit(130);
  };
  process.on("SIGINT", onSigInt);

  try {
    await waitForServer(url);

    const browser = await chromium.launch({
      args: ["--no-sandbox", "--disable-dev-shm-usage"],
    });
    const context = await browser.newContext({
      viewport: VIEWPORT,
      deviceScaleFactor: 1,
      colorScheme: "light",
      // no reducedMotion — keep hero video visible for a realistic frame
    });
    const page = await context.newPage();
    await page.goto(url, { waitUntil: "networkidle", timeout: 120000 });
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.addStyleTag({
      content:
        "#usf-theme-toggle, #usf-back-to-top { display: none !important; }",
    });
    await page.waitForFunction(() => document.fonts.ready);

    const hero = page.locator("[data-og-thumbnail]");
    await hero.waitFor({ state: "visible", timeout: 30000 });
    const bgVideos = page.locator("video[data-title-bg-video]");
    if ((await bgVideos.count()) > 0) {
      await bgVideos.evaluateAll(async (elements, timeSec) => {
        const t = Number(timeSec);
        for (const el of elements) {
          const v = el;
          if (!(v instanceof HTMLVideoElement)) continue;
          await new Promise((resolve) => {
            const seek = () => {
              const dur = v.duration;
              const target =
                Number.isFinite(dur) && dur > 0
                  ? Math.min(t, Math.max(0, dur - 0.05))
                  : t;
              let settled = false;
              const done = () => {
                if (settled) return;
                settled = true;
                clearTimeout(failsafe);
                v.removeEventListener("seeked", onSeeked);
                resolve();
              };
              const onSeeked = () => done();
              const failsafe = setTimeout(done, 4000);
              v.addEventListener("seeked", onSeeked, { once: true });
              v.pause();
              try {
                v.currentTime = target;
              } catch {
                done();
              }
            };
            if (v.readyState >= HTMLMediaElement.HAVE_METADATA) seek();
            else v.addEventListener("loadedmetadata", seek, { once: true });
          });
        }
      }, OG_VIDEO_TIME_SEC);
      await sleep(400);
    }

    await page.screenshot({
      path: outFile,
      type: "png",
      clip: { x: 0, y: 0, width: VIEWPORT.width, height: VIEWPORT.height },
    });

    await browser.close();
    console.log(`Wrote ${outFile}`);
  } catch (e) {
    if (stderrBuf.trim()) console.error(stderrBuf);
    throw e;
  } finally {
    process.off("SIGINT", onSigInt);
    await stopPreviewServer(preview);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
