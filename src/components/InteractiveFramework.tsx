import {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
  useSyncExternalStore,
  type ReactNode,
  type RefObject,
  type TouchEvent,
} from "react";
import katex from "katex";
import { AdaptiveSvg, useIsDesktop } from "./AdaptiveSvg";
import pinholeImg from "../assets/pinhole.png";
import fisheyeImg from "../assets/fisheye.png";
import panoramaImg from "../assets/panorama.jpg";

interface Card {
  title: string;
  content: ReactNode;
  aspectRatio: string;
}

interface Section {
  id: string;
  label: string;
  region: { left: number; width: number; top?: number; height?: number };
  description: ReactNode;
  cards: Card[];
  extra?: ReactNode;
  layout?: "horizontal" | "vertical";
  rightColumnWidth?: string;
  /** Zero-based index of the card shown on top by default. */
  defaultCard?: number;
}

const iframeClass = "block h-full w-full rounded-xl border-0";

const poolingEquationHtml = katex.renderToString(
  String.raw`x_{o} = \mathcal{K}_{\text{pool}}(\mathcal{X}_{i}, \mathbf{p}_{o}) = f_{\text{pool}}\left( x_{k}: k \in \mathcal{N}(\mathbf{p}_{o}) \right)`,
  { displayMode: true, throwOnError: false },
);

const convolutionEquationHtml = katex.renderToString(
  String.raw`\begin{aligned}
  x_{o} & = \mathcal{K}_{\text{conv}}(\mathcal{X}_{i}, \mathbf{p}_{o}) \\
        & = \frac{1}{|\mathcal{N}(\mathbf{p}_{o})|}\sum\limits_{k\in\mathcal{N}(\mathbf{p}_o)}x_{k}\prod\limits_{m}f^{(m)}_{\text{weight}}\left( \mathcal{M}_{m}(\mathbf{p}_{k}, \mathbf{p}_{o})\right ) \\
  \mathcal{N}(\mathbf{p}_{o}) & = \{ k: \mathbf{p}_{k}\in \mathcal{P}_{i}, \; d \; (\mathbf{p}_{k}, \mathbf{p}_{o}) \leq r \}
  \end{aligned}`,
  { displayMode: true, throwOnError: false },
);

const TOP_ROW_HEIGHT_PCT = 34;

const sections: Section[] = [
  {
    id: "planar-image",
    label: "Planar Image",
    region: { left: 0, width: 17 },
    defaultCard: 2,
    description: (
      <>
        Input images from <strong>any calibrated camera</strong> (pinhole,
        fisheye, or panoramic) with known intrinsics. Each pixel maps to a ray
        direction on the unit sphere, defined either by a parametric camera
        model or a dense <em>lens normal map</em>.
      </>
    ),
    cards: [
      {
        title: "Pinhole Image",
        aspectRatio: "640/480",
        content: (
          <img
            src={pinholeImg.src}
            width={pinholeImg.width}
            height={pinholeImg.height}
            alt="Scene captured with a pinhole camera"
            loading="lazy"
            decoding="async"
            className="block h-full w-full rounded-xl object-cover"
          />
        ),
      },
      {
        title: "Fisheye Image",
        aspectRatio: "1224/1028",
        content: (
          <img
            src={fisheyeImg.src}
            width={fisheyeImg.width}
            height={fisheyeImg.height}
            alt="Scene captured with a fisheye lens"
            loading="lazy"
            decoding="async"
            className="block h-full w-full rounded-xl object-cover"
          />
        ),
      },
      {
        title: "360° Panorama",
        aspectRatio: "1024/512",
        content: (
          <img
            src={panoramaImg.src}
            width={panoramaImg.width}
            height={panoramaImg.height}
            alt="Scene captured as a 360° panorama"
            loading="lazy"
            decoding="async"
            className="block h-full w-full rounded-xl object-cover"
          />
        ),
      },
    ],
  },
  {
    id: "projection",
    label: "Projection",
    region: { left: 17, width: 15 },
    defaultCard: 2,
    description: (
      <>
        Projecting all pixels onto the unit sphere yields a{" "}
        <strong>spherical image</strong> where geometry (unit-norm R³ vectors)
        and scalar values (RGB, features) are <em>explicitly separated</em>.
        Different lenses produce non-uniform density and distribution on the
        sphere (zoom in on the sphere to see), motivating the resampling step.
      </>
    ),
    extra: (
      <div className="mt-4 grid grid-cols-2 items-end gap-3">
        <figure className="m-0">
          <AdaptiveSvg
            src="pinhole_projection.svg"
            alt="Pinhole camera projection: pixels map to a narrow cone of ray directions on the unit sphere, producing a concentrated point distribution"
            className="h-auto w-full"
          />
          <figcaption className="mt-1 text-center text-sm text-zinc-500 dark:text-zinc-400">
            Pinhole Camera Model
          </figcaption>
        </figure>
        <figure className="m-0">
          <AdaptiveSvg
            src="fisheye_projection.svg"
            alt="Fisheye camera projection: pixels map to a wide hemisphere of ray directions on the unit sphere, producing a spread-out point distribution"
            className="h-auto w-full"
          />
          <figcaption className="mt-1 text-center text-sm text-zinc-500 dark:text-zinc-400">
            Fisheye Camera Model
          </figcaption>
        </figure>
      </div>
    ),
    cards: [
      {
        title: "Pinhole Spherical Image",
        aspectRatio: "1/1",
        content: (
          <iframe
            title="Pinhole Spherical Image"
            src="spherical-viewer/index.html?url=../spherical-image/pinhole.bin&fps=0&theme=auto"
            allow="fullscreen"
            loading="lazy"
            className={iframeClass}
          />
        ),
      },
      {
        title: "Fisheye Spherical Image",
        aspectRatio: "1/1",
        content: (
          <iframe
            title="Fisheye Spherical Image"
            src="spherical-viewer/index.html?url=../spherical-image/fisheye.bin&fps=0&theme=auto"
            allow="fullscreen"
            loading="lazy"
            className={iframeClass}
          />
        ),
      },
      {
        title: "360° Spherical Image",
        aspectRatio: "1/1",
        content: (
          <iframe
            title="360° Spherical Image"
            src="spherical-viewer/index.html?url=../spherical-image/panorama.bin&fps=0&theme=auto"
            allow="fullscreen"
            loading="lazy"
            className={iframeClass}
          />
        ),
      },
    ],
  },
  {
    id: "resampling",
    label: "Resampling",
    region: { left: 32, width: 15 },
    description: (
      <>
        Resampling the raw projection into a{" "}
        <strong>near-uniform distribution</strong> on the sphere via two
        decoupled steps: (1) <em>location sampling</em> selects new points with
        improved spatial uniformity, and (2) <em>value interpolation</em>{" "}
        assigns features via geodesic-distance-based weighting. No grid, mesh,
        or sequential ordering is assumed. The entire pipeline is{" "}
        <strong>geometry-cacheable</strong>.
      </>
    ),
    cards: [
      {
        title: "Fibonacci",
        aspectRatio: "1/1",
        content: (
          <iframe
            title="Fibonacci"
            src="spherical-viewer/index.html?url=../spherical-image/fibonacci.bin&fps=0&theme=auto"
            allow="fullscreen"
            loading="lazy"
            className={iframeClass}
          />
        ),
      },
      {
        title: "Quasi-Random",
        aspectRatio: "1/1",
        content: (
          <iframe
            title="Quasi-Random"
            src="spherical-viewer/index.html?url=../spherical-image/quasirandom.bin&fps=0&theme=auto"
            allow="fullscreen"
            loading="lazy"
            className={iframeClass}
          />
        ),
      },
      {
        title: "HEALPix",
        aspectRatio: "1/1",
        content: (
          <iframe
            title="HEALPix"
            src="spherical-viewer/index.html?url=../spherical-image/healpix.bin&fps=0&theme=auto"
            allow="fullscreen"
            loading="lazy"
            className={iframeClass}
          />
        ),
      },
      {
        title: "Icosahedron",
        aspectRatio: "1/1",
        content: (
          <iframe
            title="Icosahedron"
            src="spherical-viewer/index.html?url=../spherical-image/icosahedron.bin&fps=0&theme=auto"
            allow="fullscreen"
            loading="lazy"
            className={iframeClass}
          />
        ),
      },
      {
        title: "Octahedron",
        aspectRatio: "1/1",
        content: (
          <iframe
            title="Octahedron"
            src="spherical-viewer/index.html?url=../spherical-image/octahedron.bin&fps=0&theme=auto"
            allow="fullscreen"
            loading="lazy"
            className={iframeClass}
          />
        ),
      },
      {
        title: "Hexahedron",
        aspectRatio: "1/1",
        content: (
          <iframe
            title="Hexahedron"
            src="spherical-viewer/index.html?url=../spherical-image/hexahedron.bin&fps=0&theme=auto"
            allow="fullscreen"
            loading="lazy"
            className={iframeClass}
          />
        ),
      },
      {
        title: "Tetrahedron",
        aspectRatio: "1/1",
        content: (
          <iframe
            title="Tetrahedron"
            src="spherical-viewer/index.html?url=../spherical-image/tetrahedron.bin&fps=0&theme=auto"
            allow="fullscreen"
            loading="lazy"
            className={iframeClass}
          />
        ),
      },
      {
        title: "Equirectangular",
        aspectRatio: "1/1",
        content: (
          <iframe
            title="Equirectangular"
            src="spherical-viewer/index.html?url=../spherical-image/equirectangular.bin&fps=0&theme=auto"
            allow="fullscreen"
            loading="lazy"
            className={iframeClass}
          />
        ),
      },
    ],
  },
  {
    id: "spherical-convolution",
    label: "Spherical Convolution",
    region: { left: 47.5, width: 26 },
    description: (
      <>
        Spatial-domain convolution over{" "}
        <strong>circular-cap neighborhoods</strong> on the sphere. Weights are
        computed from two relative geometric measurements,{" "}
        <em>geodesic distance</em> and <em>direction</em>, via decomposable
        weighting functions. When the direction branch is absent, the kernel
        reduces to a zonal filter that is{" "}
        <strong>rotation-equivariant by construction</strong>, avoiding harmonic
        transforms entirely.
      </>
    ),
    extra: (
      <div
        className="mt-4 [&_.katex-display]:!mx-0 [&_.katex-display]:!overflow-x-auto [&_.katex-display]:!overflow-y-hidden"
        // eslint-disable-next-line @eslint-react/dom-no-dangerously-set-innerhtml -- trusted KaTeX output
        dangerouslySetInnerHTML={{ __html: convolutionEquationHtml }}
      />
    ),
    cards: [
      {
        title: "Continuous Distance × Direction",
        aspectRatio: "1/1",
        content: (
          <iframe
            title="Continuous Distance × Direction"
            src="spherical-viewer/index.html?url=../spherical-image/continuous_distance_direction_kernel.bin&fps=0&theme=auto&enablePan=true&lookFrom=0.9997,0.0240,-0.0065&spinAxis=0.9997,0.0240,-0.0065&zoom=1.3"
            allow="fullscreen"
            loading="lazy"
            className={iframeClass}
          />
        ),
      },
      {
        title: "Continuous Distance",
        aspectRatio: "1/1",
        content: (
          <iframe
            title="Continuous Distance"
            src="spherical-viewer/index.html?url=../spherical-image/continuous_distance_kernel.bin&fps=0&theme=auto&enablePan=true&lookFrom=0.9997,0.0240,-0.0065&spinAxis=0.9997,0.0240,-0.0065&zoom=1.3"
            allow="fullscreen"
            loading="lazy"
            className={iframeClass}
          />
        ),
      },
      {
        title: "Discrete Distance",
        aspectRatio: "1/1",
        content: (
          <iframe
            title="Discrete Distance"
            src="spherical-viewer/index.html?url=../spherical-image/discrete_distance_kernel.bin&fps=0&theme=auto&enablePan=true&lookFrom=0.9997,0.0240,-0.0065&spinAxis=0.9997,0.0240,-0.0065&zoom=1.3"
            allow="fullscreen"
            loading="lazy"
            className={iframeClass}
          />
        ),
      },
    ],
  },
  {
    id: "spherical-pooling",
    label: "Spherical Pooling",
    region: { left: 74, width: 25 },
    description: (
      <>
        Pooling over the same geodesic circular-cap neighborhoods as
        convolution, with a configurable reducer (<em>min</em>, <em>max</em>,{" "}
        <em>average</em>, or higher-order statistics like upper-quartile mean).
        Downsampling and upsampling are controlled by a per-layer resolution
        factor applied to the location sampler.
      </>
    ),
    extra: (
      <div
        className="mt-4 [&_.katex-display]:!mx-0 [&_.katex-display]:!overflow-x-auto [&_.katex-display]:!overflow-y-hidden"
        // eslint-disable-next-line @eslint-react/dom-no-dangerously-set-innerhtml -- trusted KaTeX output
        dangerouslySetInnerHTML={{ __html: poolingEquationHtml }}
      />
    ),
    cards: [
      {
        title: "Max Pooling",
        aspectRatio: "1/1",
        content: (
          <iframe
            title="Max Pooling"
            src="spherical-viewer/index.html?url=../spherical-image/max_pool.bin&fps=0&theme=auto"
            allow="fullscreen"
            loading="lazy"
            className={iframeClass}
          />
        ),
      },
      {
        title: "Mean Pooling",
        aspectRatio: "1/1",
        content: (
          <iframe
            title="Mean Pooling"
            src="spherical-viewer/index.html?url=../spherical-image/mean_pool.bin&fps=0&theme=auto"
            allow="fullscreen"
            loading="lazy"
            className={iframeClass}
          />
        ),
      },
    ],
  },
  {
    id: "backbone",
    label: "Backbone",
    region: { left: 47, width: 26, top: 0, height: TOP_ROW_HEIGHT_PCT },
    description: (
      <>
        Spherical convolution and pooling serve as{" "}
        <strong>plug-and-play replacements</strong> for their planar
        counterparts in any existing backbone. The macro architecture is
        preserved; only the layer primitives change.
      </>
    ),
    layout: "vertical",
    cards: [
      {
        title: "YOLOv11",
        aspectRatio: "1680/480",
        content: (
          <AdaptiveSvg
            src="YOLOv11.svg"
            alt="YOLOv11 architecture: CBNA blocks, C3K/C3K2 bottlenecks, SPPF spatial pyramid pooling, and C2PSA partial spatial attention modules"
            className="h-full w-full overflow-hidden"
          />
        ),
      },
      {
        title: "UNet",
        aspectRatio: "1680/576",
        content: (
          <AdaptiveSvg
            src="UNet.svg"
            alt="UNet architecture: symmetric encoder-decoder with skip connections, CBNA blocks, max pooling downsampling, and interpolation upsampling"
            className="h-full w-full overflow-hidden"
          />
        ),
      },
      {
        title: "DeepLab v3",
        aspectRatio: "960/480",
        content: (
          <AdaptiveSvg
            src="DeepLab_v3.svg"
            alt="DeepLab v3 architecture: ResNet backbone with bottleneck stages, atrous spatial pyramid pooling (ASPP) module, and decoder"
            className="h-full w-full overflow-hidden"
          />
        ),
      },
    ],
  },
];

/** First-visit pulse on `framework.svg` (% of diagram box, 0–100; same system as click regions). */
const FRAMEWORK_CLICK_HINT_LEFT_PCT = 39.35;
const FRAMEWORK_CLICK_HINT_TOP_PCT = 65;

const DIAGONAL_X = 20;
const DIAGONAL_Y = 16;
const SWIPE_EXIT_X = 120;
const SWIPE_EXIT_Y = -60;
const SWIPE_EXIT_ROTATE = -15;

function CardDeck({
  cards,
  sectionId,
  deckRef,
  fullWidth,
  defaultCard = 0,
}: {
  cards: Card[];
  sectionId: string;
  deckRef?: RefObject<HTMLDivElement | null>;
  fullWidth?: boolean;
  defaultCard?: number;
}) {
  const [frontIndex, setFrontIndex] = useState(defaultCard);
  const [anim, setAnim] = useState<"none" | "swipe-out" | "fly-in">("none");
  const animatingRef = useRef(false);

  useEffect(() => {
    setFrontIndex(defaultCard); // eslint-disable-line @eslint-react/set-state-in-effect -- reset on section change
    setAnim("none"); // eslint-disable-line @eslint-react/set-state-in-effect
  }, [sectionId, defaultCard]);

  if (cards.length === 0) return null;

  const maxVisible = Math.min(cards.length, 3);

  const navigate = (dir: "next" | "prev") => {
    if (animatingRef.current) return;
    animatingRef.current = true;

    if (dir === "next") {
      setAnim("swipe-out");
      setTimeout(() => {
        setFrontIndex((prev) => (prev + 1) % cards.length);
        setAnim("none");
        animatingRef.current = false;
      }, 250);
    } else {
      setFrontIndex((prev) => (prev - 1 + cards.length) % cards.length);
      setAnim("fly-in");
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          setAnim("none");
          setTimeout(() => {
            animatingRef.current = false;
          }, 400);
        });
      });
    }
  };

  const SHADOW_BLEED = 12;

  return (
    <div className="flex flex-col items-center">
      {/* Card stack area — extra padding for diagonal offset + shadow */}
      <div
        ref={deckRef}
        className="relative"
        style={{
          width: fullWidth
            ? `calc(100% - ${DIAGONAL_X * (maxVisible - 1) + SHADOW_BLEED}px)`
            : `min(560px, calc(100% - ${DIAGONAL_X * (maxVisible - 1) + SHADOW_BLEED}px))`,
          aspectRatio: cards[frontIndex].aspectRatio,
          marginRight: `${DIAGONAL_X * (maxVisible - 1)}px`,
          marginLeft: `${SHADOW_BLEED}px`,
          marginTop: `${DIAGONAL_Y * (maxVisible - 1)}px`,
          transition: "aspect-ratio 0.45s cubic-bezier(0.25, 0.46, 0.45, 0.94)",
        }}
      >
        {cards.map((card, i) => {
          const offset = (i - frontIndex + cards.length) % cards.length;
          if (offset >= maxVisible) return null;

          const isFront = offset === 0;
          const isSwipingOut = isFront && anim === "swipe-out";
          const isFlyingIn = isFront && anim === "fly-in";

          let tx: number,
            ty: number,
            rot: number,
            scale: number,
            opacity: number;
          let transition: string;

          if (isSwipingOut) {
            tx = SWIPE_EXIT_X;
            ty = SWIPE_EXIT_Y;
            rot = SWIPE_EXIT_ROTATE;
            scale = 1;
            opacity = 0;
            transition =
              "transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.2s";
          } else if (isFlyingIn) {
            tx = -SWIPE_EXIT_X;
            ty = SWIPE_EXIT_Y;
            rot = -SWIPE_EXIT_ROTATE;
            scale = 1;
            opacity = 0;
            transition = "none";
          } else {
            tx = offset * DIAGONAL_X;
            ty = -offset * DIAGONAL_Y;
            rot = 0;
            scale = 1 - offset * 0.03;
            opacity = isFront ? 1 : 0.5;
            transition =
              "transform 0.45s cubic-bezier(0.25, 0.46, 0.45, 0.94), opacity 0.4s, box-shadow 0.4s, filter 0.4s";
          }

          const isRestingFront = isFront && anim === "none";

          return (
            <div
              key={card.title}
              className={`absolute rounded-xl ${isRestingFront ? "" : "overflow-hidden"}`}
              style={{
                inset: 0,
                transform: isRestingFront
                  ? "none"
                  : `translate(${tx}px, ${ty}px) rotate(${rot}deg) scale(${scale})`,
                zIndex: maxVisible - offset + (isSwipingOut ? -1 : 0),
                opacity,
                filter:
                  isFront || isSwipingOut || isFlyingIn ? "none" : "blur(2px)",
                boxShadow:
                  isFront && !isFlyingIn
                    ? "4px 6px 14px rgba(0,0,0,0.22)"
                    : `${3 - offset}px ${4 - offset}px ${8 - offset * 2}px rgba(0,0,0,0.15)`,
                transition,
                pointerEvents: isRestingFront ? "auto" : "none",
              }}
            >
              {card.content}
            </div>
          );
        })}
      </div>

      {/* Title right below the card */}
      <p className="mt-3 text-center text-base font-medium text-zinc-700 dark:text-zinc-300">
        {cards[frontIndex]?.title}
      </p>

      {/* Navigation controls */}
      {cards.length > 1 && (
        <div className="mt-2 flex items-center gap-4">
          <button
            type="button"
            onClick={() => navigate("prev")}
            className="flex h-9 w-9 items-center justify-center rounded-full bg-zinc-200 transition-colors hover:bg-zinc-300 dark:bg-zinc-700 dark:hover:bg-zinc-600"
            aria-label="Previous card"
          >
            <svg
              className="h-4 w-4 text-zinc-700 dark:text-zinc-200"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 19l-7-7 7-7"
              />
            </svg>
          </button>
          <span className="min-w-[80px] text-center text-sm text-zinc-500 dark:text-zinc-400">
            {frontIndex + 1} / {cards.length}
          </span>
          <button
            type="button"
            onClick={() => navigate("next")}
            className="flex h-9 w-9 items-center justify-center rounded-full bg-zinc-200 transition-colors hover:bg-zinc-300 dark:bg-zinc-700 dark:hover:bg-zinc-600"
            aria-label="Next card"
          >
            <svg
              className="h-4 w-4 text-zinc-700 dark:text-zinc-200"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 5l7 7-7 7"
              />
            </svg>
          </button>
        </div>
      )}
    </div>
  );
}

const CLICK_HINT_KEY = "usf-framework-hint-dismissed";
/** Same-tab notification (sessionStorage does not fire `storage` in the active document). */
const CLICK_HINT_SYNC_EVENT = "usf-framework-hint-sync";
/** Slightly longer than caption `grid-template-rows` / opacity transition so unmount happens after paint. */
const CLICK_HINT_EXIT_MS = 560;

const RESAMPLING_PEEK_KEY = "usf-framework-resampling-peek-done";
const RESAMPLING_TOUCHED_KEY = "usf-framework-resampling-touched";
/** Accordion `grid-template-rows` duration is 500ms; hold briefly so content registers. */
const RESAMPLING_PEEK_CLOSE_DELAY_MS = 500 + 380;
/** Peek when methodology end (sentinel) crosses this far above the viewport bottom — early, before Results dominates. */
const RESAMPLING_PEEK_FROM_VIEWPORT_BOTTOM_PX = 56;

function getResamplingPeekDoneFromStorage(): boolean {
  try {
    return sessionStorage.getItem(RESAMPLING_PEEK_KEY) === "1";
  } catch {
    return false;
  }
}

function getResamplingTouchedFromStorage(): boolean {
  try {
    return sessionStorage.getItem(RESAMPLING_TOUCHED_KEY) === "1";
  } catch {
    return false;
  }
}

function persistResamplingPeekDone() {
  try {
    sessionStorage.setItem(RESAMPLING_PEEK_KEY, "1");
  } catch {
    /* sessionStorage unavailable */
  }
}

function persistResamplingTouched() {
  try {
    sessionStorage.setItem(RESAMPLING_TOUCHED_KEY, "1");
  } catch {
    /* sessionStorage unavailable */
  }
}

/** Widen diagram + caption + accordion to ~80rem; click hint stays in the section text column. */
const FRAMEWORK_DIAGRAM_BLEED_CLASS =
  "mx-[calc(max(var(--minimum-inline-margin),(100cqw-80rem)/2)-var(--actual-inline-margin))]";

function getClickHintGoneFromStorage(): boolean {
  try {
    return sessionStorage.getItem(CLICK_HINT_KEY) === "1";
  } catch {
    return false;
  }
}

function subscribeClickHintGone(onStoreChange: () => void) {
  if (typeof window === "undefined") return () => {};
  const run = () => onStoreChange();
  window.addEventListener(CLICK_HINT_SYNC_EVENT, run);
  window.addEventListener("storage", run);
  return () => {
    window.removeEventListener(CLICK_HINT_SYNC_EVENT, run);
    window.removeEventListener("storage", run);
  };
}

function persistClickHintDismissed() {
  try {
    sessionStorage.setItem(CLICK_HINT_KEY, "1");
  } catch {
    /* sessionStorage unavailable */
  }
  window.dispatchEvent(new Event(CLICK_HINT_SYNC_EVENT));
}

export function InteractiveFramework() {
  const [activeId, setActiveId] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const svgWrapperRef = useRef<HTMLDivElement>(null);
  const touchStartXRef = useRef<number>(0);
  const isMobile = !useIsDesktop();
  const hintDismissedRef = useRef(false);
  const clickHintGone = useSyncExternalStore(
    subscribeClickHintGone,
    getClickHintGoneFromStorage,
    () => false,
  );
  const [clickHintExiting, setClickHintExiting] = useState(false);
  const clickHintExitTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );
  const deckElRef = useRef<HTMLDivElement>(null);
  const [deckHeight, setDeckHeight] = useState(0);
  const methodologyBoundaryRef = useRef<HTMLDivElement>(null);
  /** Previous `getBoundingClientRect().top` of the methodology-end sentinel (px). */
  const sentinelTopPrevRef = useRef<number | null>(null);
  const scrollDirectionRef = useRef<"up" | "down" | "none">("none");
  const resamplingPeekCloseTimeoutRef = useRef<ReturnType<
    typeof setTimeout
  > | null>(null);
  const activeIdRef = useRef(activeId);
  activeIdRef.current = activeId;

  const scheduleResamplingPeek = useCallback(() => {
    if (typeof window === "undefined") return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    if (getResamplingPeekDoneFromStorage()) return;
    if (getResamplingTouchedFromStorage()) return;
    if (activeIdRef.current !== null) return;

    persistResamplingPeekDone();
    setActiveId("resampling");
    if (resamplingPeekCloseTimeoutRef.current != null) {
      clearTimeout(resamplingPeekCloseTimeoutRef.current);
    }
    resamplingPeekCloseTimeoutRef.current = setTimeout(() => {
      resamplingPeekCloseTimeoutRef.current = null;
      setActiveId((open) => (open === "resampling" ? null : open));
    }, RESAMPLING_PEEK_CLOSE_DELAY_MS);
  }, []);

  useEffect(() => {
    const onMessage = (evt: MessageEvent) => {
      if (!evt.data || evt.data.type !== "usf-maximize") return;
      const html = document.documentElement;
      const container = containerRef.current;
      if (evt.data.maximized) {
        html.style.containerType = "normal";
        if (container) {
          container.style.position = "relative";
          container.style.zIndex = "999999";
        }
      } else {
        html.style.containerType = "";
        if (container) {
          container.style.position = "";
          container.style.zIndex = "";
        }
      }
    };
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, []);

  const activeSection = sections.find((s) => s.id === activeId) ?? null;
  const hoveredSection = hoveredId
    ? (sections.find((s) => s.id === hoveredId) ?? null)
    : null;
  const primarySpotlight = activeSection ?? hoveredSection;
  const secondarySpotlight =
    activeSection && hoveredSection && hoveredId !== activeId
      ? hoveredSection
      : null;

  const [displayedSection, setDisplayedSection] = useState(activeSection);
  /** Same-frame content when opening (useEffect would leave one frame of empty `1fr`). */
  const panelSection = activeSection ?? displayedSection;

  useEffect(() => {
    if (activeSection) {
      setDisplayedSection(activeSection); // eslint-disable-line @eslint-react/set-state-in-effect -- keep content during close animation
    }
  }, [activeSection]);

  useEffect(() => {
    const el = deckElRef.current;
    if (!el) {
      setDeckHeight(0);
      return;
    }
    const ro = new ResizeObserver(() => {
      const mt = parseFloat(getComputedStyle(el).marginTop) || 0;
      setDeckHeight(el.offsetHeight + mt);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [panelSection?.id]);

  useEffect(
    () => () => {
      if (clickHintExitTimeoutRef.current != null) {
        clearTimeout(clickHintExitTimeoutRef.current);
        clickHintExitTimeoutRef.current = null;
      }
      if (resamplingPeekCloseTimeoutRef.current != null) {
        clearTimeout(resamplingPeekCloseTimeoutRef.current);
        resamplingPeekCloseTimeoutRef.current = null;
      }
    },
    [],
  );

  useEffect(() => {
    if (typeof window === "undefined") return;
    const initialY = window.scrollY;
    let lastY = window.scrollY;
    const onScroll = () => {
      const y = window.scrollY;
      if (y > lastY + 2) scrollDirectionRef.current = "down";
      else if (y < lastY - 2) scrollDirectionRef.current = "up";
      lastY = y;

      /** Short viewport: sentinel already above trigger line on load; fire after a small scroll down. */
      if (scrollDirectionRef.current !== "down" || y <= initialY + 24) return;
      const boundary = methodologyBoundaryRef.current;
      if (!boundary) return;
      const line = window.innerHeight - RESAMPLING_PEEK_FROM_VIEWPORT_BOTTOM_PX;
      const top = boundary.getBoundingClientRect().top;
      if (top <= line && top > -80) {
        scheduleResamplingPeek();
      }
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [scheduleResamplingPeek]);

  useEffect(() => {
    if (typeof IntersectionObserver === "undefined") return;
    const el = methodologyBoundaryRef.current;
    if (!el) return;

    const triggerTopMax = () =>
      window.innerHeight - RESAMPLING_PEEK_FROM_VIEWPORT_BOTTOM_PX;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry) return;
        const top = entry.boundingClientRect.top;
        const prevTop = sentinelTopPrevRef.current;
        sentinelTopPrevRef.current = top;
        if (prevTop === null) return;

        const line = triggerTopMax();
        /** Sentinel moves up as user scrolls down; crossing `line` is methodology end + a nudge, not Results header. */
        const crossedEarlyDown =
          scrollDirectionRef.current === "down" &&
          prevTop > line &&
          top <= line;

        if (!crossedEarlyDown) return;
        scheduleResamplingPeek();
      },
      { threshold: [0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1] },
    );

    observer.observe(el);
    return () => {
      observer.disconnect();
      sentinelTopPrevRef.current = null;
      if (resamplingPeekCloseTimeoutRef.current != null) {
        clearTimeout(resamplingPeekCloseTimeoutRef.current);
        resamplingPeekCloseTimeoutRef.current = null;
        setActiveId((open) => (open === "resampling" ? null : open));
      }
    };
  }, [scheduleResamplingPeek]);

  const dismissHint = useCallback(() => {
    if (hintDismissedRef.current) return;
    if (getClickHintGoneFromStorage()) {
      hintDismissedRef.current = true;
      return;
    }
    hintDismissedRef.current = true;
    const instant =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (instant) {
      persistClickHintDismissed();
      return;
    }
    setClickHintExiting(true);
    clickHintExitTimeoutRef.current = setTimeout(() => {
      clickHintExitTimeoutRef.current = null;
      setClickHintExiting(false);
      persistClickHintDismissed();
    }, CLICK_HINT_EXIT_MS);
  }, []);

  const handleRegionClick = useCallback(
    (id: string) => {
      dismissHint();
      if (id === "resampling") persistResamplingTouched();
      if (resamplingPeekCloseTimeoutRef.current != null) {
        clearTimeout(resamplingPeekCloseTimeoutRef.current);
        resamplingPeekCloseTimeoutRef.current = null;
      }
      setActiveId((prev) => (prev === id ? null : id));
    },
    [dismissHint],
  );

  const handleTouchStart = useCallback((e: TouchEvent) => {
    touchStartXRef.current = e.touches[0].clientX;
  }, []);

  const handleTouchEnd = useCallback(
    (e: TouchEvent) => {
      if (!activeId || !isMobile) return;
      const dx = e.changedTouches[0].clientX - touchStartXRef.current;
      if (Math.abs(dx) < 50) return;

      const idx = sections.findIndex((s) => s.id === activeId);
      if (dx < 0 && idx < sections.length - 1) {
        setActiveId(sections[idx + 1].id);
      } else if (dx > 0 && idx > 0) {
        setActiveId(sections[idx - 1].id);
      }
    },
    [activeId, isMobile],
  );

  const mobileZoom = useMemo(() => {
    if (!isMobile || !activeSection)
      return { transform: "none", paddingTop: 0, paddingBottom: 0 };
    const { left, width } = activeSection.region;
    const rTop = activeSection.region.top ?? TOP_ROW_HEIGHT_PCT;
    const rHeight = activeSection.region.height ?? 100 - TOP_ROW_HEIGHT_PCT;
    const centerX = left + width / 2;
    const centerY = rTop + rHeight / 2;
    const scaleFactor = Math.min(90 / width, 3);
    const translateX = 50 - centerX;
    const translateY = 50 - centerY;
    const SVG_ASPECT = 528 / 1728;
    const overflow = Math.max(
      0,
      SVG_ASPECT * ((scaleFactor * rHeight) / 2 - 50),
    );
    return {
      transform: `scale(${scaleFactor}) translate(${translateX}%, ${translateY}%)`,
      paddingTop: overflow,
      paddingBottom: overflow,
    };
  }, [isMobile, activeSection]);

  return (
    <div ref={containerRef} className="not-prose">
      {!clickHintGone && (
        <div
          className="grid transition-[grid-template-rows] duration-500 ease-in-out motion-reduce:transition-none"
          style={{
            gridTemplateRows: clickHintExiting ? "0fr" : "1fr",
          }}
        >
          <div
            className={`min-h-0 overflow-hidden transition-opacity duration-500 ease-in-out motion-reduce:transition-none ${
              clickHintExiting ? "pointer-events-none opacity-0" : "opacity-100"
            }`}
          >
            <p className="mb-3 text-sm text-zinc-500 dark:text-zinc-400">
              Click the diagram to explore
            </p>
          </div>
        </div>
      )}

      {/* Diagram, caption, accordion share full bleed; hint above stays column-aligned */}
      <div className={FRAMEWORK_DIAGRAM_BLEED_CLASS}>
        <div
          className="relative overflow-hidden rounded-lg"
          style={{
            paddingTop: mobileZoom.paddingTop
              ? `${mobileZoom.paddingTop}%`
              : undefined,
            paddingBottom: mobileZoom.paddingBottom
              ? `${mobileZoom.paddingBottom}%`
              : undefined,
            transition: "padding 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94)",
          }}
          onTouchStart={handleTouchStart}
          onTouchEnd={handleTouchEnd}
        >
          <div
            ref={svgWrapperRef}
            className="relative w-full"
            style={{
              transform: mobileZoom.transform,
              transformOrigin: "center center",
              transition: "transform 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94)",
            }}
          >
            <AdaptiveSvg
              src="framework.svg"
              alt="Unified Spherical Frontend pipeline: planar images are projected onto the unit sphere, resampled to near-uniform distribution, then processed by spherical convolution and pooling layers"
              className="h-auto w-full"
            />

            {/* Global dim overlay — SVG mask with gaussian-blurred holes */}
            <svg
              className="pointer-events-none absolute inset-0"
              style={{ width: "100%", height: "100%" }}
            >
              <defs>
                <filter
                  id="usf-mask-blur"
                  x="-20%"
                  y="-10%"
                  width="140%"
                  height="120%"
                >
                  <feGaussianBlur stdDeviation="15" />
                </filter>
                <mask id="usf-dim-mask">
                  <rect width="100%" height="100%" fill="white" />
                  <g filter="url(#usf-mask-blur)">
                    {sections.map((s) => {
                      const isPrimary = primarySpotlight?.id === s.id;
                      const isSecondary = secondarySpotlight?.id === s.id;
                      const visible = isPrimary || isSecondary;
                      const fill =
                        isPrimary && secondarySpotlight ? "#909090" : "black";
                      const rTop = s.region.top ?? TOP_ROW_HEIGHT_PCT;
                      const rHeight =
                        s.region.height ?? 100 - TOP_ROW_HEIGHT_PCT;
                      return (
                        <rect
                          key={s.id}
                          x={`${s.region.left}%`}
                          y={`${rTop}%`}
                          width={`${s.region.width}%`}
                          height={`${rHeight}%`}
                          fill={fill}
                          style={{
                            fillOpacity: visible ? 1 : 0,
                            transition:
                              "fill-opacity 0.5s ease, fill 0.4s ease",
                          }}
                        />
                      );
                    })}
                  </g>
                </mask>
              </defs>
              <rect
                width="100%"
                height="100%"
                fill="var(--dim-bg)"
                mask="url(#usf-dim-mask)"
                style={{
                  opacity: primarySpotlight ? (activeSection ? 0.75 : 0.5) : 0,
                  transition: "opacity 0.6s ease",
                }}
              />
            </svg>

            {/* Clickable region buttons (transparent, just for interaction) */}
            {sections.map((section) => (
              <button
                key={section.id}
                type="button"
                onClick={() => handleRegionClick(section.id)}
                onMouseEnter={() => setHoveredId(section.id)}
                onMouseLeave={() => setHoveredId(null)}
                className="absolute border-0 bg-transparent p-0"
                style={{
                  left: `${section.region.left}%`,
                  width: `${section.region.width}%`,
                  top: `${section.region.top ?? TOP_ROW_HEIGHT_PCT}%`,
                  height: `${section.region.height ?? 100 - TOP_ROW_HEIGHT_PCT}%`,
                  cursor: "pointer",
                }}
                aria-label={section.label}
              />
            ))}

            {/* Onboarding-style pulse; deeper violet than SVG fill #D9D2E9 (that hex reads as white on the page) */}
            {!clickHintGone && (
              <div
                className={`pointer-events-none absolute z-20 h-0 w-0 -translate-x-1/2 -translate-y-1/2 transition-opacity duration-500 ease-in-out motion-reduce:transition-none ${
                  clickHintExiting ? "opacity-0" : "opacity-100"
                }`}
                style={{
                  left: `${FRAMEWORK_CLICK_HINT_LEFT_PCT}%`,
                  top: `${FRAMEWORK_CLICK_HINT_TOP_PCT}%`,
                }}
                aria-hidden
              >
                <span
                  className="usf-onboarding-hotspot absolute rounded-full"
                  style={{
                    left: "50%",
                    top: "50%",
                    width: "2rem",
                    height: "2rem",
                    marginLeft: "-1rem",
                    marginTop: "-1rem",
                    transformOrigin: "50% 50%",
                    animation: clickHintExiting
                      ? "none"
                      : "usf-onboarding-pulsate 3s cubic-bezier(0.25, 1, 0.5, 1) infinite",
                  }}
                />
              </div>
            )}
          </div>
        </div>

        <style>{`
        .usf-onboarding-hotspot {
          background-color: rgba(104, 78, 148, 0.5);
        }
        [data-theme="dark"] .usf-onboarding-hotspot,
        .dark .usf-onboarding-hotspot {
          background-color: rgba(168, 138, 212, 0.45);
        }
        /* codepen.io/SamuelEiche/pen/zYEZEVg — .hotspot--active::after @keyframes pulsate */
        @keyframes usf-onboarding-pulsate {
          0% {
            transform: scale(0.7);
            opacity: 0.6;
          }
          45% {
            transform: scale(4.5);
            opacity: 0;
          }
          100% {
            transform: scale(4.5);
            opacity: 0;
          }
        }
        @media (prefers-reduced-motion: reduce) {
          .usf-onboarding-hotspot {
            animation: none !important;
            opacity: 0.5 !important;
            transform: scale(0.7);
          }
        }
      `}</style>

        {/* Caption */}
        <div className="relative mt-3">
          <p className="text-center text-sm leading-relaxed text-zinc-500 dark:text-zinc-400">
            <strong className="text-zinc-700 dark:text-zinc-200">
              Unified Spherical Frontend.
            </strong>{" "}
            (i)&nbsp;A <span style={{ color: "#CB4154" }}>planar image</span>{" "}
            and its <span style={{ color: "#2563EB" }}>lens normal map</span>{" "}
            can be combined to form a (ii)&nbsp;
            <span style={{ color: "#7C3AED" }}>spherical image</span>. Cameras
            with different lenses produce spatially varying densities and
            distributions of pixels when projected onto the sphere. Thus, it is
            crucial to perform (iii)&nbsp;resampling before (iv)&nbsp;feeding
            into the backbone composed of spherical convolution and pooling
            layers. Optionally, the results can be (v)&nbsp;resampled back into
            the raw projected{" "}
            <span style={{ color: "#7C3AED" }}>spherical image</span> pixel
            locations, and (vi)&nbsp;unprojected back to the{" "}
            <span style={{ color: "#CB4154" }}>planar image</span> for
            downstream integration.
          </p>
          <div
            className="pointer-events-none absolute inset-0 rounded"
            style={{
              backgroundColor: "var(--dim-bg)",
              opacity: primarySpotlight ? (activeSection ? 0.75 : 0.5) : 0,
              transition: "opacity 0.6s ease",
            }}
          />
        </div>

        {/* Below caption, above accordion — stable while the panel opens (peek). */}
        <div
          ref={methodologyBoundaryRef}
          className="pointer-events-none h-px w-full max-w-full shrink-0"
          aria-hidden
        />

        {/* Accordion panel — grows downward from under the caption */}
        <div
          className="grid transition-[grid-template-rows] duration-500 ease-[cubic-bezier(0.4,0,0.2,1)] motion-reduce:transition-none"
          style={{
            gridTemplateRows: activeSection ? "1fr" : "0fr",
          }}
          onTransitionEnd={() => {
            if (!activeSection) setDisplayedSection(null);
          }}
        >
          <div className="min-h-0 overflow-hidden">
            {panelSection && (
              <div className="pt-6 pb-4">
                {panelSection.layout === "vertical" ? (
                  <div className="flex flex-col gap-6">
                    <h3 className="mb-0 text-xl font-semibold text-zinc-900 md:text-2xl dark:text-zinc-100">
                      {panelSection.label}
                    </h3>
                    <p className="text-lg leading-relaxed text-zinc-600 dark:text-zinc-400">
                      {panelSection.description}
                    </p>
                    {panelSection.extra}
                    {panelSection.cards.length > 0 && (
                      <CardDeck
                        key={panelSection.id}
                        cards={panelSection.cards}
                        sectionId={panelSection.id}
                        deckRef={deckElRef}
                        fullWidth
                        defaultCard={panelSection.defaultCard}
                      />
                    )}
                  </div>
                ) : (
                  <div className="flex flex-col gap-8 md:flex-row md:items-start">
                    <div
                      className={`flex flex-col ${panelSection.rightColumnWidth ? "min-w-0 flex-1" : "md:w-2/5"}`}
                      style={{
                        minHeight:
                          !isMobile && deckHeight > 0 ? deckHeight : undefined,
                      }}
                    >
                      <h3 className="mb-4 text-xl font-semibold text-zinc-900 md:text-2xl dark:text-zinc-100">
                        {panelSection.label}
                      </h3>
                      <div className="flex flex-1 flex-col justify-center">
                        <p className="text-lg leading-relaxed text-zinc-600 dark:text-zinc-400">
                          {panelSection.description}
                        </p>
                        {panelSection.extra}
                      </div>
                    </div>
                    {panelSection.cards.length > 0 ? (
                      <div
                        className={panelSection.rightColumnWidth ?? "md:w-3/5"}
                      >
                        <CardDeck
                          key={panelSection.id}
                          cards={panelSection.cards}
                          sectionId={panelSection.id}
                          deckRef={deckElRef}
                          defaultCard={panelSection.defaultCard}
                        />
                      </div>
                    ) : (
                      <div
                        className={`flex min-h-[200px] items-center justify-center rounded-xl bg-zinc-100 dark:bg-zinc-800 ${panelSection.rightColumnWidth ?? "md:w-3/5"}`}
                      >
                        <span className="text-sm text-zinc-400 dark:text-zinc-500">
                          Coming soon
                        </span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
