import { useCallback, useId, useState, useSyncExternalStore } from "react";

const MD_BREAKPOINT = 768;

function subscribeDesktop(cb: () => void) {
  const mql = window.matchMedia(`(min-width: ${MD_BREAKPOINT}px)`);
  mql.addEventListener("change", cb);
  return () => mql.removeEventListener("change", cb);
}

function getDesktopSnapshot() {
  return window.innerWidth >= MD_BREAKPOINT;
}

function getServerDesktopSnapshot() {
  return true;
}

export function useIsDesktop() {
  return useSyncExternalStore(
    subscribeDesktop,
    getDesktopSnapshot,
    getServerDesktopSnapshot,
  );
}

function resolveUrl(src: string) {
  const base = import.meta.env.BASE_URL ?? "/";
  return `${base.endsWith("/") ? base : base + "/"}${src}`;
}

interface AdaptiveSvgProps {
  src: string;
  alt?: string;
  className?: string;
  /** Use <img> instead of <object> on mobile (may cause shadow artifacts). */
  mobileImg?: boolean;
}

export function AdaptiveSvg({
  src,
  alt,
  className,
  mobileImg = false,
}: AdaptiveSvgProps) {
  const isDesktop = useIsDesktop();
  const url = resolveUrl(src);
  const useObject = isDesktop || !mobileImg;
  const labelId = useId();
  const [aspectRatio, setAspectRatio] = useState<string>();

  const handleLoad = useCallback(
    (e: React.SyntheticEvent<HTMLObjectElement>) => {
      try {
        const svg = e.currentTarget.contentDocument?.documentElement;
        if (!svg) return;
        const vb = svg.getAttribute("viewBox");
        if (!vb) return;
        const parts = vb
          .trim()
          .split(/[\s,]+/)
          .map(Number);
        if (parts.length === 4 && parts[2] > 0 && parts[3] > 0) {
          setAspectRatio(`${parts[2]} / ${parts[3]}`);
        }
      } catch {
        // cross-origin — ignore
      }
    },
    [],
  );

  return (
    <div
      className={`dark:hue-rotate-180 dark:invert${className ? ` ${className}` : ""}`}
    >
      {useObject ? (
        <>
          <span id={labelId} className="sr-only">
            {alt || src}
          </span>
          <object
            data={url}
            type="image/svg+xml"
            title={alt || src}
            aria-label={alt || src}
            aria-labelledby={labelId}
            role="img"
            className="block h-auto w-full"
            style={aspectRatio ? { aspectRatio } : undefined}
            onLoad={handleLoad}
          >
            {alt || src}
          </object>
        </>
      ) : (
        <img src={url} alt={alt || ""} className="block h-full w-full" />
      )}
    </div>
  );
}
