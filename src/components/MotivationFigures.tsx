import { useState } from "react";
import { AdaptiveSvg, useIsDesktop } from "./AdaptiveSvg";

interface FigureDef {
  src: string;
  label: string;
  alt: string;
}

const figures: FigureDef[] = [
  {
    src: "Tailored_Planar_Model.svg",
    label: "Lens-Specific Planar Model",
    alt: "Diagram showing how pinhole, fisheye, and 360° cameras each require separate planar models with lens-specific weights",
  },
  {
    src: "Equivariance_vs_Invariance.svg",
    label: "Equivariance vs Invariance",
    alt: "Diagram contrasting rotation equivariance, where rotated inputs produce equivalently rotated outputs, with rotation invariance, where outputs remain identical regardless of input rotation",
  },
];

function DesktopLayout() {
  return (
    <div className="not-prose mx-auto flex max-w-[90%] items-stretch gap-1">
      <figure className="m-0 flex flex-[1.2963] flex-col justify-end">
        <AdaptiveSvg
          src={figures[0].src}
          alt={figures[0].alt}
          className="h-auto w-full"
        />
        <figcaption className="mt-2 text-center text-sm font-semibold text-zinc-500 dark:text-zinc-400">
          {figures[0].label}
        </figcaption>
      </figure>
      <figure className="m-0 flex flex-[1.3846] flex-col justify-between px-[8%]">
        <AdaptiveSvg
          src={figures[1].src}
          alt={figures[1].alt}
          className="h-auto w-full"
        />
        <figcaption className="mt-2 text-center text-sm font-semibold text-zinc-500 dark:text-zinc-400">
          {figures[1].label}
        </figcaption>
      </figure>
    </div>
  );
}

function MobileLayout() {
  const [active, setActive] = useState(0);
  return (
    <div className="not-prose">
      <div className="mx-auto flex w-fit flex-wrap items-center justify-center gap-2">
        {figures.map((f, i) => (
          <button
            key={f.src}
            type="button"
            onClick={() => setActive(i)}
            className={[
              "inline-flex items-center justify-center rounded-full border border-transparent px-3 py-1.5 font-medium whitespace-nowrap transition-[background-color]",
              i === active
                ? "bg-zinc-200 dark:bg-zinc-700"
                : "hover:bg-zinc-300 dark:hover:bg-zinc-600",
            ].join(" ")}
          >
            {f.label}
          </button>
        ))}
      </div>
      <div className="mx-auto mt-4 w-[70%]">
        <AdaptiveSvg
          src={figures[active].src}
          alt={figures[active].alt}
          className="h-auto w-full"
        />
      </div>
    </div>
  );
}

export function MotivationFigures() {
  const isDesktop = useIsDesktop();
  return isDesktop ? <DesktopLayout /> : <MobileLayout />;
}
